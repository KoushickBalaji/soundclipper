"""Validate Slakh2100-redux and prepare guitar/rest references without resampling.

Run ``python prepare_slakh.py --help``; see DATA_PREPARATION.md for the contract.
Only this module's derived output files are ever written. Raw audio is read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf
import yaml

SAMPLE_RATE = 44100
SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = 1
BLOCK_FRAMES = 44100 * 10


class TrackRejected(ValueError):
    """A data issue; quarantine the track rather than constructing wrong labels."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(detail)


class RunAlreadyActive(RuntimeError):
    """A second writer must not overwrite an active process's status."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def signature(path: Path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def is_guitar(info: dict) -> bool:
    # MIDI drum flags take precedence over program numbers.
    if info.get("is_drum", False):
        return False
    program = info.get("program_num")
    try:
        program = int(program) if program is not None else None
    except (ValueError, TypeError):
        raise TrackRejected("invalid_metadata", f"Invalid program number: {program!r}")
    return info.get("inst_class") == "Guitar" or program in range(24, 32)


def discover_tracks(root: Path):
    """Use the official directories verbatim, never assign new splits."""
    tracks = []
    for split in SPLITS:
        directory = root / split
        if not directory.is_dir():
            raise ValueError(f"Missing official split directory: {directory}")
        tracks.extend((split, track) for track in sorted(directory.glob("Track*")) if track.is_dir())
    if not tracks:
        raise ValueError(f"No tracks found under {root}")
    return tracks


def scan_dataset(root: Path):
    rows, groups, invalid = [], defaultdict(list), []
    estimated_frames = 0
    for split, track in discover_tracks(root):
        row = {"split": split, "track_id": track.name, "raw_track": str(track.resolve())}
        try:
            metadata = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
            uuid = metadata.get("UUID")
            if not isinstance(uuid, str) or not uuid.strip():
                raise ValueError("Missing original MIDI UUID")
            row["uuid"] = uuid
            groups[uuid].append({"split": split, "track_id": track.name})
            info = sf.info(track / "mix.flac")
            row["frames"] = info.frames
            estimated_frames += info.frames
        except Exception as exc:
            row["preflight_error"] = str(exc)
            invalid.append({**row})
        rows.append(row)
    duplicate_groups = [{"uuid": uuid, "tracks": members} for uuid, members in groups.items() if len(members) > 1]
    cross_split = [group for group in duplicate_groups if len({x["split"] for x in group["tracks"]}) > 1]
    # Quarantine all members of any cross-split group; never silently move them.
    quarantine = {group["uuid"] for group in cross_split}
    for row in rows:
        row["duplicate_quarantine"] = row.get("uuid") in quarantine
    audit = {
        "schema_version": SCHEMA_VERSION, "raw_root": str(root.resolve()), "created_at": utc_now(),
        "official_counts": dict(Counter(row["split"] for row in rows)),
        "omitted_excluded": sum(1 for p in (root / "omitted").glob("Track*") if p.is_dir()),
        "uuid_duplicates": duplicate_groups, "cross_split_uuid_duplicates": cross_split,
        "preflight_invalid": invalid, "estimated_max_output_bytes": estimated_frames * 8,
        "audio_hours": estimated_frames / SAMPLE_RATE / 3600,
        "audit_limit": "Exact original-MIDI UUID matching; does not detect near-duplicate performances with different UUIDs.",
    }
    return rows, audit


def inspect_track(track: Path):
    metadata_path = track / "metadata.yaml"
    metadata_bytes = metadata_path.read_bytes()
    metadata = yaml.safe_load(metadata_bytes)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("stems"), dict):
        raise TrackRejected("invalid_metadata", "Expected metadata with stems dictionary")
    mix_path = track / "mix.flac"
    if not mix_path.is_file():
        raise TrackRejected("missing_mixture", str(mix_path))
    mix_info = sf.info(mix_path)
    if mix_info.samplerate != SAMPLE_RATE or mix_info.channels != 1 or mix_info.frames <= 0:
        raise TrackRejected("unsupported_audio", f"Expected nonempty mono 44.1 kHz mix, got {mix_info}")
    rendered, unrendered_guitar = {}, []
    expected = set()
    for stem_id, stem_meta in metadata["stems"].items():
        if not isinstance(stem_meta, dict) or not isinstance(stem_id, str) or Path(stem_id).name != stem_id:
            raise TrackRejected("invalid_metadata", f"Invalid stem entry: {stem_id!r}")
        guitar = is_guitar(stem_meta)
        path = track / "stems" / (stem_id + ".flac")
        if stem_meta.get("audio_rendered") is not True:
            if path.exists():
                raise TrackRejected("unexpected_audio", f"Unrendered stem has audio: {stem_id}")
            if guitar:
                unrendered_guitar.append(stem_id)
            continue
        expected.add(path.name)
        if not path.is_file():
            raise TrackRejected("missing_rendered_stem", f"Missing rendered {'guitar' if guitar else 'other'} stem {stem_id}")
        info = sf.info(path)
        if (info.samplerate, info.channels, info.frames) != (SAMPLE_RATE, 1, mix_info.frames):
            raise TrackRejected("source_alignment", f"{stem_id} sample rate/channels/length differ from mix")
        rendered[stem_id] = {"path": path, "guitar": guitar, "subtype": info.subtype}
    actual = {p.name for p in (track / "stems").iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".flac")}
    if actual != expected:
        raise TrackRejected("unexpected_audio", f"Audio files absent from rendered metadata: {sorted(actual - expected)}")
    if not any(stem["guitar"] for stem in rendered.values()):
        raise TrackRejected("no_rendered_guitar", "No guitar sources have rendered audio")
    paths = [mix_path] + [stem["path"] for stem in rendered.values()]
    fingerprint = hashlib.sha256(metadata_bytes)
    for path in paths:
        fingerprint.update(json.dumps([str(path.resolve()), signature(path)], sort_keys=True).encode())
    # Each PCM16 waveform has at most one LSB of quantization error; FLOAT uses float32 precision.
    quantization_bounds = {"PCM_16": 1 / 32768, "PCM_24": 1 / 8388608, "FLOAT": 2**-23, "DOUBLE": 2**-52}
    try:
        max_residual = quantization_bounds[mix_info.subtype] + sum(quantization_bounds[x["subtype"]] for x in rendered.values()) + 2e-7
    except KeyError as exc:
        raise TrackRejected("unsupported_audio", f"Unrecognized encoding: {exc}")
    return metadata, mix_info, rendered, unrendered_guitar, fingerprint.hexdigest(), max_residual


def prepare_track(row: dict, output_root: Path):
    start = time.monotonic()
    base = {"schema_version": SCHEMA_VERSION, "split": row["split"], "track_id": row["track_id"], "uuid": row.get("uuid"), "raw_track": row["raw_track"]}
    if row.get("duplicate_quarantine"):
        return {**base, "status": "skipped", "reason": "cross_split_uuid_duplicate"}
    if row.get("preflight_error"):
        return {**base, "status": "skipped", "reason": "preflight_error", "detail": row["preflight_error"]}
    track = Path(row["raw_track"])
    output = output_root / row["split"] / row["track_id"]
    partials = [output / "guitar.partial.wav", output / "other.partial.wav"]
    try:
        metadata, mix_info, stems, unrendered_guitar, fingerprint, tolerance = inspect_track(track)
        marker = output / "record.json"
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("schema_version") == SCHEMA_VERSION and saved.get("input_fingerprint") == fingerprint:
                if all((output / name).is_file() and signature(output / name) == saved.get("output_signatures", {}).get(name) for name in ("guitar.wav", "other.wav")):
                    return {**saved, "resumed": True}
        output.mkdir(parents=True, exist_ok=True)
        stats = {"frames": 0, "mix_energy": 0.0, "guitar_energy": 0.0, "other_energy": 0.0, "residual_energy": 0.0, "residual_max": 0.0, "reconstruction_max": 0.0}
        stem_peaks = {stem_id: 0.0 for stem_id in stems}
        peaks = {"mixture": 0.0, "guitar": 0.0, "other": 0.0}
        with ExitStack() as stack:
            mixture_file = stack.enter_context(sf.SoundFile(track / "mix.flac"))
            stem_files = {stem_id: stack.enter_context(sf.SoundFile(stem["path"])) for stem_id, stem in stems.items()}
            guitar_file, other_file = [stack.enter_context(sf.SoundFile(p, mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="FLOAT")) for p in partials]
            while stats["frames"] < mix_info.frames:
                count = min(BLOCK_FRAMES, mix_info.frames - stats["frames"])
                mix = mixture_file.read(count, dtype="float32")
                if len(mix) != count or not np.all(np.isfinite(mix)):
                    raise TrackRejected("corrupt_mixture", "Short read or nonfinite mixture samples")
                total = np.zeros(count, dtype=np.float64)
                guitar64 = np.zeros(count, dtype=np.float64)
                for stem_id, reader in stem_files.items():
                    audio = reader.read(count, dtype="float32")
                    if len(audio) != count or not np.all(np.isfinite(audio)):
                        raise TrackRejected("corrupt_stem", f"Short read or nonfinite samples: {stem_id}")
                    stem_peaks[stem_id] = max(stem_peaks[stem_id], float(np.max(np.abs(audio))))
                    total += audio
                    if stems[stem_id]["guitar"]:
                        guitar64 += audio
                residual = mix.astype(np.float64) - total
                residual_max = float(np.max(np.abs(residual)))
                if residual_max > tolerance:
                    raise TrackRejected("mixture_stem_mismatch", f"At sample {stats['frames']}: sum(stems)-mix peak error {residual_max:.8g} exceeds quantization allowance {tolerance:.8g}; gain must not be reapplied")
                guitar = guitar64.astype(np.float32)
                other = mix - guitar
                guitar_file.write(guitar)
                other_file.write(other)
                for name, signal in (("mixture", mix), ("guitar", guitar), ("other", other)):
                    peaks[name] = max(peaks[name], float(np.max(np.abs(signal))))
                stats["frames"] += count
                stats["mix_energy"] += float(np.sum(np.square(mix, dtype=np.float64)))
                stats["guitar_energy"] += float(np.sum(np.square(guitar, dtype=np.float64)))
                stats["other_energy"] += float(np.sum(np.square(other, dtype=np.float64)))
                stats["residual_energy"] += float(np.sum(residual**2))
                stats["residual_max"] = max(stats["residual_max"], residual_max)
                stats["reconstruction_max"] = max(stats["reconstruction_max"], float(np.max(np.abs((guitar + other) - mix))))
        for source, key in (("mixture", "mix_energy"), ("guitar", "guitar_energy"), ("other", "other_energy")):
            if stats[key] == 0:
                raise TrackRejected("silent_" + source, f"Entire {source} source is silent")
        if stats["reconstruction_max"] > 1e-6:
            raise TrackRejected("reconstruction_failed", f"Float32 references fail reconstruction: {stats['reconstruction_max']}")
        for partial, name in zip(partials, ("guitar.wav", "other.wav")):
            partial.replace(output / name)
        record = {
            **base, "status": "ready", "input_fingerprint": fingerprint,
            "samplerate": SAMPLE_RATE, "stored_channels": 1, "model_channels": 2,
            "frames": mix_info.frames, "duration_seconds": mix_info.frames / SAMPLE_RATE,
            "mixture_path": str((track / "mix.flac").resolve()),
            "guitar_path": str((output / "guitar.wav").resolve()), "other_path": str((output / "other.wav").resolve()),
            "guitar_stems": [stem_id for stem_id, stem in stems.items() if stem["guitar"]],
            "unrendered_guitar_stems": unrendered_guitar, "silent_rendered_stems": [stem_id for stem_id, peak in stem_peaks.items() if peak == 0],
            "overall_gain_metadata": metadata.get("overall_gain"), "additional_gain_applied": 1.0,
            "source_sum_residual_rms": (stats["residual_energy"] / stats["frames"]) ** 0.5,
            "source_sum_residual_peak": stats["residual_max"], "quantization_tolerance": tolerance,
            "reconstruction_max_abs": stats["reconstruction_max"], "peaks": peaks,
            "rms": {name: (stats[key] / stats["frames"]) ** 0.5 for name, key in (("mixture", "mix_energy"), ("guitar", "guitar_energy"), ("other", "other_energy"))},
            "output_signatures": {name: signature(output / name) for name in ("guitar.wav", "other.wav")},
            "prepared_at": utc_now(), "elapsed_seconds": time.monotonic() - start,
        }
        write_json(output / "record.json", record)
        return record
    except TrackRejected as exc:
        return {**base, "status": "skipped", "reason": exc.reason, "detail": str(exc)}
    except Exception as exc:
        return {**base, "status": "error", "reason": type(exc).__name__, "detail": str(exc)}
    finally:
        for path in partials:
            if path.exists():
                path.unlink()


def load_prepared_audio(record: dict, start_frame: int = 0, frames: int = -1):
    """Return float32 mixture[2,T], targets[2,2,T] (guitar, other) at 44.1 kHz.

    Mono waveforms are duplicated, not divided by sqrt(2). Exact cropping is
    shared across all three files; no implicit padding or resampling occurs.
    """
    if record.get("status") != "ready" or record.get("samplerate") != SAMPLE_RATE:
        raise ValueError("Expected a ready 44.1 kHz manifest record")
    if start_frame < 0 or start_frame >= record["frames"] or frames == 0 or frames < -1:
        raise ValueError("Invalid crop bounds")
    count = record["frames"] - start_frame if frames == -1 else frames
    if start_frame + count > record["frames"]:
        raise ValueError("Crop exceeds the track; pad explicitly if required")
    waveforms = []
    for key in ("mixture_path", "guitar_path", "other_path"):
        with sf.SoundFile(record[key]) as reader:
            if reader.samplerate != SAMPLE_RATE or reader.channels != 1 or len(reader) != record["frames"]:
                raise ValueError(f"Audio contract changed for {key}")
            reader.seek(start_frame)
            audio = reader.read(count, dtype="float32")
        if len(audio) != count or not np.all(np.isfinite(audio)):
            raise ValueError(f"Invalid crop in {key}")
        waveforms.append(np.repeat(audio[np.newaxis, :], 2, axis=0))
    return waveforms[0], np.stack(waveforms[1:], axis=0)


class RunLock:
    """OS lock protects resumable outputs from a second preparation process."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.handle = open(self.path, "a+b")
        try:
            # Reading a byte already locked by msvcrt fails on Windows. Stat the
            # file instead, and include first-time initialization in cleanup.
            if os.fstat(self.handle.fileno()).st_size == 0:
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RunAlreadyActive("Another preparation process owns this output directory")
        return self

    def __exit__(self, *_):
        self.handle.close()


def run(args):
    raw_root, output_root, run_root = Path(args.input).resolve(), Path(args.out).resolve(), Path(args.run_dir).resolve()
    if raw_root == output_root or raw_root in output_root.parents:
        raise ValueError("Derived output must be outside the raw input tree")
    output_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    with RunLock(output_root / "prepare.lock"):
        started = time.monotonic()
        write_json(run_root / "status.json", {"state": "auditing", "pid": os.getpid(), "updated_at": utc_now()})
        rows, audit = scan_dataset(raw_root)
        if args.max_tracks_per_split:
            rows = [row for split in SPLITS for row in [x for x in rows if x["split"] == split][:args.max_tracks_per_split]]
        audit["selected_counts"] = dict(Counter(row["split"] for row in rows))
        audit["disk_free_bytes"] = shutil.disk_usage(output_root).free
        estimated = sum(row.get("frames", 0) * 8 for row in rows)
        existing = sum(p.stat().st_size for p in output_root.glob("*/*/*.wav"))
        audit["additional_output_estimate_bytes"] = max(0, estimated - existing)
        write_json(run_root / "audit.json", audit)
        if audit["disk_free_bytes"] < audit["additional_output_estimate_bytes"] + 10 * 1024**3:
            raise RuntimeError("Insufficient free disk: need derived audio estimate plus 10 GiB reserve")
        results, ready_counts, reasons = [], Counter(), Counter()
        def status(state):
            elapsed = time.monotonic() - started
            return {"state": state, "pid": os.getpid(), "updated_at": utc_now(), "completed": len(results), "total": len(rows), "ready_counts": dict(ready_counts), "skipped_or_error_reasons": dict(reasons), "elapsed_seconds": elapsed, "estimated_remaining_seconds": elapsed / len(results) * (len(rows) - len(results)) if results else None, "raw_root": str(raw_root), "output_root": str(output_root)}
        write_json(run_root / "status.json", status("running"))
        with (run_root / "events.jsonl").open("a", encoding="utf-8") as events, ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs = {executor.submit(prepare_track, row, output_root): row for row in rows}
            for job in as_completed(jobs):
                result = job.result()
                results.append(result)
                if result["status"] == "ready":
                    ready_counts[result["split"]] += 1
                else:
                    reasons[result["reason"]] += 1
                event = {"at": utc_now(), **{key: result.get(key) for key in ("split", "track_id", "status", "reason", "detail", "elapsed_seconds", "resumed")}}
                events.write(json.dumps(event) + "\n")
                events.flush()
                write_json(run_root / "status.json", status("running"))
                print(f"[{len(results)}/{len(rows)}] {result['split']}/{result['track_id']} {result['status']} {result.get('reason', '')}", flush=True)
        results.sort(key=lambda row: (SPLITS.index(row["split"]), row["track_id"]))
        for split in SPLITS:
            manifest = output_root / (split + ".jsonl")
            partial = manifest.with_suffix(".jsonl.partial")
            with partial.open("w", encoding="utf-8") as handle:
                for result in results:
                    if result["split"] == split and result["status"] == "ready":
                        handle.write(json.dumps(result, allow_nan=False) + "\n")
            partial.replace(manifest)
        skipped = [row for row in results if row["status"] != "ready"]
        write_json(run_root / "skipped.json", skipped)
        summary = {**status("complete"), "ready_audio_hours": sum(row.get("duration_seconds", 0) for row in results if row["status"] == "ready") / 3600, "error_count": sum(row["status"] == "error" for row in results), "max_reconstruction_error": max((row.get("reconstruction_max_abs", 0) for row in results), default=0), "resumed_tracks": sum(bool(row.get("resumed")) for row in results), "audit_file": str(run_root / "audit.json"), "skipped_file": str(run_root / "skipped.json")}
        write_json(run_root / "summary.json", summary)
        write_json(run_root / "status.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/slakh2100_flac_redux")
    parser.add_argument("--out", default="data/slakh_prepared")
    parser.add_argument("--run-dir", default="runs/slakh_preparation")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-tracks-per-split", type=int, default=0, help="Pilot: first N tracks from EACH official split;0=all")
    args = parser.parse_args()
    if args.max_tracks_per_split < 0:
        parser.error("--max-tracks-per-split must be >=0")
    try:
        summary = run(args)
    except RunAlreadyActive:
        # The existing writer owns status.json as well as the audio outputs.
        raise
    except Exception as exc:
        write_json(Path(args.run_dir) / "status.json", {"state": "failed", "pid": os.getpid(), "updated_at": utc_now(), "reason": str(exc)})
        raise
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
