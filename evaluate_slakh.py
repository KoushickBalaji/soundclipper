"""Score the pretrained htdemucs_6s guitar output on prepared Slakh2100 songs (no training).

    python evaluate_slakh.py --split test
    python evaluate_slakh.py --split validation

Full songs at native 44.1 kHz, so nothing is resampled. Per-song scores are saved
as each song finishes, so an interrupted run resumes where it stopped. Full-length
audio is NOT saved (about 200 MB per song); short listening clips are written for
the songs that improved most and least. Scores are also grouped by guitar type,
using the Slakh MIDI program numbers of each song's guitar parts.

Use `validation` for digging into failures and making decisions. Treat `test` as the
final exam: record its numbers, but don't tune anything by looking at its songs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import yaml

SAMPLE_RATE = 44100
# General MIDI guitar programs 24-31 grouped by how they sound.
GUITAR_TYPES = {24: "acoustic", 25: "acoustic", 26: "clean electric", 27: "clean electric",
                28: "muted electric", 29: "distorted electric", 30: "distorted electric",
                31: "harmonics"}
TYPE_ORDER = ["acoustic", "clean electric", "muted electric", "distorted electric",
              "harmonics", "other guitar"]


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def guitar_types(record: dict) -> list[str]:
    """Sorted, de-duplicated guitar types for the stems that make up the reference."""
    metadata = yaml.safe_load((Path(record["raw_track"]) / "metadata.yaml").read_text())
    found = set()
    for sid in record["guitar_stems"]:
        info = metadata["stems"][sid] or {}
        try:
            program = int(info.get("program_num"))
        except (TypeError, ValueError):
            program = None
        found.add(GUITAR_TYPES.get(program, "other guitar"))
    return [name for name in TYPE_ORDER if name in found]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"songs": 0, "median_db": None, "mean_db": None, "improved_share": None}
    arr = np.asarray(values, dtype=float)
    return {"songs": int(arr.size), "median_db": float(np.median(arr)),
            "mean_db": float(arr.mean()), "improved_share": float((arr > 0).mean())}


def group_by_type(rows):
    """Improvement grouped two ways: songs containing a type, and songs with only that type."""
    groups = []
    for name in TYPE_ORDER:
        contains = [r["guitar_improvement_db"] for r in rows if name in r["guitar_types"]]
        only = [r["guitar_improvement_db"] for r in rows if r["guitar_types"] == [name]]
        if contains:
            groups.append({"guitar_type": name, "contains": stats(contains), "only": stats(only)})
    return groups


def loudest_window(reference: np.ndarray, seconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Start frame of the window where the reference guitar has the most energy (1 s hops)."""
    length = int(seconds * sample_rate)
    if len(reference) <= length:
        return 0
    energy = np.concatenate([[0.0], np.cumsum(reference.astype(np.float64) ** 2)])
    starts = np.arange(0, len(reference) - length + 1, sample_rate)
    return int(starts[np.argmax(energy[starts + length] - energy[starts])])


def fmt(value, digits=2, suffix=" dB"):
    return "–" if value is None else f"{value:+.{digits}f}{suffix}"


def write_report(out: Path, split: str, summary: dict, rows: list, groups: list, clips: list):
    lines = [f"# Pretrained guitar extraction: Slakh2100 {split} split", "",
             f"Model `htdemucs_6s`, unchanged (no fine-tuning). {summary['songs_scored']} full songs, "
             f"{summary['audio_hours']:.1f} hours, native 44.1 kHz, {summary['elapsed_seconds']/60:.1f} minutes.", "",
             "## Overall", "",
             "| | Median | Mean |", "| --- | ---: | ---: |",
             f"| Guitar score (SI-SDR) | {fmt(summary['guitar_si_sdr']['median_db'])} | {fmt(summary['guitar_si_sdr']['mean_db'])} |",
             f"| Do-nothing score (full song) | {fmt(summary['mixture_si_sdr']['median_db'])} | {fmt(summary['mixture_si_sdr']['mean_db'])} |",
             f"| **Improvement** | **{fmt(summary['improvement']['median_db'])}** | **{fmt(summary['improvement']['mean_db'])}** |",
             f"| Backing score | {fmt(summary['backing_si_sdr']['median_db'])} | {fmt(summary['backing_si_sdr']['mean_db'])} |", "",
             f"Songs where the model helped (improvement above 0): "
             f"{summary['improvement']['improved_share']*100:.0f}%.", "",
             "## By guitar type", "",
             "A song can contain several guitar types. \"Contains\" counts every song with that type; "
             "\"only\" counts songs where it is the sole guitar type, which isolates its effect.", "",
             "| Guitar type | Songs (contains) | Median improvement | Helped | Songs (only) | Median improvement | Helped |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        c, o = g["contains"], g["only"]
        share = lambda s: "–" if s["improved_share"] is None else f"{s['improved_share']*100:.0f}%"
        lines.append(f"| {g['guitar_type']} | {c['songs']} | {fmt(c['median_db'])} | {share(c)} | "
                     f"{o['songs']} | {fmt(o['median_db'])} | {share(o)} |")
    ranked = sorted(rows, key=lambda r: r["guitar_improvement_db"])
    lines += ["", "## Least and most improved songs", "",
              "| Song | Guitar types | Guitar share of song energy | Guitar score | Improvement |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in ranked[:5] + ranked[-5:]:
        lines.append(f"| {r['track']} | {', '.join(r['guitar_types'])} | {r['guitar_energy_share']*100:.0f}% | "
                     f"{fmt(r['guitar_si_sdr_db'])} | {fmt(r['guitar_improvement_db'])} |")
    if clips:
        lines += ["", "## Listening clips", "",
                  "20-second windows where the real guitar is loudest, in `clips/<song>/`: "
                  "`mixture.wav`, `guitar.wav` (model), `guitar_reference.wav` (real), `backing.wav`.", ""]
        lines += [f"- {c['track']} ({c['label']}, improvement {fmt(c['improvement_db'])}, from {c['start_seconds']:.0f} s)"
                  for c in clips]
    lines += ["", "## Files", "", "`summary.json`, `scores.csv`, `tracks/<song>.json` (one per song).", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("test", "validation"), required=True)
    parser.add_argument("--data", default="data/slakh_prepared")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clips", type=int, default=3, help="songs per end (least/most improved)")
    parser.add_argument("--clip-seconds", type=float, default=20.0)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from prepare_slakh import load_prepared_audio
    from pretrained_separator import GuitarSeparator
    from baseline_babyslakh import score

    torch.set_num_threads(4)
    out = Path(args.out or f"runs/slakh_{args.split}_htdemucs_6s")
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (Path(args.data) / f"{args.split}.jsonl").read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get("status") == "ready"]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("No ready records found")
    status = {"state": "loading_model", "split": args.split, "total_songs": len(records), "completed_songs": 0}
    save_json(out / "status.json", status)
    started = time.time()
    rows = []
    try:
        separator = GuitarSeparator(args.device)
        for i, record in enumerate(records):
            path = out / "tracks" / f"{record['track_id']}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("uuid") == record["uuid"]:
                    rows.append(row)
                    continue
            status.update(state="running", current_song=record["track_id"], completed_songs=i,
                          updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(out / "status.json", status)
            t0 = time.time()
            mixture, targets = load_prepared_audio(record)
            result = separator.separate_audio(mixture.T, SAMPLE_RATE)
            estimate = result["guitar"].mean(axis=1)
            mix, guitar_ref, other_ref = mixture[0], targets[0, 0], targets[1, 0]
            if len(estimate) != len(mix):
                raise ValueError(f"Length changed for {record['track_id']}")
            guitar_sdr = score(estimate, guitar_ref)
            mixture_sdr = score(mix, guitar_ref)
            row = {"track": record["track_id"], "uuid": record["uuid"], "split": args.split,
                   "duration_sec": record["frames"] / SAMPLE_RATE,
                   "guitar_types": guitar_types(record), "guitar_stems": record["guitar_stems"],
                   "guitar_energy_share": float(np.sum(guitar_ref.astype(np.float64) ** 2) /
                                                np.sum(mix.astype(np.float64) ** 2)),
                   "guitar_si_sdr_db": guitar_sdr, "mixture_guitar_si_sdr_db": mixture_sdr,
                   "guitar_improvement_db": None if guitar_sdr is None or mixture_sdr is None else guitar_sdr - mixture_sdr,
                   "backing_si_sdr_db": score(mix - estimate, other_ref),
                   "clip_start_frame": loudest_window(guitar_ref, args.clip_seconds),
                   "seconds_elapsed": time.time() - t0}
            save_json(path, row)
            rows.append(row)
            print(f"[{i+1}/{len(records)}] {row['track']}: guitar={fmt(guitar_sdr)} "
                  f"improvement={fmt(row['guitar_improvement_db'])} types={','.join(row['guitar_types'])} "
                  f"({row['seconds_elapsed']:.1f}s)", flush=True)

        scored = [r for r in rows if r["guitar_improvement_db"] is not None]
        ranked = sorted(scored, key=lambda r: r["guitar_improvement_db"])
        picks = [(r, "least improved") for r in ranked[:args.clips]] + \
                [(r, "most improved") for r in ranked[::-1][:args.clips]]
        clips = []
        by_id = {r["track_id"]: r for r in records}
        status.update(state="writing_clips", completed_songs=len(rows))
        save_json(out / "status.json", status)
        for row, label in picks:
            record = by_id[row["track"]]
            mixture, targets = load_prepared_audio(record)
            result = separator.separate_audio(mixture.T, SAMPLE_RATE)
            s, n = row["clip_start_frame"], int(args.clip_seconds * SAMPLE_RATE)
            folder = out / "clips" / row["track"]
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / "mixture.wav", mixture.T[s:s+n], SAMPLE_RATE, subtype="FLOAT")
            sf.write(folder / "guitar.wav", result["guitar"][s:s+n], SAMPLE_RATE, subtype="FLOAT")
            sf.write(folder / "backing.wav", result["backing"][s:s+n], SAMPLE_RATE, subtype="FLOAT")
            sf.write(folder / "guitar_reference.wav", targets[0, 0, s:s+n], SAMPLE_RATE, subtype="FLOAT")
            clips.append({"track": row["track"], "label": label, "improvement_db": row["guitar_improvement_db"],
                          "start_seconds": s / SAMPLE_RATE})

        summary = {"model": separator.name, "split": args.split, "device": str(separator.device),
                   "songs_scored": len(scored), "songs_total": len(rows),
                   "audio_hours": sum(r["duration_sec"] for r in rows) / 3600,
                   "elapsed_seconds": time.time() - started,
                   "guitar_si_sdr": stats([r["guitar_si_sdr_db"] for r in scored]),
                   "mixture_si_sdr": stats([r["mixture_guitar_si_sdr_db"] for r in scored]),
                   "improvement": stats([r["guitar_improvement_db"] for r in scored]),
                   "backing_si_sdr": stats([r["backing_si_sdr_db"] for r in scored]),
                   "by_guitar_type": group_by_type(scored), "clips": clips}
        save_json(out / "summary.json", summary)
        columns = ["track", "duration_sec", "guitar_types", "guitar_energy_share", "guitar_si_sdr_db",
                   "mixture_guitar_si_sdr_db", "guitar_improvement_db", "backing_si_sdr_db", "seconds_elapsed"]
        with (out / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({**r, "guitar_types": ";".join(r["guitar_types"])})
        write_report(out, args.split, summary, scored, summary["by_guitar_type"], clips)
        status.update(state="complete", completed_songs=len(rows), elapsed_seconds=summary["elapsed_seconds"],
                      updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(out / "status.json", status)
        print(json.dumps({k: summary[k] for k in ("songs_scored", "improvement", "guitar_si_sdr")}, indent=2))
    except BaseException as error:
        status.update(state="failed", error=repr(error), completed_songs=len(rows))
        save_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
