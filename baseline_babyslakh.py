"""Run pretrained HTDemucs on every BabySlakh track, save audio and baseline scores.

This is inference only: no training or weight updates. Scores are computed at the
original sample rate after averaging estimated stereo channels to match mono truth.
BabySlakh's 16 kHz input is upsampled for the model; this does not restore bandwidth.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from julius import resample_frac

from build_dataset import is_guitar
from pretrained_separator import GuitarSeparator
from train import si_sdr


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_reference(track):
    mixture, sr = sf.read(track / "mix.wav", dtype="float32", always_2d=True)
    if mixture.shape[1] != 1 or not len(mixture) or not np.isfinite(mixture).all():
        raise ValueError(f"Invalid mono BabySlakh mixture: {track}")
    mixture = mixture[:, 0]
    metadata = yaml.safe_load((track / "metadata.yaml").read_text())
    guitar = np.zeros_like(mixture)
    stems = []
    for sid, info in metadata["stems"].items():
        if not is_guitar(info or {}):
            continue
        path = track / "stems" / f"{sid}.wav"
        if not path.is_file():
            raise ValueError(f"Missing guitar reference stem: {path}")
        audio, stem_sr = sf.read(path, dtype="float32", always_2d=True)
        if stem_sr != sr or audio.shape != (len(mixture), 1) or not np.isfinite(audio).all():
            raise ValueError(f"Misaligned or invalid guitar reference: {path}")
        # Slakh stems are already normalized; applying overall_gain again is wrong.
        guitar += audio[:, 0]
        stems.append(sid)
    if not stems:
        raise ValueError(f"No guitar reference stems: {track}")
    return mixture, guitar, sr, stems


def score(estimate, reference):
    ref = torch.from_numpy(np.ascontiguousarray(reference)).double()
    if (ref - ref.mean()).square().sum().item() <= 1e-8:
        return None
    est = torch.from_numpy(np.ascontiguousarray(estimate)).double()
    return float(si_sdr(est, ref).item())


def format_db(value):
    return "undefined (silent reference)" if value is None else f"{value:.2f} dB"


def mean_defined(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/babyslakh_16k")
    parser.add_argument("--out", default="runs/babyslakh_htdemucs_6s")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    tracks = sorted(Path(args.data).glob("Track*"))
    if args.limit:
        tracks = tracks[:args.limit]
    if not tracks:
        raise ValueError("No BabySlakh tracks found")
    status_file = output / "status.json"
    results = []
    started = time.time()
    status = {"state": "loading_model", "total_tracks": len(tracks), "completed_tracks": 0}
    save_json(status_file, status)
    try:
        separator = GuitarSeparator(args.device)
        print(f"[model] {separator.name} on {separator.device}, {separator.sample_rate} Hz, "
              f"sources={separator.model.sources}", flush=True)
        checkpoint = Path("models/torch/checkpoints/5c90dfd2-34c22ccb.th")
        provenance = {"name": separator.name, "sources": separator.model.sources,
                      "device": str(separator.device), "sample_rate": separator.sample_rate,
                      "channels": separator.channels, "segment": separator.segment,
                      "shifts": 0, "overlap": separator.overlap,
                      "torch_version": torch.__version__,
                      "demucs_version": importlib.metadata.version("demucs"),
                      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                      if checkpoint.exists() else None}
        for i, track in enumerate(tracks):
            status.update(state="running", current_track=track.name, completed_tracks=i,
                          updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(status_file, status)
            track_output = output / track.name
            track_output.mkdir(exist_ok=True)
            t0 = time.time()
            mixture, guitar_reference, sr, guitar_stems = load_reference(track)
            extracted = separator.separate_audio(mixture, sr)
            for name, values in extracted.items():
                sf.write(track_output / f"{name}.wav", values, separator.sample_rate, subtype="FLOAT")
            # Native-rate, mono truth for comparisons; the model never sees these stems.
            sf.write(track_output / "guitar_reference.wav", guitar_reference, sr, subtype="FLOAT")
            guitar = torch.from_numpy(extracted["guitar"].mean(axis=1).copy())
            guitar = resample_frac(guitar, separator.sample_rate, sr).numpy()
            if len(guitar) != len(mixture):
                raise ValueError(f"Resampling changed reference alignment for {track.name}")
            other_reference = mixture - guitar_reference
            guitar_sdr = score(guitar, guitar_reference)
            mixture_sdr = score(mixture, guitar_reference)
            row = {"track": track.name, "duration_sec": len(mixture) / sr,
                   "input_sample_rate": sr, "guitar_stems": guitar_stems,
                   "guitar_si_sdr_db": guitar_sdr, "mixture_guitar_si_sdr_db": mixture_sdr,
                   "guitar_improvement_db": guitar_sdr - mixture_sdr
                   if guitar_sdr is not None and mixture_sdr is not None else None,
                   "backing_si_sdr_db": score(mixture - guitar, other_reference),
                   "reconstruction_max_error": float(np.max(np.abs(
                       extracted["guitar"] + extracted["backing"] - extracted["mixture"]))),
                   "seconds_elapsed": time.time() - t0,
                   "files": {name: str((track_output / f"{name}.wav").resolve())
                             for name in [*extracted, "guitar_reference"]}}
            save_json(track_output / "metrics.json", row)
            results.append(row)
            save_json(output / "results.json", {"model": provenance, "tracks": results})
            print(f"[{i+1}/{len(tracks)}] {track.name}: guitar={format_db(guitar_sdr)}, "
                  f"improvement={format_db(row['guitar_improvement_db'])}, "
                  f"time={row['seconds_elapsed']:.1f}s", flush=True)
        metrics = ("guitar_si_sdr_db", "mixture_guitar_si_sdr_db", "guitar_improvement_db",
                   "backing_si_sdr_db")
        summary = {"model": provenance, "tracks_completed": len(results),
                   "input_audio_seconds": sum(r["duration_sec"] for r in results),
                   "elapsed_seconds": time.time() - started,
                   "means": {key: mean_defined(results, key) for key in metrics},
                   "guitar_improved_tracks": sum(r["guitar_improvement_db"] > 0 for r in results
                                                 if r["guitar_improvement_db"] is not None),
                   "gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9
                   if separator.device.type == "cuda" else None,
                   "limitations": ["BabySlakh contains synthesized music at 16 kHz.",
                                   "Upsampling does not restore missing high frequencies.",
                                   "These are baseline inference results, not fine-tuning results."]}
        save_json(output / "summary.json", summary)
        columns = ["track", "duration_sec", *metrics, "seconds_elapsed"]
        with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        status.update(state="complete", completed_tracks=len(results),
                      elapsed_seconds=summary["elapsed_seconds"],
                      updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(status_file, status)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        status.update(state="failed", completed_tracks=len(results), error=str(error))
        save_json(status_file, status)
        raise


if __name__ == "__main__":
    main()
