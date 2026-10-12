"""Score a separation model on MoisesDB (real recordings), mixing songs in memory.

    python evaluate_moisesdb.py --splits validation test

Nothing is written except scores and a few short listening clips. Each song's full
mix and real guitar are summed from its tracks in memory, scored, and discarded.

Reports SI-SDR (the leaderboard's ranking metric, volume-independent), plain SDR
(volume counts), improvement over doing nothing, and missed time. Songs whose guitar
tracks are flagged with bleed are reported separately, because their answer key
contains a little of the other instruments.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from evaluate_slakh import fmt, group_by_type, save_json, stats
from index_moisesdb import load_moisesdb_song
from metrics import missed_share, peak_memory_gb, sdr_db, si_sdr_db
from separators import CHOICES, RUN_SUFFIX, make_separator

SAMPLE_RATE = 44100


def score_song(mixture, guitar_reference, estimate, sample_rate=SAMPLE_RATE):
    """All scores for one song. Scored in stereo, so left/right errors are not hidden.

    The backing scored here is what SoundClipper actually hands back: mixture minus the
    estimated guitar. Mono versions of the same scores are kept as a diagnostic.
    """
    mix, real, est = mixture, guitar_reference, estimate
    guitar = si_sdr_db(est, real)
    do_nothing = si_sdr_db(mix, real)
    mono_guitar = si_sdr_db(est.mean(axis=0), real.mean(axis=0))
    return {"guitar_si_sdr_db": guitar, "guitar_sdr_db": sdr_db(est, real),
            "mixture_guitar_si_sdr_db": do_nothing,
            "guitar_improvement_db": None if guitar is None or do_nothing is None else guitar - do_nothing,
            "backing_si_sdr_db": si_sdr_db(mix - est, mix - real),
            "backing_sdr_db": sdr_db(mix - est, mix - real),
            "missed_time_share": missed_share(est, real, sample_rate),
            "guitar_si_sdr_mono_db": mono_guitar,
            "stereo_penalty_db": None if guitar is None or mono_guitar is None else mono_guitar - guitar,
            "guitar_energy_share": float(np.sum(real.astype(np.float64) ** 2) /
                                         max(np.sum(mix.astype(np.float64) ** 2), 1e-20))}


def false_extraction_db(mixture, estimate) -> float:
    """For songs with NO guitar: how loud the invented 'guitar' is, relative to the song.

    -60 dB means the model correctly returned near-silence; 0 dB means it handed back
    something as loud as the song itself.
    """
    mix_energy = float(np.sum(np.asarray(mixture, dtype=np.float64) ** 2))
    est_energy = float(np.sum(np.asarray(estimate, dtype=np.float64) ** 2))
    return 10.0 * np.log10((est_energy + 1e-20) / (mix_energy + 1e-20))


def _median(values):
    kept = [v for v in values if v is not None]
    return float(np.median(kept)) if kept else None


def summarise(rows):
    scored = [r for r in rows if r["guitar_improvement_db"] is not None]
    clean = [r for r in scored if not r["guitar_has_bleed"]]
    bleed = [r for r in scored if r["guitar_has_bleed"]]
    return {"songs_scored": int(len(scored)), "songs_total": int(len(rows)),
            "hours": round(sum(r["duration_seconds"] for r in rows) / 3600, 2),
            "guitar_si_sdr": stats([r["guitar_si_sdr_db"] for r in scored]),
            "guitar_sdr": stats([r["guitar_sdr_db"] for r in scored]),
            "improvement_worst_tenth_db": (float(np.percentile([r["guitar_improvement_db"] for r in scored], 10))
                                           if scored else None),
            "songs_made_worse": int(sum(r["guitar_improvement_db"] < 0 for r in scored)),
            "stereo_penalty_median_db": _median([r.get("stereo_penalty_db") for r in scored]),
            "mixture_si_sdr": stats([r["mixture_guitar_si_sdr_db"] for r in scored]),
            "improvement": stats([r["guitar_improvement_db"] for r in scored]),
            "backing_si_sdr": stats([r["backing_si_sdr_db"] for r in scored]),
            "missed_time_share_median": _median([r.get("missed_time_share") for r in scored]),
            "improvement_clean_guitar_tracks": stats([r["guitar_improvement_db"] for r in clean]),
            "improvement_guitar_tracks_with_bleed": stats([r["guitar_improvement_db"] for r in bleed]),
            "by_guitar_type": group_by_type(scored)}


def write_report(out: Path, model: str, per_split: dict, clips: list, peak_gb, probe=None):
    lines = [f"# {model} on MoisesDB (real recordings, no training)", "",
             "Full songs at 44.1 kHz, mixed in memory from each song's tracks. "
             "SI-SDR ignores overall volume (one of MVSep's ranking options); SDR does not (their guitar leaderboard is headed by SDR).", ""]
    if peak_gb:
        lines += [f"Peak memory used: {peak_gb:.2f} GB.", ""]
    for split, block in per_split.items():
        s = block["summary"]
        lines += [f"## {split} split ({s['songs_scored']} songs, {s['hours']} h)", "",
                  "| | Median | Mean |", "| --- | ---: | ---: |",
                  f"| Guitar SI-SDR | {fmt(s['guitar_si_sdr']['median_db'])} | {fmt(s['guitar_si_sdr']['mean_db'])} |",
                  f"| Guitar SDR (volume counts) | {fmt(s['guitar_sdr']['median_db'])} | {fmt(s['guitar_sdr']['mean_db'])} |",
                  f"| Do-nothing SI-SDR | {fmt(s['mixture_si_sdr']['median_db'])} | {fmt(s['mixture_si_sdr']['mean_db'])} |",
                  f"| **Improvement** | **{fmt(s['improvement']['median_db'])}** | **{fmt(s['improvement']['mean_db'])}** |",
                  f"| Improvement, worst tenth of songs | {fmt(s['improvement_worst_tenth_db'])} | |",
                  f"| Backing SI-SDR | {fmt(s['backing_si_sdr']['median_db'])} | {fmt(s['backing_si_sdr']['mean_db'])} |", "",
                  f"Helped on {s['improvement']['improved_share']*100:.0f}% of songs. "
                  f"Median missed time: {s['missed_time_share_median']*100:.0f}%." if s['missed_time_share_median'] is not None
                  else f"Helped on {s['improvement']['improved_share']*100:.0f}% of songs.", "",
                  f"Made {s['songs_made_worse']} of {s['songs_scored']} songs worse than doing nothing. "
                  f"Median stereo penalty (mono score minus stereo score): {fmt(s['stereo_penalty_median_db'])}.", "",
                  f"Guitar tracks without bleed: {s['improvement_clean_guitar_tracks']['songs']} songs, "
                  f"median improvement {fmt(s['improvement_clean_guitar_tracks']['median_db'])}. "
                  f"With bleed: {s['improvement_guitar_tracks_with_bleed']['songs']} songs, "
                  f"median {fmt(s['improvement_guitar_tracks_with_bleed']['median_db'])}.", "",
                  "| Guitar type | Songs (contains) | Median improvement | Helped | Songs (only) | Median improvement |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for g in s["by_guitar_type"]:
            c, o = g["contains"], g["only"]
            lines.append(f"| {g['guitar_type']} | {c['songs']} | {fmt(c['median_db'])} | "
                         f"{c['improved_share']*100:.0f}% | {o['songs']} | {fmt(o['median_db'])} |")
        lines.append("")
    if probe and probe["songs"]:
        lines += [f"## Invented guitar ({probe['songs']} songs with no guitar at all)", "",
                  "How loud the model's 'guitar' is on songs that contain none, relative to the song. "
                  "Lower is better; near-silence is the right answer.", "",
                  f"- Median: **{probe['median_false_extraction_db']:+.1f} dB**",
                  f"- Worst song: **{probe['worst_false_extraction_db']:+.1f} dB**", ""]
    if clips:
        lines += ["## Listening clips", "", "20-second windows in `clips/<song>/`.", ""]
        lines += [f"- {c['song']} — {c['label']}, improvement {fmt(c['improvement_db'])}, from {c['start_seconds']:.0f} s"
                  for c in clips]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/moisesdb_index.jsonl")
    parser.add_argument("--split-file", default="data/moisesdb_split.json")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--model", default="htdemucs", choices=CHOICES)
    parser.add_argument("--weight", type=float, default=0.5,
                        help="ensemble only: share of the community model in the blend")
    parser.add_argument("--out", default=None, help="defaults to runs/moisesdb_<model>")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clips", type=int, default=3, help="clips per end, from the validation split")
    parser.add_argument("--clip-seconds", type=float, default=20.0)
    parser.add_argument("--skip-probe", action="store_true",
                        help="skip the guitar-free songs that test for invented guitar")
    args = parser.parse_args()

    import soundfile as sf
    import torch

    torch.set_num_threads(4)
    out = Path(args.out or f"runs/moisesdb_{RUN_SUFFIX[args.model]}")
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    split_map = json.loads(Path(args.split_file).read_text())["songs"]
    all_records = [json.loads(l) for l in Path(args.index).read_text().splitlines() if l.strip()]
    index = {r["song_id"]: r for r in all_records if r["status"] == "ready"}
    guitar_free = [r for r in all_records if r["problems"] == ["no guitar"]]
    todo = [(split, index[sid]) for sid, split in sorted(split_map.items())
            if split in args.splits and sid in index]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        raise ValueError("No songs selected")
    status = {"state": "loading_model", "splits": args.splits, "total_songs": len(todo), "completed_songs": 0}
    save_json(out / "status.json", status)
    started = time.time()
    rows = []
    try:
        separator = make_separator(args.model, args.device, args.weight)
        for i, (split, record) in enumerate(todo):
            path = out / "tracks" / f"{record['song_id']}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            status.update(state="running", current_song=record["song_id"], completed_songs=i,
                          updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(out / "status.json", status)
            t0 = time.time()
            mixture, guitar_reference = load_moisesdb_song(record)
            estimate = separator.separate_audio(mixture.T, SAMPLE_RATE)["guitar"].T
            row = {"song_id": record["song_id"], "split": split, "song": record.get("song"),
                   "artist": record.get("artist"), "genre": record.get("genre"),
                   "duration_seconds": record["duration_seconds"], "guitar_types": record["guitar_types"],
                   "guitar_has_bleed": record["guitar_has_bleed"],
                   **score_song(mixture, guitar_reference, estimate),
                   "seconds_elapsed": time.time() - t0}
            save_json(path, row)
            rows.append(row)
            missed = row["missed_time_share"]
            missed_text = "-" if missed is None else f"{missed * 100:.0f}%"
            print(f"[{i+1}/{len(todo)}] {record['song_id'][:8]} ({split}): "
                  f"si-sdr={fmt(row['guitar_si_sdr_db'])} sdr={fmt(row['guitar_sdr_db'])} "
                  f"improvement={fmt(row['guitar_improvement_db'])} missed={missed_text} "
                  f"({row['seconds_elapsed']:.1f}s)", flush=True)

        per_split = {split: {"summary": summarise([r for r in rows if r["split"] == split])}
                     for split in args.splits if any(r["split"] == split for r in rows)}
        clips = []
        pool = sorted([r for r in rows if r["split"] == "validation" and r["guitar_improvement_db"] is not None],
                      key=lambda r: r["guitar_improvement_db"])
        for row, label in [(r, "least improved") for r in pool[:args.clips]] + \
                          [(r, "most improved") for r in pool[::-1][:args.clips]]:
            record = index[row["song_id"]]
            mixture, guitar_reference = load_moisesdb_song(record)
            result = separator.separate_audio(mixture.T, SAMPLE_RATE)
            real = guitar_reference.mean(axis=0)
            n = int(args.clip_seconds * SAMPLE_RATE)
            energy = np.concatenate([[0.0], np.cumsum(real.astype(np.float64) ** 2)])
            starts = np.arange(0, max(1, len(real) - n), SAMPLE_RATE)
            s = int(starts[np.argmax(energy[np.minimum(starts + n, len(real))] - energy[starts])])
            folder = out / "clips" / row["song_id"][:8]
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / "mixture.wav", mixture.T[s:s+n], SAMPLE_RATE, subtype="FLOAT")
            sf.write(folder / "guitar.wav", result["guitar"][s:s+n], SAMPLE_RATE, subtype="FLOAT")
            sf.write(folder / "backing.wav", result["backing"][s:s+n], SAMPLE_RATE, subtype="FLOAT")
            sf.write(folder / "guitar_reference.wav", guitar_reference.T[s:s+n], SAMPLE_RATE, subtype="FLOAT")
            clips.append({"song": row["song_id"][:8], "label": label,
                          "improvement_db": row["guitar_improvement_db"], "start_seconds": s / SAMPLE_RATE})

        probe = []
        if not args.skip_probe:
            status.update(state="guitar_free_probe")
            save_json(out / "status.json", status)
            for record in guitar_free:
                path = out / "tracks" / f"probe_{record['song_id']}.json"
                if path.exists():
                    probe.append(json.loads(path.read_text()))
                    continue
                mixture, _ = load_moisesdb_song(record)
                estimate = separator.separate_audio(mixture.T, SAMPLE_RATE)["guitar"].T
                entry = {"song_id": record["song_id"], "artist": record.get("artist"),
                         "duration_seconds": record["duration_seconds"],
                         "false_extraction_db": false_extraction_db(mixture, estimate)}
                save_json(path, entry)
                probe.append(entry)
                print(f"[probe] {record['song_id'][:8]}: invented guitar at "
                      f"{entry['false_extraction_db']:+.1f} dB relative to the song", flush=True)

        peak = peak_memory_gb()
        summary = {"model": separator.name, "device": str(separator.device), "splits": args.splits,
                   "elapsed_seconds": time.time() - started, "peak_memory_gb": peak,
                   "per_split": per_split, "clips": clips,
                   "guitar_free_probe": {"songs": len(probe),
                                         "median_false_extraction_db": _median([p["false_extraction_db"] for p in probe]),
                                         "worst_false_extraction_db": max((p["false_extraction_db"] for p in probe),
                                                                          default=None)}}
        save_json(out / "summary.json", summary)
        columns = ["song_id", "split", "artist", "genre", "duration_seconds", "guitar_types", "guitar_has_bleed",
                   "guitar_energy_share", "guitar_si_sdr_db", "guitar_sdr_db", "mixture_guitar_si_sdr_db",
                   "guitar_improvement_db", "backing_si_sdr_db", "missed_time_share", "seconds_elapsed"]
        with (out / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({**r, "guitar_types": ";".join(r["guitar_types"])})
        write_report(out, separator.name, per_split, clips, peak, summary["guitar_free_probe"])
        status.update(state="complete", completed_songs=len(rows), elapsed_seconds=summary["elapsed_seconds"],
                      peak_memory_gb=peak, updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(out / "status.json", status)
        print(json.dumps({s: per_split[s]["summary"]["improvement"] for s in per_split}, indent=2))
        print(f"peak memory: {peak:.2f} GB" if peak else "peak memory: unknown")
    except BaseException as error:
        status.update(state="failed", error=repr(error), completed_songs=len(rows))
        save_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
