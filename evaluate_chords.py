"""Does separating the guitar make its chords easier to read? The end-task test.

    python evaluate_chords.py --songs 20

For each Slakh validation song with a chord-playing guitar, the same chord detector runs on
three versions and is scored against the guitar's own MIDI:

  full song        what you would transcribe with no separation at all
  separated guitar what SoundClipper produces today
  true guitar stem the ceiling: what perfect separation would give you

If the ceiling and the separated guitar score alike, better separation cannot help the
product much, and the effort belongs in the chord detector instead.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from chords import chordal_share, labels_from_midi, score_labels
from midi_notes import read_notes

SAMPLE_RATE = 44100
FRAME_SECONDS = 0.5


def guitar_midi_notes(record: dict) -> list[dict]:
    """Notes from every guitar stem of the song, with the song's own tempo map."""
    raw = Path(record["raw_track"])
    notes = []
    for stem in record["guitar_stems"]:
        path = raw / "MIDI" / f"{stem}.mid"
        if path.is_file():
            notes += read_notes(path, tempo_source=raw / "all_src.mid")
    return sorted(notes, key=lambda n: n["start"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/slakh_prepared")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--songs", type=int, default=20)
    parser.add_argument("--min-chordal", type=float, default=0.25,
                        help="skip songs whose guitar mostly plays single notes")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--out", default="runs/chords_slakh")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()

    import torch
    from chords import labels_from_audio
    from prepare_slakh import load_prepared_audio
    from separators import make_separator

    torch.set_num_threads(4)
    out = Path(args.out)
    (out / "songs").mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in
               (Path(args.data) / f"{args.split}.jsonl").read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get("status") == "ready"]
    separator = make_separator(args.model, args.device)
    rows, skipped = [], []
    for record in records:
        if len(rows) >= args.songs:
            break
        song_file = out / "songs" / f"{record['track_id']}.json"
        if song_file.exists():
            rows.append(json.loads(song_file.read_text()))
            continue
        notes = guitar_midi_notes(record)
        if not notes:
            skipped.append((record["track_id"], "no guitar MIDI"))
            continue
        times = np.arange(FRAME_SECONDS / 2, record["frames"] / SAMPLE_RATE, FRAME_SECONDS)
        share = chordal_share(notes, times)
        if share < args.min_chordal:
            skipped.append((record["track_id"], f"guitar plays chords only {share:.0%} of the time"))
            continue
        started = time.time()
        mixture, targets = load_prepared_audio(record)
        truth = labels_from_midi(notes, times)
        estimate = separator.separate_audio(mixture.T, SAMPLE_RATE)["guitar"].T
        conditions = {"full_song": mixture, "separated_guitar": estimate, "true_guitar_stem": targets[0]}
        row = {"track": record["track_id"], "chordal_share": float(share),
               "seconds_elapsed": None, "scores": {}}
        for name, audio in conditions.items():
            row["scores"][name] = score_labels(truth, labels_from_audio(audio, SAMPLE_RATE, times))
        row["seconds_elapsed"] = time.time() - started
        song_file.write_text(json.dumps(row, indent=2))
        rows.append(row)
        line = "  ".join(f"{name}: {row['scores'][name]['root_only']:.0%} root / "
                         f"{row['scores'][name]['exact']:.0%} exact" for name in conditions)
        print(f"[{len(rows)}/{args.songs}] {record['track_id']} (chords {share:.0%})  {line}", flush=True)

    summary = {"model": separator.name, "split": args.split, "songs": len(rows),
               "frame_seconds": FRAME_SECONDS, "skipped": skipped[:20],
               "conditions": {}}
    for name in ("full_song", "separated_guitar", "true_guitar_stem"):
        exact = [r["scores"][name]["exact"] for r in rows if r["scores"][name]["exact"] is not None]
        root = [r["scores"][name]["root_only"] for r in rows if r["scores"][name]["root_only"] is not None]
        summary["conditions"][name] = {"median_exact": float(np.median(exact)) if exact else None,
                                       "mean_exact": float(np.mean(exact)) if exact else None,
                                       "median_root_only": float(np.median(root)) if root else None,
                                       "mean_root_only": float(np.mean(root)) if root else None}
    paired = {}
    for a, b in (("full_song", "separated_guitar"), ("separated_guitar", "true_guitar_stem")):
        diff = [r["scores"][b]["root_only"] - r["scores"][a]["root_only"] for r in rows
                if r["scores"][a]["root_only"] is not None and r["scores"][b]["root_only"] is not None]
        paired[f"{b}_minus_{a}"] = {"songs": len(diff), "median": float(np.median(diff)) if diff else None,
                                    "wins": int(sum(d > 0 for d in diff))}
    summary["paired_root_only"] = paired
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["track", "chordal_share", "condition", "frames", "exact", "root_only", "said_no_chord"])
        for row in rows:
            for name, s in row["scores"].items():
                writer.writerow([row["track"], f"{row['chordal_share']:.3f}", name, s["frames"],
                                 s["exact"], s["root_only"], s["said_no_chord"]])
    lines = ["# Chord reading: does separation help?", "",
             f"{len(rows)} Slakh {args.split} songs whose guitar actually plays chords, "
             f"{FRAME_SECONDS}s frames, scored against the guitar's own MIDI.", "",
             "| Input to the chord detector | Root correct (median) | Full chord correct (median) |",
             "| --- | ---: | ---: |"]
    for name, label in (("full_song", "Full song, no separation"),
                        ("separated_guitar", "Separated guitar (SoundClipper)"),
                        ("true_guitar_stem", "True guitar stem (ceiling)")):
        c = summary["conditions"][name]
        lines.append(f"| {label} | {c['median_root_only']:.0%} | {c['median_exact']:.0%} |")
    lines += ["", "Paired per song (root accuracy):", ""]
    for name, p in paired.items():
        lines.append(f"- {name.replace('_', ' ')}: median {p['median']:+.1%}, better on {p['wins']} of {p['songs']} songs")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
