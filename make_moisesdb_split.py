"""Split MoisesDB into train / validation / test BY ARTIST, once, and write it down.

    python make_moisesdb_split.py

Songs by the same artist share a guitar sound and recording style. If one of an
artist's songs were used for training and another for testing, the test score would
flatter the model (this is called leakage), so every artist lands in exactly one group.

The split is deterministic: artists are taken longest-first and each one goes to the
group furthest below its target share of audio. Rerunning gives the same answer.
Written to data/moisesdb_split.json -- do not regenerate it once experiments start,
or later results stop being comparable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def build_split(records, targets=TARGETS):
    ready = [r for r in records if r.get("status") == "ready"]
    if not ready:
        raise ValueError("No ready songs to split")
    by_artist = defaultdict(list)
    for r in ready:
        by_artist[(r.get("artist") or f"unknown::{r['song_id']}").strip().lower()].append(r)
    total = sum(r["duration_seconds"] for r in ready)
    # Longest artist first, so the big blocks are placed while there is room to balance.
    order = sorted(by_artist.items(), key=lambda kv: (-sum(r["duration_seconds"] for r in kv[1]), kv[0]))
    seconds = {name: 0.0 for name in targets}
    artist_split = {}
    for artist, songs in order:
        deficit = {name: targets[name] * total - seconds[name] for name in targets}
        choice = max(sorted(targets), key=lambda name: deficit[name])
        artist_split[artist] = choice
        seconds[choice] += sum(r["duration_seconds"] for r in songs)
    songs_split = {r["song_id"]: artist_split[(r.get("artist") or f"unknown::{r['song_id']}").strip().lower()]
                   for r in ready}
    stats = {name: {"songs": sum(v == name for v in songs_split.values()),
                    "artists": sum(v == name for v in artist_split.values()),
                    "hours": round(seconds[name] / 3600, 2),
                    "share": round(seconds[name] / total, 3),
                    "guitar_types": {t: sum(1 for r in ready if songs_split[r["song_id"]] == name
                                            and t in r["guitar_types"])
                                     for t in sorted({t for r in ready for t in r["guitar_types"]})},
                    "songs_with_guitar_bleed": sum(1 for r in ready if songs_split[r["song_id"]] == name
                                                   and r.get("guitar_has_bleed"))}
             for name in targets}
    return {"targets": targets, "artists": artist_split, "songs": songs_split, "stats": stats,
            "total_hours": round(total / 3600, 2), "artist_count": len(by_artist)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/moisesdb_index.jsonl")
    parser.add_argument("--out", default="data/moisesdb_split.json")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split (avoid)")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists. Experiments depend on it; pass --force only on purpose.")
    records = [json.loads(line) for line in Path(args.index).read_text().splitlines() if line.strip()]
    split = build_split(records)
    out.write_text(json.dumps(split, indent=2))
    print(json.dumps({"total_hours": split["total_hours"], "artists": split["artist_count"],
                      "stats": split["stats"]}, indent=2))


if __name__ == "__main__":
    main()
