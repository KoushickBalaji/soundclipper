"""Compare two scoring runs song by song — the only fair way to say one model is better.

    python compare_runs.py runs/cambridge_htdemucs_6s runs/cambridge_roformer_guitar

Two separate medians can differ just because two runs covered different songs. This pairs
the songs both runs scored and looks at the difference within each song, which removes
"this song is easy" from the comparison entirely.

Reported: the paired median and mean difference, how often B beats A, and a sign test
p-value (the chance of seeing a split this lopsided if the two models were equally good).
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

METRICS = ("guitar_si_sdr_db", "guitar_sdr_db", "guitar_improvement_db", "backing_si_sdr_db")


def load_rows(run: Path) -> dict:
    rows = {}
    for path in sorted((run / "tracks").glob("*.json")):
        if path.name.startswith("probe_"):
            continue
        row = json.loads(path.read_text())
        rows[row.get("song") or row.get("song_id") or row.get("track")] = row
    return rows


def sign_test(wins: int, losses: int) -> float:
    """Two-sided p-value: how surprising this win/loss split is if the models were equal."""
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(comb(n, k) for k in range(extreme + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a_rows: dict, b_rows: dict, metrics=METRICS) -> dict:
    shared = sorted(set(a_rows) & set(b_rows))
    result = {"songs_in_a": len(a_rows), "songs_in_b": len(b_rows), "songs_compared": len(shared),
              "only_in_a": sorted(set(a_rows) - set(b_rows))[:10],
              "only_in_b": sorted(set(b_rows) - set(a_rows))[:10], "metrics": {}}
    for metric in metrics:
        pairs = [(a_rows[s].get(metric), b_rows[s].get(metric)) for s in shared]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        diff = np.array([b - a for a, b in pairs], dtype=float)
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        result["metrics"][metric] = {
            "songs": len(diff), "median_a": float(np.median([a for a, _ in pairs])),
            "median_b": float(np.median([b for _, b in pairs])),
            "paired_median_gain_db": float(np.median(diff)), "paired_mean_gain_db": float(diff.mean()),
            "b_wins": wins, "a_wins": losses, "b_win_share": float(wins / len(diff)),
            "worst_regression_db": float(diff.min()), "best_gain_db": float(diff.max()),
            "sign_test_p": sign_test(wins, losses)}
    biggest = None
    if shared and "guitar_improvement_db" in result["metrics"]:
        deltas = {s: (b_rows[s].get("guitar_improvement_db") or 0) - (a_rows[s].get("guitar_improvement_db") or 0)
                  for s in shared}
        ordered = sorted(deltas.items(), key=lambda kv: kv[1])
        biggest = {"b_worst_songs": ordered[:5], "b_best_songs": ordered[-5:][::-1]}
    result["per_song_extremes"] = biggest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--out", default=None, help="where to write comparison.json/md (default: run_b)")
    args = parser.parse_args()
    a, b = Path(args.run_a), Path(args.run_b)
    result = compare(load_rows(a), load_rows(b))
    result["run_a"], result["run_b"] = str(a), str(b)
    out = Path(args.out or b)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = [f"# {b.name} vs {a.name}", "",
             f"{result['songs_compared']} songs scored by both runs.", "",
             "| Metric | A median | B median | Paired median gain | B wins | p |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for metric, m in result["metrics"].items():
        lines.append(f"| {metric} | {m['median_a']:+.2f} | {m['median_b']:+.2f} | "
                     f"**{m['paired_median_gain_db']:+.2f} dB** | {m['b_wins']}/{m['songs']} | "
                     f"{m['sign_test_p']:.4f} |")
    if result["per_song_extremes"]:
        lines += ["", "Songs where B is furthest behind A:", ""]
        lines += [f"- {song}: {delta:+.2f} dB" for song, delta in result["per_song_extremes"]["b_worst_songs"]]
        lines += ["", "Songs where B is furthest ahead:", ""]
        lines += [f"- {song}: {delta:+.2f} dB" for song, delta in result["per_song_extremes"]["b_best_songs"]]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
