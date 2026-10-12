"""Where did the guitar go? Look inside all 6 buckets for the worst (and best) songs.

    python diagnose_guitar_buckets.py --split validation --worst 12 --best 3

Needs a finished `evaluate_slakh.py` run for that split (it picks songs from its scores).
Use the validation split: the test split stays untouched for final scoring.

For every chosen song it reports:
  * share of the real guitar found in each bucket (drums, bass, other, vocals, guitar, piano).
    This is the same "volume dial" as step 2 of SI-SDR: 100% means the whole real guitar
    sits in that bucket, 0% means none of it does.
  * missed time: of the seconds where the real guitar is clearly playing, the share where
    the model's guitar bucket is nearly silent (more than 20 dB quieter than the real guitar).
"""
import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100


def guitar_share(bucket: np.ndarray, reference: np.ndarray) -> float:
    ref = reference.astype(np.float64) - reference.mean()
    est = bucket.astype(np.float64) - bucket.mean()
    return float(np.dot(est, ref) / np.dot(ref, ref))


def missed_time(estimate: np.ndarray, reference: np.ndarray, sample_rate: int = SAMPLE_RATE,
                playing_db: float = -40.0, silent_db: float = -20.0) -> float | None:
    n = len(reference) // sample_rate
    if n == 0:
        return None
    ref = reference[:n * sample_rate].astype(np.float64).reshape(n, sample_rate)
    est = estimate[:n * sample_rate].astype(np.float64).reshape(n, sample_rate)
    ref_e, est_e = (ref ** 2).sum(1), (est ** 2).sum(1)
    playing = ref_e > ref_e.max() * 10 ** (playing_db / 10)
    if not playing.any():
        return None
    missed = playing & (est_e < ref_e * 10 ** (silent_db / 10))
    return float(missed.sum() / playing.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--data", default="data/slakh_prepared")
    parser.add_argument("--worst", type=int, default=12)
    parser.add_argument("--best", type=int, default=3)
    args = parser.parse_args()

    import torch
    from demucs.apply import apply_model
    from prepare_slakh import load_prepared_audio
    from pretrained_separator import GuitarSeparator

    run = Path(f"runs/slakh_{args.split}_htdemucs_6s")
    rows = [json.loads(p.read_text()) for p in sorted((run / "tracks").glob("*.json"))]
    rows = sorted([r for r in rows if r["guitar_improvement_db"] is not None], key=lambda r: r["guitar_improvement_db"])
    picks = [(r, "worst") for r in rows[:args.worst]] + [(r, "best") for r in rows[::-1][:args.best]]
    records = {json.loads(l)["track_id"]: json.loads(l) for l in (Path(args.data) / f"{args.split}.jsonl").read_text().splitlines() if l.strip()}
    separator = GuitarSeparator()
    sources = separator.model.sources
    out = []
    for i, (row, label) in enumerate(picks):
        mixture, targets = load_prepared_audio(records[row["track"]])
        wave = torch.from_numpy(mixture)
        ref_mono = wave.mean(dim=0)
        mean, std = ref_mono.mean(), ref_mono.std() + 1e-8   # same normalization as pretrained_separator.py
        with torch.inference_mode():
            est = apply_model(separator.model, ((wave - mean) / std)[None], shifts=0, split=True,
                              overlap=separator.overlap, segment=separator.segment, device=separator.device,
                              progress=False, num_workers=0)[0].cpu() * std + mean
        if i == 0:  # guard: our guitar bucket must match the evaluated wrapper's output
            check = separator.separate_audio(mixture.T, SAMPLE_RATE)["guitar"].T
            guitar_bucket = est[separator.guitar_index]
            diff = float((guitar_bucket - torch.from_numpy(check)).abs().max())
            if diff > 1e-4:
                raise RuntimeError(f"Bucket extraction does not match the wrapper (max diff {diff})")
        buckets = {name: est[k].mean(dim=0).numpy() for k, name in enumerate(sources)}
        real = targets[0, 0]
        result = {"track": row["track"], "label": label, "guitar_types": row["guitar_types"],
                  "improvement_db": row["guitar_improvement_db"], "guitar_energy_share": row["guitar_energy_share"],
                  "real_guitar_share_by_bucket": {n: guitar_share(b, real) for n, b in buckets.items()},
                  "missed_time_share": missed_time(buckets["guitar"], real)}
        out.append(result)
        shares = " ".join(f"{n}={v*100:.0f}%" for n, v in result["real_guitar_share_by_bucket"].items())
        print(f"[{i+1}/{len(picks)}] {row['track']} ({label}, {row['guitar_improvement_db']:+.1f} dB): {shares} "
              f"missed={result['missed_time_share']*100:.0f}%", flush=True)
    folder = run / "diagnosis"
    folder.mkdir(exist_ok=True)
    (folder / "where_guitar_went.json").write_text(json.dumps(out, indent=2))
    lines = ["# Where did the guitar go?", "",
             "Share of the real guitar found in each of the model's 6 buckets, and how much of the time "
             "the guitar bucket stays nearly silent while the real guitar plays.", "",
             "| Song | Group | Guitar types | Improvement | " + " | ".join(sources) + " | Missed time |",
             "| --- | --- | --- | ---: | " + " | ".join("---:" for _ in sources) + " | ---: |"]
    for r in out:
        cells = " | ".join(f"{r['real_guitar_share_by_bucket'][n]*100:.0f}%" for n in sources)
        lines.append(f"| {r['track']} | {r['label']} | {', '.join(r['guitar_types'])} | {r['improvement_db']:+.1f} dB | "
                     f"{cells} | {r['missed_time_share']*100:.0f}% |")
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {folder / 'REPORT.md'}")


if __name__ == "__main__":
    main()
