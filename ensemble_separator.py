"""Run both guitar models and blend their answers.

The two models fail on different songs (on Cambridge, one wins by >1 dB on 21 songs and
loses by >1 dB on 12), so averaging them should land above both: where one model invents
something the other doesn't, the average halves the mistake; where both agree, nothing
changes. This is how the top entries on public separation leaderboards are built.

    python ensemble_separator.py song.wav --out runs/my_song --weight 0.5

`weight` is how much of the community RoFormer goes into the blend; 0.5 is an equal
average, 0 is pure HTDemucs, 1 is pure RoFormer. Backing stays `mixture - guitar`, so the
two outputs still add back up to the song.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def blend(estimates, weights) -> np.ndarray:
    """Weighted average of several guitar estimates, trimmed to their common length."""
    if not estimates:
        raise ValueError("Nothing to blend")
    if len(estimates) != len(weights):
        raise ValueError("One weight per estimate is required")
    total = float(sum(weights))
    if not np.isclose(total, 1.0):
        raise ValueError(f"Weights must sum to 1, got {total}")
    length = min(e.shape[0] for e in estimates)
    if max(e.shape[0] for e in estimates) - length > 2:
        raise ValueError("Models returned very different lengths; refusing to blend")
    stacked = np.stack([np.asarray(e[:length], dtype=np.float64) * w for e, w in zip(estimates, weights)])
    return stacked.sum(axis=0).astype(np.float32)


class EnsembleSeparator:
    """Same interface as the single models; runs each in turn and averages the guitar."""

    def __init__(self, device: str | None = None, weight: float = 0.5):
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        from pretrained_separator import GuitarSeparator
        from roformer_guitar import RoformerGuitarSeparator

        self.weight = weight
        self.members = [(GuitarSeparator(device), 1.0 - weight), (RoformerGuitarSeparator(device), weight)]
        self.name = f"ensemble_htdemucs{1 - weight:.2f}_roformer{weight:.2f}"
        first = self.members[0][0]
        self.device, self.sample_rate, self.channels = first.device, first.sample_rate, first.channels

    def separate_audio(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        results = [model.separate_audio(audio, sample_rate) for model, _ in self.members]
        weights = [w for _, w in self.members]
        guitar = blend([r["guitar"] for r in results], weights)
        mixture = results[0]["mixture"][:len(guitar)]
        backing = mixture - guitar
        if not (np.isfinite(guitar).all() and np.isfinite(backing).all()):
            raise RuntimeError("Blend produced non-finite audio")
        return {"mixture": mixture, "guitar": guitar, "backing": backing}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    import soundfile as sf
    audio, rate = sf.read(args.input, dtype="float32", always_2d=True)
    separator = EnsembleSeparator(args.device, args.weight)
    result = separator.separate_audio(audio, rate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in result.items():
        sf.write(out / f"{name}.wav", values, separator.sample_rate, subtype="FLOAT")
    print(json.dumps({"model": separator.name, "device": str(separator.device)}, indent=2))


if __name__ == "__main__":
    main()
