"""Community guitar model (Mel-Band RoFormer, becruily) behind the same interface as ours.

    python roformer_guitar.py song.wav --out runs/my_song      # one song, for a sanity check

This is a SINGLE-PURPOSE guitar model: it outputs guitar, and backing is the song minus
that guitar — the same contract as `pretrained_separator.GuitarSeparator`, so the scoring
harness can run either model without changes.

Weights and config come from https://huggingface.co/becruily/mel-band-roformer-guitar
(45 MB), model code from ZFTurbo's Music-Source-Separation-Training. Both are fetched by
`fetch_community_model.py`. Neither states a licence, so treat this as a benchmark only,
not something to ship.

The song is processed in overlapping chunks (11.0 s, 50% overlap) and cross-faded with a
Hann window, which sums to exactly 1 at that overlap, so the joins are seamless.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("models/community/mel_band_roformer_guitar")
DEFAULT_CODE = Path("third_party")


def chunk_starts(length: int, chunk: int, hop: int) -> list[int]:
    """Start offsets covering `length` samples, always including a final full chunk."""
    if chunk <= 0 or hop <= 0 or hop > chunk:
        raise ValueError("chunk and hop must be positive with hop <= chunk")
    if length <= chunk:
        return [0]
    starts = list(range(0, length - chunk + 1, hop))
    if starts[-1] + chunk < length:
        starts.append(length - chunk)
    return starts


def fade_window(chunk: int, hop: int, first: bool, last: bool) -> np.ndarray:
    """Hann cross-fade, but flat at the very start and very end of the song.

    A plain Hann window is zero at its first and last sample, which would silence the
    beginning and end of the song. The first chunk therefore keeps full weight until its
    fade-out begins, and the last chunk keeps full weight after its fade-in ends.
    """
    window = np.hanning(chunk + 1)[:-1].astype(np.float64).copy()
    overlap = chunk - hop
    if first:
        window[:chunk - overlap] = 1.0
    if last:
        window[overlap:] = 1.0
    return window


def overlap_add(pieces, starts, length: int, chunk: int, hop: int) -> np.ndarray:
    """Cross-fade chunk outputs back into one signal, dividing out the window sum."""
    channels = pieces[0].shape[0]
    out = np.zeros((channels, length), dtype=np.float64)
    weight = np.zeros(length, dtype=np.float64)
    for index, (piece, start) in enumerate(zip(pieces, starts)):
        window = fade_window(chunk, hop, index == 0, index == len(starts) - 1)
        n = min(piece.shape[1], length - start)
        out[:, start:start + n] += piece[:, :n] * window[:n]
        weight[start:start + n] += window[:n]
    return (out / np.maximum(weight, 1e-8)).astype(np.float32)


class RoformerGuitarSeparator:
    """Same three methods the harness uses from GuitarSeparator."""

    name = "mel_band_roformer_guitar_becruily"

    def __init__(self, device: str | None = None, root: str | Path = DEFAULT_ROOT,
                 code: str | Path = DEFAULT_CODE, overlap: int = 2):
        import sys
        import torch
        import yaml

        root, code = Path(root), Path(code)
        sys.path.insert(0, str(code.resolve()))          # provides models.bs_roformer.*
        from models.bs_roformer.mel_band_roformer import MelBandRoformer

        config = yaml.unsafe_load((root / "config_guitar_becruily.yaml").read_text())
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.channels = int(config["audio"]["num_channels"])
        self.chunk = int(config["audio"]["chunk_size"])
        self.hop = self.chunk // max(1, overlap)
        model = MelBandRoformer(**config["model"])
        state = torch.load(root / "becruily_guitar.ckpt", map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint does not match the model code: "
                               f"{len(missing)} missing, {len(unexpected)} unexpected keys")
        self.model = model.eval().to(self.device)

    def separate_audio(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """Accept mono [frames] or stereo [frames, channels]; return stereo arrays."""
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not len(audio):
            raise ValueError("Expected non-empty mono or stereo audio")
        if sample_rate != self.sample_rate:
            raise ValueError(f"This model needs {self.sample_rate} Hz audio, got {sample_rate}")
        mixture = audio.T
        if mixture.shape[0] == 1:
            mixture = np.repeat(mixture, 2, axis=0)
        padded = np.pad(mixture, ((0, 0), (0, max(0, self.chunk - mixture.shape[1]))))
        starts = chunk_starts(padded.shape[1], self.chunk, self.hop)
        pieces = []
        with torch.inference_mode():
            for start in starts:
                piece = torch.from_numpy(padded[:, start:start + self.chunk]).to(self.device)
                estimate = self.model(piece[None])[0].cpu().numpy()
                pieces.append(estimate)
        guitar = overlap_add(pieces, starts, padded.shape[1], self.chunk, self.hop)[:, :mixture.shape[1]]
        backing = mixture - guitar
        if not (np.isfinite(guitar).all() and np.isfinite(backing).all()):
            raise RuntimeError("Model produced non-finite audio")
        return {"mixture": mixture.T.copy(), "guitar": guitar.T.copy(), "backing": backing.T.copy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    import soundfile as sf
    audio, rate = sf.read(args.input, dtype="float32", always_2d=True)
    separator = RoformerGuitarSeparator(args.device)
    result = separator.separate_audio(audio, rate)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in result.items():
        sf.write(out / f"{name}.wav", values, separator.sample_rate, subtype="FLOAT")
    print(json.dumps({"model": separator.name, "device": str(separator.device),
                      "chunk_seconds": separator.chunk / separator.sample_rate,
                      "files": sorted(p.name for p in out.glob('*.wav'))}, indent=2))


if __name__ == "__main__":
    main()
