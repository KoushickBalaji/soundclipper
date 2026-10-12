"""Pretrained guitar extraction with the official six-source HTDemucs model.

    python pretrained_separator.py song.wav --out runs/my_song

Exports 44.1 kHz stereo FLOAT WAVs. Backing is mixture minus estimated guitar,
so their sum reconstructs the resampled input without independent normalization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf
import torch


class GuitarSeparator:
    def __init__(self, device: str | None = None, cache_dir: str | Path = "models",
                 segment: float = 7.8, overlap: float = 0.25):
        cache = Path(cache_dir).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        # Keep downloaded weights inside the project, including the legacy fallback.
        os.environ["HF_HOME"] = str(cache / "huggingface")
        torch.hub.set_dir(str(cache / "torch"))
        from demucs.pretrained import get_model, REMOTE_ROOT

        self.name = "htdemucs_6s"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if segment <= 0 or not 0 <= overlap < 1:
            raise ValueError("Segment must be positive and overlap must be in [0, 1)")
        # Reuse the official checksum-validated legacy checkpoint when available.
        local_repo = cache / "torch" / "checkpoints"
        checkpoint = local_repo / "5c90dfd2-34c22ccb.th"
        if checkpoint.exists():
            if not (local_repo / f"{self.name}.yaml").exists():
                shutil.copyfile(REMOTE_ROOT / f"{self.name}.yaml", local_repo / f"{self.name}.yaml")
            model = get_model(self.name, repo=local_repo)
        else:
            # Use Windows' trusted certificate store for HF, without disabling TLS checks.
            if os.name == "nt":
                import ssl
                import httpx
                from huggingface_hub import set_client_factory
                set_client_factory(lambda: httpx.Client(verify=ssl.create_default_context(),
                                                       follow_redirects=True))
            model = get_model(self.name)
        self.model = model.eval().to(self.device)
        self.sample_rate = int(self.model.samplerate)
        self.channels = int(self.model.audio_channels)
        self.guitar_index = self.model.sources.index("guitar")
        self.segment = min(segment, self.model.max_allowed_segment)
        self.overlap = overlap

    @torch.inference_mode()
    def separate_audio(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """Accept mono [frames] or stereo [frames, channels], return stereo arrays."""
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not len(audio):
            raise ValueError("Expected non-empty mono or stereo audio")
        if sample_rate <= 0 or not np.isfinite(audio).all():
            raise ValueError("Audio must be finite and sample rate positive")
        waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
        waveform = convert_audio(waveform, sample_rate, self.sample_rate, self.channels)
        reference = waveform.mean(dim=0)
        mean = reference.mean()
        # Match the publisher's normalization, with explicit handling for tiny inputs.
        std = reference.std() + 1e-8 if reference.numel() > 1 else torch.tensor(0.0)
        if std.item() <= 1e-8:
            std = waveform.std(unbiased=False)
        if std.item() <= 1e-8:
            guitar = torch.zeros_like(waveform)
        else:
            estimates = apply_model(self.model, ((waveform - mean) / std)[None],
                                    shifts=0, split=True, overlap=self.overlap,
                                    segment=self.segment, device=self.device,
                                    progress=False, num_workers=0)[0]
            guitar = estimates[self.guitar_index].cpu() * std + mean
        backing = waveform - guitar
        if not torch.isfinite(guitar).all() or not torch.isfinite(backing).all():
            raise RuntimeError("Model produced non-finite audio")
        return {name: value.T.contiguous().numpy()
                for name, value in (("mixture", waveform), ("guitar", guitar), ("backing", backing))}

    def separate_file(self, source: str | Path, output_dir: str | Path) -> dict:
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        result = self.separate_audio(audio, sr)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, values in result.items():
            path = output / f"{name}.wav"
            sf.write(path, values, self.sample_rate, subtype="FLOAT")
            paths[name] = str(path.resolve())
        details = {"model": self.name, "device": str(self.device), "sample_rate": self.sample_rate,
                   "channels": self.channels, "sources": self.model.sources,
                   "input": str(Path(source).resolve()), "files": paths,
                   "mixture_reconstruction_max_error": float(np.max(np.abs(
                       result["mixture"] - result["guitar"] - result["backing"])))}
        (output / "separation.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--cache-dir", default="models")
    parser.add_argument("--segment", type=float, default=7.8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    separator = GuitarSeparator(args.device, args.cache_dir, args.segment)
    print(json.dumps(separator.separate_file(args.input, args.out), indent=2))


if __name__ == "__main__":
    main()
