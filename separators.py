"""Pick which separation model the harness runs. Both share one interface.

Every model here takes (frames, channels) audio at 44.1 kHz and returns a dict with
'mixture', 'guitar' and 'backing' (= mixture - guitar), so the scoring code never has to
know which model produced the numbers.
"""
from __future__ import annotations

CHOICES = ("htdemucs", "roformer", "ensemble")
RUN_SUFFIX = {"htdemucs": "htdemucs_6s", "roformer": "roformer_guitar", "ensemble": "ensemble"}


def make_separator(name: str = "htdemucs", device: str | None = None, weight: float = 0.5):
    if name == "htdemucs":
        from pretrained_separator import GuitarSeparator
        return GuitarSeparator(device)
    if name == "roformer":
        from roformer_guitar import RoformerGuitarSeparator
        return RoformerGuitarSeparator(device)
    if name == "ensemble":
        from ensemble_separator import EnsembleSeparator
        return EnsembleSeparator(device, weight)
    raise ValueError(f"Unknown model {name!r}; choose from {CHOICES}")
