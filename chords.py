"""Chord labels from audio and from MIDI, using the same rule for both.

The point is not to build a great chord detector — it is to ask one question fairly:
does separating the guitar make the chords easier to read than the full song does?
So audio and ground truth go through the same template matching, and only the input
changes (full song / separated guitar / true guitar stem).

A frame is labelled with the major or minor triad whose three pitch classes best match
what is sounding, or "N" when too little is sounding to call.
"""
from __future__ import annotations

import numpy as np

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR, MINOR = (0, 4, 7), (0, 3, 7)
NO_CHORD = "N"


def templates() -> tuple[np.ndarray, list[str]]:
    """24 triads as 12-dim patterns, plus their names ('G', 'Em', …)."""
    rows, names = [], []
    for root in range(12):
        for intervals, suffix in ((MAJOR, ""), (MINOR, "m")):
            row = np.zeros(12)
            for interval in intervals:
                row[(root + interval) % 12] = 1.0
            rows.append(row / np.linalg.norm(row))
            names.append(NAMES[root] + suffix)
    return np.stack(rows), names


TEMPLATES, TEMPLATE_NAMES = templates()


def label_from_chroma(chroma: np.ndarray, min_energy: float = 1e-3) -> str:
    """Best-matching triad for one 12-dim chroma vector."""
    vector = np.asarray(chroma, dtype=np.float64)
    if vector.shape != (12,):
        raise ValueError("A chroma frame must have 12 values")
    norm = np.linalg.norm(vector)
    if norm < min_energy or (vector > 0).sum() < 2:
        return NO_CHORD
    return TEMPLATE_NAMES[int(np.argmax(TEMPLATES @ (vector / norm)))]


def labels_from_midi(notes, times, weight_by_duration: bool = False) -> list[str]:
    """Ground-truth chord per frame from note events."""
    out = []
    for time in times:
        chroma = np.zeros(12)
        for note in notes:
            if note["start"] <= time < note["end"]:
                chroma[note["pitch"] % 12] += 1.0
        out.append(label_from_chroma(chroma, min_energy=0.5))
    return out


def labels_from_audio(audio, sample_rate: int, times, hop: int = 2048) -> list[str]:
    """Chord per frame from audio, via a chroma (pitch-class energy) analysis."""
    import librosa

    mono = np.asarray(audio, dtype=np.float32)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sample_rate, hop_length=hop)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sample_rate, hop_length=hop)
    out = []
    for time in times:
        window = (frame_times >= time - 0.25) & (frame_times < time + 0.25)
        if not window.any():
            out.append(NO_CHORD)
            continue
        out.append(label_from_chroma(np.median(chroma[:, window], axis=1), min_energy=1e-2))
    return out


def root_of(label: str) -> str:
    return NO_CHORD if label == NO_CHORD else label.rstrip("m")


def score_labels(truth: list[str], estimate: list[str]) -> dict:
    """Accuracy over frames where the ground truth has a chord at all."""
    pairs = [(t, e) for t, e in zip(truth, estimate) if t != NO_CHORD]
    if not pairs:
        return {"frames": 0, "exact": None, "root_only": None, "said_no_chord": None}
    exact = sum(t == e for t, e in pairs) / len(pairs)
    root = sum(root_of(t) == root_of(e) for t, e in pairs) / len(pairs)
    silent = sum(e == NO_CHORD for _, e in pairs) / len(pairs)
    return {"frames": len(pairs), "exact": float(exact), "root_only": float(root),
            "said_no_chord": float(silent)}


def chordal_share(notes, times) -> float:
    """How much of the time the guitar plays three or more different pitch classes."""
    if not len(times):
        return 0.0
    chords = sum(1 for time in times
                 if len({n["pitch"] % 12 for n in notes if n["start"] <= time < n["end"]}) >= 3)
    return chords / len(times)
