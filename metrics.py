"""Scores for comparing an estimated guitar with the real one. Plain NumPy, no torch.

si_sdr_db  : how much of the output matches the real guitar, ignoring overall volume
             (the "volume dial" version). MVSep offers si_sdr as one of several ranking
             options; its guitar leaderboard is headed by plain SDR, so report both.
sdr_db     : the same comparison WITHOUT the volume dial, so a too-loud or too-quiet
             output is punished (MVSep's sdr). Exact agreement with their numbers is not
             claimed: channel handling, chunking and averaging are not published.

Both take mono (samples,) or multichannel (channels, samples) audio. Multichannel is
scored as one signal with ONE volume dial, so left/right balance errors still count;
averaging to mono first would hide them.
missed_share / junk-side helpers: second-by-second view of where the output fails
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
SILENT_FLOOR_DB = -100.0   # an all-silent output has no ratio to report; treat it as this bad


def _prepare(estimate, reference):
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if est.ndim > 2 or ref.ndim > 2:
        raise ValueError("Expected mono (samples,) or multichannel (channels, samples) audio")
    if est.shape != ref.shape or est.size == 0:
        raise ValueError(f"Shapes must match and be non-empty: {est.shape} vs {ref.shape}")
    if not (np.isfinite(est).all() and np.isfinite(ref).all()):
        raise ValueError("Audio must be finite")
    return est, ref


def si_sdr_db(estimate, reference) -> float | None:
    """10*log10(matching part / leftover), after removing means and the best volume match."""
    est, ref = _prepare(estimate, reference)
    # One offset and one dial for the whole signal keeps the channel balance meaningful.
    est, ref = (est - est.mean(axis=-1, keepdims=True)).ravel(), (ref - ref.mean(axis=-1, keepdims=True)).ravel()
    ref_energy = float(ref @ ref)
    if ref_energy <= EPS:
        return None                      # a silent reference has no defined score
    if float(est @ est) <= 1e-10 * ref_energy:
        return SILENT_FLOOR_DB           # output is silence: nothing right and nothing wrong
    target = (float(est @ ref) / ref_energy) * ref
    residual = est - target
    num, den = float(target @ target), float(residual @ residual)
    # The tiny EPS keeps a perfect match (or a perfectly wrong one) finite instead of infinite.
    return 10.0 * np.log10((num + EPS) / (den + EPS))


def sdr_db(estimate, reference) -> float | None:
    """10*log10(real guitar / error), keeping the output's own volume (no dial)."""
    est, ref = _prepare(estimate, reference)
    est, ref = est.ravel(), ref.ravel()
    ref_energy = float(ref @ ref)
    if ref_energy <= EPS:
        return None
    error = est - ref
    err_energy = float(error @ error)
    return 10.0 * np.log10((ref_energy + EPS) / (err_energy + EPS))


def second_profile(estimate, reference, sample_rate: int):
    """Per-second energy of the real guitar and of the estimate, aligned."""
    est, ref = _prepare(estimate, reference)
    if est.ndim == 2:                      # per-second energy of all channels together
        est, ref = est.sum(axis=0), ref.sum(axis=0)
    n = len(ref) // sample_rate
    if n == 0:
        return None, None
    ref_e = (ref[:n * sample_rate].reshape(n, sample_rate) ** 2).sum(1)
    est_e = (est[:n * sample_rate].reshape(n, sample_rate) ** 2).sum(1)
    return ref_e, est_e


def missed_share(estimate, reference, sample_rate: int, playing_db: float = -40.0,
                 silent_db: float = -20.0) -> float | None:
    """Share of the seconds where the guitar plays but the estimate is >20 dB quieter."""
    ref_e, est_e = second_profile(estimate, reference, sample_rate)
    if ref_e is None:
        return None
    playing = ref_e > ref_e.max() * 10 ** (playing_db / 10)
    if not playing.any():
        return None
    return float((playing & (est_e < ref_e * 10 ** (silent_db / 10))).sum() / playing.sum())


def peak_memory_gb() -> float | None:
    """Peak memory this process has used, so runs can report it honestly."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        # Modern Windows exposes this from kernel32 as K32GetProcessMemoryInfo; psapi is the fallback.
        for library, name in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                              (ctypes.windll.psapi, "GetProcessMemoryInfo")):
            function = getattr(library, name, None)
            if function and function(handle, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / 1e9
    except (AttributeError, OSError, ImportError):
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9
    except (ImportError, ValueError):
        return None
