"""Draw spectrograms (full song / real guitar / model's guitar) for BabySlakh baseline tracks.

    python plot_spectrograms.py Track00011 Track00002

Picks the 10 seconds where the real guitar is loudest. Saves PNGs to
runs/babyslakh_htdemucs_6s/spectrograms/. Needs only numpy and matplotlib.
"""
import numpy as np, struct, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "babyslakh_htdemucs_6s")

def wav_info(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"RIFF"; f.read(4); assert f.read(4) == b"WAVE"
        fmt = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8: raise ValueError("no data")
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                d = f.read(size); fmt = struct.unpack("<HHIIHH", d[:16])
            elif cid == b"data":
                return fmt, f.tell(), size
            else:
                f.seek(size + (size & 1), 1)

def read(path, start=None, dur=None):
    (tag, ch, sr, _, _, bits), off, size = wav_info(path)
    assert tag == 3 and bits == 32, (tag, bits)
    n = size // (4 * ch)
    s = 0 if start is None else int(start * sr)
    k = n - s if dur is None else min(int(dur * sr), n - s)
    x = np.fromfile(path, dtype="<f4", count=k * ch, offset=off + s * 4 * ch).reshape(-1, ch)
    return x.mean(axis=1), sr

def stft_db(x, sr, win_sec=0.093, hop_sec=0.0232):
    n = int(2 ** np.round(np.log2(win_sec * sr))); hop = int(hop_sec * sr)
    w = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * w
    mag = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr); times = np.arange(len(frames)) * hop / sr + n / 2 / sr
    return 10 * np.log10(mag + 1e-12), freqs, times

def loudest_window(track, dur):
    ref, sr = read(f"{RUN}/{track}/guitar_reference.wav")
    hop = sr // 2; L = int(dur * sr)
    e = np.array([np.sum(ref[i:i+L] ** 2) for i in range(0, len(ref) - L, hop)])
    return int(np.argmax(e)) * hop / sr

def plot(track, score, improve, dur=10.0, start=None):
    if start is None: start = loudest_window(track, dur)
    rows = [("The full song (what the model hears)", "mixture.wav"),
            ("The REAL guitar (answer key)", "guitar_reference.wav"),
            ("The MODEL's guitar (its answer)", "guitar.wav")]
    specs = []
    for label, fn in rows:
        x, sr = read(f"{RUN}/{track}/{fn}", start, dur)
        specs.append((label, *stft_db(x, sr)))
    top = max(s[1].max() for s in specs)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True, constrained_layout=True)
    for ax, (label, S, f, t) in zip(axes, specs):
        keep = (f >= 40) & (f <= 16000)
        m = ax.pcolormesh(t, f[keep], S[:, keep].T, cmap="magma", vmin=top - 75, vmax=top, shading="auto", rasterized=True)
        ax.set_facecolor("black"); ax.set_yscale("log"); ax.set_ylim(40, 16000)
        ticks = [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(["50", "100", "250", "500", "1k", "2k", "4k", "8k", "16k"]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.axhline(8000, color="#7fd8ff", lw=1, ls="--")
        ax.set_ylabel("Pitch (Hz)\nlow → high")
        ax.set_title(label, loc="left", fontsize=13, fontweight="bold")
    axes[0].text(t[-1], 8600, "8 kHz: BabySlakh has nothing above this line", color="#7fd8ff", ha="right", va="bottom", fontsize=9)
    axes[-1].set_xlabel(f"Time (seconds into the song, starting at {start:.0f} s)")
    axes[-1].set_xlim(0, dur); xt = np.arange(0, dur + 0.01, 2); axes[-1].set_xticks(xt); axes[-1].set_xticklabels([f"{start + v:.0f}" for v in xt])
    cb = fig.colorbar(m, ax=axes, shrink=0.6, pad=0.01); cb.set_label("Loudness (dB, brighter = louder)")
    fig.suptitle(f"{track} — guitar score {score:+.1f} dB, improvement {improve:+.1f} dB", fontsize=15, fontweight="bold")
    os.makedirs(f"{RUN}/spectrograms", exist_ok=True)
    out = f"{RUN}/spectrograms/{track}_spectrogram.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(out, "window start", start)

import json
for tr in sys.argv[1:]:
    m = json.load(open(f"{RUN}/{tr}/metrics.json"))
    plot(tr, m["guitar_si_sdr_db"], m["guitar_improvement_db"])
