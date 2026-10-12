"""Generate a TINY synthetic guitar/other dataset to smoke-test the training loop
WITHOUT needing the real Slakh data. Each track is a folder with:
    guitar.wav, other.wav, mixture.wav   (mixture = guitar + other)
Use this ONLY to validate that train.py runs end-to-end (forward/backward/eval/save).

    python make_fake_data.py --out data/fake --n-train 8 --n-valid 3 --sec 8 --sr 16000
"""
from __future__ import annotations
import argparse, math
import numpy as np
import soundfile as sf
import os, random

def _tone(sr:int, sec:float, base:float, n_harm:int=4, noise:float=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr*sec)) / sr
    y = np.zeros_like(t)
    for h in range(1, n_harm+1):
        y += (1.0/h) * np.sin(2*np.pi*base*h*t + rng.uniform(0, 2*np.pi))
    # slow amplitude modulation so it's not a pure tone
    env = 0.6 + 0.4*np.sin(2*np.pi*0.2*t)
    y = y * env + noise*rng.standard_normal(len(t))
    y /= (np.max(np.abs(y)) + 1e-8)
    return y.astype("float32")

def make(out:str, n_train:int, n_valid:int, sec:float, sr:int, seed:int):
    random.seed(seed); np.random.seed(seed)
    for split, n in (("train", n_train), ("valid", n_valid)):
        for i in range(n):
            d = os.path.join(out, split, f"FAKE{split[:3].upper()}{i:03d}")
            os.makedirs(d, exist_ok=True)
            guitar = _tone(sr, sec, base=random.uniform(80, 400), seed=100+i)
            other  = _tone(sr, sec, base=random.uniform(200, 1200), noise=0.05, seed=200+i)
            mix = guitar + other
            guitar = guitar/(np.max(np.abs(mix))+1e-8)
            other  = other /(np.max(np.abs(mix))+1e-8)
            mix = guitar + other
            sf.write(os.path.join(d, "guitar.wav"), guitar, sr, subtype="FLOAT")
            sf.write(os.path.join(d, "other.wav"),  other,  sr, subtype="FLOAT")
            sf.write(os.path.join(d, "mixture.wav"),mix,    sr, subtype="FLOAT")
    print(f"[fake] wrote dataset -> {out} (train={n_train}, valid={n_valid}, {sec}s @ {sr}Hz)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-valid", type=int, default=3)
    ap.add_argument("--sec", type=float, default=8.0)
    ap.add_argument("--sr", type=int, default=16000)
    a = ap.parse_args()
    make(a.out, a.n_train, a.n_valid, a.sec, a.sr, seed=0)
