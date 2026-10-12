"""Build Slakh (or BabySlakh) per-track folders into a guitar/other separation set.

Input layout (per Slakh README):
    <input>/TrackXXXXX/
        mix.wav (or mix.flac)
        stems/S01.wav ... stems/SNN.wav
        metadata.yaml   # {stems: {S01: {inst_class: "Guitar", program_num: 29, ...}, ...}}

Output layout (one folder per track, 3 files), under <out>/{train,valid}/TrackXXXXX/:
        guitar.wav   # sum of all stems identified as guitar
        other.wav    # mix - guitar   (guarantees perfect reconstruction)
        (valid split also keeps mixture.wav for SI-SDR evaluation)

Guitar is identified by metadata `inst_class == "Guitar"` (covers Electric/Acoustic/
Jazz/Distortion/etc.) OR by MIDI program numbers 24-31 (guitar family). This is the
canonical Slakh rule.

Usage:
    python build_dataset.py --input data/raw --out data/separation --valid-frac 0.1
"""
from __future__ import annotations
import argparse, os, math
from pathlib import Path
import yaml
import numpy as np
import soundfile as sf

GUITAR_INST_CLASSES = {"Guitar"}          # inst_class values that are guitar
GUITAR_PROG_NUMS = set(range(24, 32))     # 24-25 clean guitar, 26-27 jazz, 28 muted,
                                          # 29 overdrive, 30 distortion, 31 electric bass? no,
                                          # 31 = clean electric. (24..31) guitar family.


def _find(base: Path, name: str, exts=(".wav", ".flac")):
    """`name` is a path RELATIVE to `base` (e.g. 'mix' or 'stems/S01')."""
    for e in exts:
        p = Path(base) / (name + e)
        if p.exists():
            return e, p
    return None, None


def is_guitar(meta: dict) -> bool:
    ic = meta.get("inst_class")
    pn = meta.get("program_num")
    if ic in GUITAR_INST_CLASSES:
        return True
    if pn is not None and int(pn) in GUITAR_PROG_NUMS:
        return True
    return False


def resample_mono(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr == target_sr:
        return x.astype(np.float32)
    # simple linear resample via soundfile
    import soundfile as _sf
    # use numpy-based interpolation (good enough for our purposes)
    n_out = int(round(len(x) * target_sr / sr))
    idx = np.linspace(0, len(x) - 1, n_out)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = (idx - i0)
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.float32)


def build_one(track: Path, target_sr: int) -> tuple[np.ndarray, np.ndarray] | None:
    with open(track / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    stems = meta.get("stems", {})
    mix_ext, mix_path = _find(track, "mix")
    if mix_path is None:
        return None
    mix, sr = sf.read(mix_path, dtype="float32", always_2d=True)
    mix = mix.mean(axis=1) if mix.ndim > 1 else mix
    mix = resample_mono(mix, sr, target_sr)

    guitar = np.zeros_like(mix)
    found_any = False
    for sid, sinfo in stems.items():
        if not is_guitar(sinfo or {}):
            continue
        ext, sp = _find(track / "stems", sid)
        if sp is None:
            continue
        s, s_sr = sf.read(sp, dtype="float32", always_2d=True)
        s = s.mean(axis=1) if s.ndim > 1 else s
        s = resample_mono(s, s_sr, target_sr)
        m = min(mix.shape[0], s.shape[0])
        guitar[:m] += s[:m]
        found_any = True
    if not found_any:
        return None
    other = mix - guitar
    return guitar, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir containing TrackXXXXX/ folders")
    ap.add_argument("--out", required=True, help="output root; will write train/ and valid/")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--target-sr", type=int, default=16000, help="target sample rate")
    ap.add_argument("--max-tracks", type=int, default=0, help="0=all (cap for smoke testing)")
    a = ap.parse_args()

    in_root = Path(a.input)
    tracks = sorted([d for d in in_root.iterdir() if d.is_dir() and d.name.startswith("Track")])
    print(f"[build] found {len(tracks)} Track* dirs under {in_root}")
    if a.max_tracks:
        tracks = tracks[:a.max_tracks]
        print(f"[build] capped to {len(tracks)}")

    n_valid = max(1, int(len(tracks) * a.valid_frac))
    # deterministic split: last n_valid -> valid
    val = tracks[-n_valid:]
    trn = tracks[:-n_valid]

    stats = {"train": 0, "valid": 0, "skipped_no_guitar": 0, "skipped_nomix": 0}
    for split, items in (("train", trn), ("valid", val)):
        out_split = Path(a.out) / split
        out_split.mkdir(parents=True, exist_ok=True)
        for trk in items:
            res = build_one(trk, a.target_sr)
            if res is None:
                stats["skipped_no_guitar" if (trk / "metadata.yaml").exists() else "skipped_nomix"] += 1
                continue
            guitar, other = res
            d = out_split / trk.name
            d.mkdir(exist_ok=True)
            sf.write(d / "guitar.wav", guitar, a.target_sr, subtype="FLOAT")
            sf.write(d / "other.wav", other, a.target_sr, subtype="FLOAT")
            if split == "valid":
                sf.write(d / "mixture.wav", (guitar + other), a.target_sr, subtype="FLOAT")
            stats[split] += 1
    print(f"[build] DONE stats={stats} sr={a.target_sr}")
    print(f"[build] -> {a.out}/train ({stats['train']})  and  {a.out}/valid ({stats['valid']})")


if __name__ == "__main__":
    main()
