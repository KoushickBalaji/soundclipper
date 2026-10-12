import os, yaml
import numpy as np
import soundfile as sf
from pathlib import Path

root = Path("data/raw/babyslakh_16k")
tracks = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("Track")])
n_with_guitar_rendered = 0
n_with_guitar = 0
energy_by_track = []
for t in tracks:
    meta = yaml.safe_load((t / "metadata.yaml").read_text())
    stems = meta.get("stems", {})
    g = [(sid, i) for sid, i in stems.items() if (i or {}).get("inst_class") == "Guitar"]
    if not g:
        continue
    n_with_guitar += 1
    energy = 0.0; nfiles = 0
    for sid, i in g:
        wav = t / "stems" / f"{sid}.wav"
        if not wav.exists():
            continue
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1: x = x.mean(axis=1)
        # rough RMS / peak
        if len(x):
            energy += float(np.sqrt(np.mean(np.square(x))))
            nfiles += 1
    is_ok = nfiles > 0
    energy_by_track.append((t.name, len(g), nfiles, is_ok, energy/max(1,nfiles)))
    if is_ok:
        n_with_guitar_rendered += 1

print(f"tracks total:                  {len(tracks)}")
print(f"tracks with Guitar stem:       {n_with_guitar}")
print(f"tracks with on-disk Guitar wav:{n_with_guitar_rendered}")
print("\nper-track guitar stems (metadata / actually-on-disk files / mean rms):")
for name, nmeta, nfiles, ok, rms in energy_by_track:
    flag = "OK " if ok else "NO "
    print(f"  {flag} {name}: {nmeta} guitar in meta, {nfiles} .wav on disk, mean rms={rms:.4f}")
