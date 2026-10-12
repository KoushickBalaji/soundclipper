"""Fetch the community guitar model (weights + the model code it needs).

    python fetch_community_model.py

Downloads, into the project:
  models/community/mel_band_roformer_guitar/   becruily_guitar.ckpt (45 MB) + its config
  third_party/models/bs_roformer/              mel_band_roformer.py, attend.py (model code)

Sources: huggingface.co/becruily/mel-band-roformer-guitar and ZFTurbo's
Music-Source-Separation-Training. Only these files are taken: that repository pins package
versions which would disturb the Demucs environment, so its requirements are NOT installed.
Checksums of everything downloaded are written to provenance.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

WEIGHTS = "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/"
CODE = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/"
FILES = [
    (WEIGHTS + "becruily_guitar.ckpt", "models/community/mel_band_roformer_guitar/becruily_guitar.ckpt"),
    (WEIGHTS + "config_guitar_becruily.yaml", "models/community/mel_band_roformer_guitar/config_guitar_becruily.yaml"),
    (CODE + "models/bs_roformer/mel_band_roformer.py", "third_party/models/bs_roformer/mel_band_roformer.py"),
    (CODE + "models/bs_roformer/attend.py", "third_party/models/bs_roformer/attend.py"),
]
PACKAGES = {"beartype": "beartype", "rotary_embedding_torch": "rotary_embedding_torch",
            "einops": "einops", "librosa": "librosa", "yaml": "PyYAML"}


def download(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"downloading {destination.name} ...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, open(destination, "wb") as sink:
            while chunk := response.read(4 * 1024 * 1024):
                sink.write(chunk)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"url": url, "path": str(destination), "bytes": destination.stat().st_size, "sha256": digest}


def main():
    for package in ("third_party/models/__init__.py", "third_party/models/bs_roformer/__init__.py"):
        Path(package).parent.mkdir(parents=True, exist_ok=True)
        Path(package).touch()
    provenance = [download(url, Path(path)) for url, path in FILES]
    Path("models/community/mel_band_roformer_guitar/provenance.json").write_text(
        json.dumps({"files": provenance,
                    "note": "Neither source states a licence; benchmark use only."}, indent=2))
    missing = []
    for module, package in PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    for entry in provenance:
        print(f"  {entry['path']}  {entry['bytes'] / 1e6:.1f} MB  sha256 {entry['sha256'][:12]}...")
    print("\nmissing packages: " + (" ".join(missing) if missing else "none"))
    if missing:
        print("install them with:\n  .venv\\Scripts\\python.exe -m pip install " + " ".join(missing))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
