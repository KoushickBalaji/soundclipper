"""Move Cambridge-MT song ZIPs out of your Downloads folder into the project.

    python collect_downloads.py                    # looks in your Downloads folder
    python collect_downloads.py --from "D:/somewhere"

Only files that are real ZIPs containing audio are moved, so unrelated downloads are
left alone. Files already in the project are skipped rather than overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile

AUDIO = (".wav", ".aif", ".aiff", ".flac")


def looks_like_a_multitrack(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(m.filename.lower().endswith(AUDIO) for m in archive.infolist())
    except (zipfile.BadZipFile, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", default=str(Path.home() / "Downloads"))
    parser.add_argument("--incoming", default="data/raw/cambridge_incoming")
    parser.add_argument("--copy", action="store_true", help="copy instead of moving")
    args = parser.parse_args()
    incoming = Path(args.incoming)
    incoming.mkdir(parents=True, exist_ok=True)
    moved = skipped = 0
    for zip_path in sorted(Path(args.source).glob("*.zip")):
        if not looks_like_a_multitrack(zip_path):
            continue
        destination = incoming / zip_path.name
        if destination.exists():
            print(f"already here: {zip_path.name}")
            skipped += 1
            continue
        (shutil.copy2 if args.copy else shutil.move)(str(zip_path), str(destination))
        print(f"{'copied' if args.copy else 'moved'}: {zip_path.name}")
        moved += 1
    print(f"\n{moved} songs added, {skipped} already there. Folder: {incoming}")


if __name__ == "__main__":
    main()
