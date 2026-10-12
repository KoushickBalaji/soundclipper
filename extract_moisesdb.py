"""Unpack moisesdb.zip into data/raw/ safely and resumably.

    python extract_moisesdb.py                 # run until finished
    python extract_moisesdb.py --max-seconds 160   # stop after ~160 s; run again to continue

Each file is written to a temporary ".part" name and renamed only after the zip's
built-in checksum (CRC) passes, so a stopped run never leaves a half-written file
that looks complete. Files already present with the right size are skipped.
Progress is written to runs/moisesdb_extract/status.json.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default="moisesdb.zip")
    parser.add_argument("--dest", default="data/raw")
    parser.add_argument("--max-seconds", type=float, default=0)
    args = parser.parse_args()
    started = time.time()
    dest = Path(args.dest).resolve()
    status_dir = Path("runs/moisesdb_extract")
    status_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        total_bytes = sum(m.file_size for m in members)
        done_bytes = done_files = 0
        for member in members:
            target = (dest / member.filename).resolve()
            if dest not in target.parents:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
            if target.exists() and target.stat().st_size == member.file_size:
                done_bytes += member.file_size
                done_files += 1
                continue
            if args.max_seconds and time.time() - started > args.max_seconds:
                break
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with archive.open(member) as source, open(part, "wb") as sink:
                shutil.copyfileobj(source, sink, 8 * 1024 * 1024)  # CRC is checked at end of read
            os.replace(part, target)
            done_bytes += member.file_size
            done_files += 1
        state = "complete" if done_files == len(members) else "partial"
        status = {"state": state, "files_done": done_files, "files_total": len(members),
                  "gb_done": round(done_bytes / 1e9, 2), "gb_total": round(total_bytes / 1e9, 2),
                  "destination": str(dest / "moisesdb"), "updated_unix": time.time()}
        (status_dir / "status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status))


if __name__ == "__main__":
    main()
