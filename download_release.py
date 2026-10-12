"""Download a release tarball from Zenodo and extract a subset of tracks.

Design notes:
- Slakh2100 full release: record 4599666, file slakh2100_flac_redux.tar.gz (~104 GB)
- BabySlakh (small, good for smoke test): record 4603870, file babyslakh_16k.tar.gz (~880 MB)
- Downloads are resumable (Range requests + .part sidecar), safe across interruptions.
- We extract only per-track subdirectories (TrackXXXXX/...) so we don't pull midis/metadata we don't need.

Usage:
    python download_release.py list --record 4603870 --file babyslakh_16k.tar.gz
    python download_release.py get  --record 4603870 --file babyslakh_16k.tar.gz --out-dir data --max 6
    python download_release.py get  --record 4599666 --file slakh2100_flac_redux.tar.gz --out-dir data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.request

ZENODO_RECORD_URL = "https://zenodo.org/records/{record}/files/{path}"


def _resolve(record, path) -> tuple[str, int]:
    req = urllib.request.Request(ZENODO_RECORD_URL.format(record=record, path=path), method="HEAD")
    with urllib.request.urlopen(req) as r:
        return r.geturl(), int(r.headers.get("Content-Length", 0))


def _download_with_resume(url: str, dest: str, expected: int) -> None:
    tmp = dest + ".part"
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expected and existing == expected:
        os.replace(tmp, dest)
        return
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    started = 0
    with urllib.request.urlopen(req) as r, open(tmp, "ab" if existing else "wb") as f:
        cr = r.headers.get("Content-Range")
        if existing and cr:
            started = int(cr.split("-")[-1].split("/")[0])
        if started != existing:
            f.seek(0); f.truncate(0); started = 0
        total = int(r.headers.get("Content-Length", 0)) + started
        done = started
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e9:8.2f} / {total/1e9:.2f} GB ({100*done/total:.1f}%)", end="", flush=True)
    sys.stderr.write("\n")
    if expected and os.path.getsize(tmp) != expected:
        raise SystemExit("size mismatch; delete .part and retry")
    os.replace(tmp, dest)


def _extract_tracks(tar_path: str, track_set: set[str], out_root: str) -> int:
    n = 0
    with tarfile.open(tar_path, "r:gz") as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if not parts or not parts[0].startswith("Track"):
                continue
            if parts[0] not in track_set:
                continue
            rel = os.path.relpath(m.name, parts[0])
            dest = os.path.join(out_root, parts[0], rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(t.extractfile(m).read())
            n += 1
            if n % 25 == 0:
                print(f"\r  extracted {n} files", end="", flush=True)
    print()
    return n


def _list_tracks(tar_path: str) -> list[str]:
    with tarfile.open(tar_path, "r:gz") as t:
        tracks = {m.name.split("/")[0] for m in t.getmembers()
                  if m.name.split("/")[0].startswith("Track")}
    return sorted(tracks)


def cmd_list(args) -> None:
    url, size = _resolve(args.record, args.file)
    print(f"URL:  {url}\nSize: {size/1e9:.2f} GB")


def cmd_get(args) -> None:
    root = args.out_dir
    raw_dir = os.path.join(root, "raw")
    tracks_dir = os.path.join(raw_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    local_tar = os.path.join(raw_dir, args.file)
    state_path = local_tar + ".state.json"

    url, expected = _resolve(args.record, args.file)
    print(f"[download] {url}")
    print(f"[download] {expected/1e9:.2f} GB -> {local_tar}")
    _download_with_resume(url, local_tar, expected)

    tracks = _list_tracks(local_tar)
    print(f"[tar] {len(tracks)} tracks total: {tracks[:5]} ... {tracks[-3:]}")
    if args.max and not args.all:
        tracks = tracks[: args.max]
        print(f"[tar] using {len(tracks)} tracks (limited by --max {args.max})")
    track_set = set(tracks)

    print(f"[extract] -> {tracks_dir}")
    n = _extract_tracks(local_tar, track_set, tracks_dir)
    print(f"[extract] {n} files across {len(track_set)} tracks")

    with open(state_path, "w") as f:
        json.dump({"record": args.record, "file": args.file, "size": expected,
                   "tracks": tracks, "extracted_files": n,
                   "tracks_root": tracks_dir}, f, indent=2)


def _add_common(p) -> None:
    p.add_argument("--record", type=int, required=True,
                   help="Zenodo record id (4599666=Slakh2100 full, 4603870=BabySlakh)")
    p.add_argument("--file", required=True, help="tarball filename within the release")
    p.add_argument("--out-dir", default="data")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    _add_common(p_list)
    p_list.set_defaults(fn=cmd_list)

    p_get = sub.add_parser("get")
    _add_common(p_get)
    p_get.add_argument("--max", type=int, default=None, help="extract at most N tracks")
    p_get.add_argument("--all", action="store_true", help="ignore --max and extract everything")
    p_get.set_defaults(fn=cmd_get)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
