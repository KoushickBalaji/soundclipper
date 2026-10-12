"""Download Cambridge-MT song ZIPs that YOU picked, a couple at a time.

    python fetch_cambridge.py                 # fetch the next 2 unfetched URLs
    python fetch_cambridge.py --max 5

This is not a crawler. It reads the URLs you put in
data/raw/cambridge_incoming/urls.txt (one per line, '#' for comments), which you copy
from https://www.cambridge-mt.com/ms3/mtk/ in your browser. It never discovers links,
never follows them, and identifies itself honestly in the request.

Their site refuses non-browser clients (it answers 403 to plain HTTP tools), so this may
simply not work. If it 403s, download those songs in your browser instead and drop the
files in the same folder; `collect_downloads.py` moves them over for you. Please also
check their FAQ for what they allow before using these recordings for training.

Partial files are resumed, finished files are skipped, and every file is verified to be a
real ZIP before it counts as done.
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

USER_AGENT = "SoundClipper/0.1 (personal research; contact via cambridge-mt.com/ms3/mtk-faq)"


def target_name(url: str) -> str:
    """File name for a download URL, without directory tricks."""
    name = unquote(Path(urlparse(url).path).name)
    if not name.lower().endswith(".zip"):
        raise ValueError(f"Not a .zip link: {url}")
    return Path(name).name


def read_urls(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    urls, seen = [], set()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if not line.lower().startswith("https://"):
            raise ValueError(f"Only https links are accepted: {line}")
        if line not in seen:
            seen.add(line)
            urls.append(line)
    return urls


def is_complete_zip(path: Path) -> bool:
    """A truncated download has no readable central directory, so opening it fails."""
    if not path.is_file() or path.stat().st_size < 64:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def download(url: str, destination: Path, timeout: float = 60.0) -> None:
    """Download to a .part file, resuming if one is already there, then rename."""
    part = destination.with_suffix(destination.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        mode = "ab" if have and response.status == 206 else "wb"
        if mode == "wb":
            have = 0
        with open(part, mode) as sink:
            while chunk := response.read(4 * 1024 * 1024):
                sink.write(chunk)
    part.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incoming", default="data/raw/cambridge_incoming")
    parser.add_argument("--urls", default=None, help="defaults to <incoming>/urls.txt")
    parser.add_argument("--max", type=int, default=2, help="songs per run (kept small on purpose)")
    parser.add_argument("--delay", type=float, default=10.0, help="seconds to wait between songs")
    args = parser.parse_args()
    incoming = Path(args.incoming)
    incoming.mkdir(parents=True, exist_ok=True)
    urls = read_urls(Path(args.urls) if args.urls else incoming / "urls.txt")
    if not urls:
        raise SystemExit(f"No links in {incoming / 'urls.txt'}. Paste the ZIP links you want, one per line.")
    done = fetched = 0
    for url in urls:
        destination = incoming / target_name(url)
        if is_complete_zip(destination):
            done += 1
            continue
        if fetched >= args.max:
            break
        if fetched:
            time.sleep(args.delay)
        print(f"downloading {destination.name} ...", flush=True)
        try:
            download(url, destination)
        except urllib.error.HTTPError as error:
            print(f"  refused by the site ({error.code} {error.reason}). Download this one in your "
                  f"browser instead, then run collect_downloads.py.", flush=True)
            break
        except urllib.error.URLError as error:
            print(f"  could not connect: {error.reason}", flush=True)
            break
        if not is_complete_zip(destination):
            print(f"  what arrived is not a valid ZIP; leaving it for you to check: {destination}", flush=True)
            break
        fetched += 1
        print(f"  saved {destination.stat().st_size / 1e6:.0f} MB", flush=True)
    print(f"\n{done + fetched} of {len(urls)} songs are in {incoming}"
          f"{'' if fetched else ' (nothing new fetched)'}.")


if __name__ == "__main__":
    main()
