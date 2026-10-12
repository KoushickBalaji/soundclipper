"""Index MoisesDB for guitar evaluation WITHOUT writing any audio.

    python index_moisesdb.py

Reads only each song's data.json and the few-byte WAV headers, then writes one small
line per song to data/moisesdb_index.jsonl. The full song and the real guitar are
built in memory when a song is loaded (`load_moisesdb_song`), so nothing is mixed
and saved to disk:

    full song   = sum of every track in the song
    real guitar = sum of the tracks in the "guitar" stem
    backing     = full song - real guitar

MoisesDB has no official train/test split. It is real recordings under a
non-commercial licence, so here it is used only for testing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

GUITAR_TYPES = {"acoustic_guitar": "acoustic", "clean_electric_guitar": "clean electric",
                "distorted_electric_guitar": "distorted electric"}


def wav_header(path: Path) -> dict:
    """Sample rate, channels, bit depth and frame count from a RIFF/RF64 WAV header."""
    with open(path, "rb") as f:
        riff = f.read(12)
        if riff[:4] not in (b"RIFF", b"RF64") or riff[8:12] != b"WAVE":
            raise ValueError(f"Not a WAV file: {path}")
        fmt, data_size, ds64_data = None, None, None
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            cid, size = head[:4], struct.unpack("<I", head[4:])[0]
            if cid == b"ds64":
                body = f.read(size)
                ds64_data = struct.unpack("<Q", body[8:16])[0]
                if size & 1:
                    f.read(1)
            elif cid == b"fmt ":
                body = f.read(size)
                tag, channels, rate, _, align, bits = struct.unpack("<HHIIHH", body[:16])
                fmt = {"format_tag": tag, "channels": channels, "samplerate": rate,
                       "bits": bits, "block_align": align}
                if size & 1:
                    f.read(1)
            elif cid == b"data":
                data_size = ds64_data if size == 0xFFFFFFFF and ds64_data else size
                break
            else:
                f.seek(size + (size & 1), 1)
    if fmt is None or data_size is None:
        raise ValueError(f"Incomplete WAV header: {path}")
    return {**fmt, "frames": data_size // fmt["block_align"]}


def track_path(record: dict, track: dict) -> Path:
    """Absolute path of one track, rebuilt from the index's relative paths."""
    return Path(record["folder"]) / track["relpath"]


def index_song(folder: Path) -> dict:
    meta = json.loads((folder / "data.json").read_text(encoding="utf-8"))
    tracks, guitar, problems = [], [], []
    for stem in meta["stems"]:
        for track in stem["tracks"]:
            path = folder / stem["stemName"] / f"{track['id']}.{track.get('extension', 'wav')}"
            if not path.is_file():
                problems.append(f"missing file {path.name}")
                continue
            # Paths are stored RELATIVE to the song folder, and the song folder relative to the
            # project, so the index works from any machine (Claude's Linux view and Windows alike).
            info = {"stem": stem["stemName"], "track_type": track["trackType"],
                    "has_bleed": bool(track.get("has_bleed")),
                    "relpath": f"{stem['stemName']}/{path.name}", **wav_header(path)}
            tracks.append(info)
            if stem["stemName"] == "guitar":
                guitar.append(info)
    rates = {t["samplerate"] for t in tracks}
    if len(rates) > 1:
        problems.append(f"tracks differ in sample rate: {sorted(rates)}")
    if not guitar:
        problems.append("no guitar")
    # Some songs have tracks that end slightly earlier. Like the official moisesdb library,
    # shorter tracks are padded with silence at the end, so no audio is dropped.
    rate = next(iter(rates)) if tracks else None
    frames = max((t["frames"] for t in tracks), default=None)
    shortest = min((t["frames"] for t in tracks), default=None)
    return {"song_id": folder.name, "folder": folder.as_posix(), "song": meta.get("song"), "artist": meta.get("artist"),
            "genre": meta.get("genre"), "status": "ready" if not problems else "skipped",
            "problems": problems, "samplerate": rate, "frames": frames,
            "duration_seconds": frames / rate if tracks and rate else None,
            "length_gap_seconds": (frames - shortest) / rate if tracks and rate else None,
            "guitar_types": sorted({GUITAR_TYPES.get(t["track_type"], t["track_type"]) for t in guitar}),
            "guitar_has_bleed": any(t["has_bleed"] for t in guitar),
            "stems": sorted({t["stem"] for t in tracks}), "tracks": tracks}


def load_moisesdb_song(record: dict, start_frame: int = 0, frames: int = -1):
    """Return float32 (mixture[2,T], guitar[2,T]) built in memory from the song's tracks."""
    import numpy as np
    import soundfile as sf
    count = record["frames"] - start_frame if frames == -1 else frames
    if start_frame < 0 or count <= 0 or start_frame + count > record["frames"]:
        raise ValueError("Invalid crop bounds")
    mixture = np.zeros((2, count), dtype=np.float32)
    guitar = np.zeros((2, count), dtype=np.float32)
    for track in record["tracks"]:
        path = track_path(record, track)
        with sf.SoundFile(path) as reader:
            if reader.samplerate != record["samplerate"] or len(reader) != track["frames"]:
                raise ValueError(f"Audio changed since indexing: {path}")
            available = max(0, min(count, track["frames"] - start_frame))
            audio = np.zeros((reader.channels, count), dtype=np.float32)
            if available:
                reader.seek(start_frame)
                audio[:, :available] = reader.read(available, dtype="float32", always_2d=True).T
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        elif audio.shape[0] != 2:
            raise ValueError(f"Unsupported channel count in {path}")
        mixture += audio
        if track["stem"] == "guitar":
            guitar += audio
    return mixture, guitar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/raw/moisesdb/moisesdb_v0.1")
    parser.add_argument("--out", default="data/moisesdb_index.jsonl")
    args = parser.parse_args()
    songs = [index_song(folder) for folder in sorted(Path(args.root).iterdir()) if (folder / "data.json").is_file()]
    Path(args.out).write_text("".join(json.dumps(s) + "\n" for s in songs), encoding="utf-8")
    ready = [s for s in songs if s["status"] == "ready"]
    summary = {"songs": len(songs), "ready": len(ready),
               "skipped": {s["song_id"]: s["problems"] for s in songs if s["status"] != "ready"},
               "hours_ready": round(sum(s["duration_seconds"] for s in ready) / 3600, 2),
               "sample_rates": sorted({s["samplerate"] for s in ready}),
               "channel_counts": sorted({t["channels"] for s in ready for t in s["tracks"]}),
               "bit_depths": sorted({t["bits"] for s in ready for t in s["tracks"]}),
               "songs_with_uneven_track_lengths": sum(s["length_gap_seconds"] > 0 for s in ready),
               "largest_length_gap_seconds": round(max(s["length_gap_seconds"] for s in ready), 2),
               "songs_with_guitar_bleed": sum(s["guitar_has_bleed"] for s in ready),
               "guitar_type_counts": {g: sum(g in s["guitar_types"] for s in ready)
                                      for g in sorted({g for s in ready for g in s["guitar_types"]})},
               "only_one_guitar_type": {g: sum(s["guitar_types"] == [g] for s in ready)
                                        for g in sorted({g for s in ready for g in s["guitar_types"]})}}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
