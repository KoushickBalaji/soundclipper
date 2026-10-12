"""Score the model on Cambridge-MT 'Mixing Secrets' multitracks: unpack one song, score, delete.

    python cambridge_mt.py --labels-only      # guess which tracks are guitar, write them for review
    python cambridge_mt.py                    # score the songs whose labels are reviewed

You download the song ZIPs yourself from https://www.cambridge-mt.com/ms3/mtk/ and drop
them in data/raw/cambridge_incoming/. This script never downloads anything: their site
blocks automated access, and their terms of use should be read by a person before these
recordings are used for anything beyond listening and testing.

For each ZIP, one at a time:
  1. unpack into a temporary folder (outside the project),
  2. label every track as guitar / not guitar / unknown from its file name,
  3. build the full mix (sum of all tracks) and the real guitar (sum of guitar tracks) in memory,
  4. score the model against them,
  5. delete the unpacked audio. Only scores and labels are kept.

Nothing is assumed about the labels: a song with an unknown track is skipped until a
person confirms it, because a mislabelled track poisons the answer key.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

import numpy as np

from separators import CHOICES, RUN_SUFFIX, make_separator

SAMPLE_RATE = 44100
# Anything else is resampled to 44.1 kHz on the way in, so the model and the answer key match.
ALLOWED_RATES = (44100, 48000, 88200, 96000)
# Names are session-engineer shorthand: "05_ElecGtrL.wav", "12_AcousticGtr_DI.wav", "03_BassGtr.wav".
NOT_GUITAR_FIRST = re.compile(r"bass|bs\s?gtr|bgtr|sub|808|banjo|mandolin|ukulele|uke|sitar", re.I)
# Strings you play like a guitar but that neither MoisesDB nor Slakh call "guitar". Kept out of
# the answer key, but listed per song so we can see if the model files them under guitar.
GUITAR_LIKE = re.compile(r"dobro|pedal\s?steel|lap\s?steel|resonator|slide\s?gtr", re.I)
GUITAR = re.compile(r"gtr|guitar|guit\b|gits?\b|acoustic|electric", re.I)
CLEARLY_OTHER = re.compile(r"kick|snare|hat|tom|ride|crash|cymb|drum|perc|shaker|tamb|room|overhead|oh\b|"
                           r"brush|conga|bongo|djembe|cajon|"
                           r"vox|vocal|voc\b|lead\s?v|bv\b|choir|piano|keys|rhodes|organ|synth|pad|string|"
                           r"wurli|hammond|clav|accordion|harmonium|melodica|"
                           r"violin|fiddle|cello|viola|brass|trumpet|sax|horn|flute|clar|whistle|pipe|"
                           r"bagpipe|chanter|drone|harp|loop|fx|noise|click|"
                           r"clap|snap|cowbell|chime|bell|glock|marimba|vibra|timp|triangle|woodblock|"
                           r"guiro|guache|llamador|paliteo|guaracha|cumbia|rim\b|sidestick|tabla|timbale|"
                           r"trombone|tuba|harmonica|stylophone|orchestra|leslie|nord|sampler|riser|"
                           r"alarm|heartbeat|speech|whisper|spoken|crowd|vos|midi|"
                           r"mix|master|ref\b", re.I)


def classify_track(name: str) -> str:
    """'guitar', 'other' or 'unknown' from a session track's file name."""
    stem = Path(name).stem.replace("_", " ").replace("-", " ")
    if GUITAR_LIKE.search(stem):               # dobro, pedal steel: guitar-ish, but not our target
        return "guitar_like_other"
    if NOT_GUITAR_FIRST.search(stem):          # bass guitar is not the guitar we want
        return "other"
    if GUITAR.search(stem):
        return "guitar"
    if CLEARLY_OTHER.search(stem):
        return "other"
    return "unknown"


def label_song(wavs, headers) -> dict:
    unreadable = [str(p) for p, h in zip(wavs, headers) if not h]
    tracks = [{"relpath": str(p), "role": classify_track(p), **h} for p, h in zip(wavs, headers) if h]
    roles = {t["role"] for t in tracks}
    guitar_like = [t["relpath"] for t in tracks if t["role"] == "guitar_like_other"]
    problems = []
    if unreadable:
        problems.append("could not read: " + ", ".join(unreadable))
    if "unknown" in roles:
        problems.append("unrecognised track names: " + ", ".join(t["relpath"] for t in tracks if t["role"] == "unknown"))
    if "guitar" not in roles:
        problems.append("no guitar track found")
    rates = {t["samplerate"] for t in tracks}
    if len(rates) > 1:
        problems.append(f"tracks differ in sample rate: {sorted(rates)}")
    elif rates and next(iter(rates)) not in ALLOWED_RATES:
        problems.append(f"unsupported sample rate {next(iter(rates))}")
    return {"tracks": tracks, "problems": problems, "guitar_like_not_counted": guitar_like,
            "status": "needs_review" if problems else "ready",
            "frames": max((t["frames"] for t in tracks), default=0),
            "samplerate": next(iter(rates), None)}


def to_project_rate(mixture, guitar, rate: int):
    """Resample a song to 44.1 kHz if it was recorded at another rate."""
    if rate == SAMPLE_RATE:
        return mixture, guitar
    import torch
    from julius import resample_frac
    out = [resample_frac(torch.from_numpy(x), rate, SAMPLE_RATE).numpy() for x in (mixture, guitar)]
    return out[0], out[1]


def mix_song(folder: Path, labels: dict):
    """Full mix and real guitar, summed in memory. Shorter tracks are padded with silence."""
    import soundfile as sf
    frames = labels["frames"]
    mixture = np.zeros((2, frames), dtype=np.float32)
    guitar = np.zeros((2, frames), dtype=np.float32)
    for track in labels["tracks"]:
        audio, rate = sf.read(folder / track["relpath"], dtype="float32", always_2d=True)
        if rate != labels["samplerate"]:
            raise ValueError(f"Sample rate changed since labelling: {track['relpath']}")
        block = audio.T
        if block.shape[0] == 1:
            block = np.repeat(block, 2, axis=0)
        mixture[:, :block.shape[1]] += block
        if track["role"] == "guitar":
            guitar[:, :block.shape[1]] += block
    return to_project_rate(mixture, guitar, labels["samplerate"])


def read_header(path: Path):
    """Sample rate / channels / length of one track, or None if it cannot be read."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return {"samplerate": info.samplerate, "channels": info.channels, "frames": info.frames}
    except ImportError:
        pass
    except Exception:
        return None
    try:
        from index_moisesdb import wav_header
        return wav_header(path)
    except Exception:
        return None


def read_headers(paths):
    return [read_header(p) for p in paths]


def is_junk(name: str) -> bool:
    """macOS adds 4 KB '._name.wav' resource forks and a __MACOSX folder inside zips."""
    parts = Path(name).parts
    return any(part == "__MACOSX" for part in parts) or Path(name).name.startswith("._") \
        or Path(name).name.startswith(".")


def unpack(zip_path: Path, target: Path):
    """Unpack the WAVs of one song into `target`, returning their paths relative to it."""
    wavs = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith((".wav", ".aiff", ".aif", ".flac")):
                continue
            if is_junk(member.filename):      # macOS resource forks etc., not audio
                continue
            out = (target / Path(member.filename).name).resolve()
            if target.resolve() not in out.parents:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
            with archive.open(member) as source, open(out, "wb") as sink:
                shutil.copyfileobj(source, sink, 8 * 1024 * 1024)
            wavs.append(out.name)
    return sorted(wavs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incoming", default="data/raw/cambridge_incoming")
    parser.add_argument("--model", default="htdemucs", choices=CHOICES)
    parser.add_argument("--weight", type=float, default=0.5,
                        help="ensemble only: share of the community model in the blend")
    parser.add_argument("--out", default=None, help="defaults to runs/cambridge_<model>")
    parser.add_argument("--labels-only", action="store_true", help="label tracks and stop, for review")
    parser.add_argument("--keep-audio", action="store_true", help="leave the unpacked audio in place (debugging)")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out or f"runs/cambridge_{RUN_SUFFIX[args.model]}")
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "tracks").mkdir(exist_ok=True)
    zips = sorted(Path(args.incoming).glob("*.zip"))
    if args.limit:
        zips = zips[:args.limit]
    if not zips:
        raise SystemExit(f"No song ZIPs in {args.incoming}. Download a few from "
                         "https://www.cambridge-mt.com/ms3/mtk/ and put them there.")

    separator = None
    if not args.labels_only:
        separator = make_separator(args.model, args.device, args.weight)
    from evaluate_moisesdb import score_song
    from evaluate_slakh import fmt, save_json
    from metrics import peak_memory_gb

    rows, reviewed = [], 0
    for i, zip_path in enumerate(zips):
        song = zip_path.stem
        label_file = out / "labels" / f"{song}.json"
        score_file = out / "tracks" / f"{song}.json"
        if score_file.exists() and not args.labels_only:
            rows.append(json.loads(score_file.read_text()))
            continue
        workspace = Path(tempfile.mkdtemp(prefix="cambridge_"))   # outside the project folder
        try:
            names = unpack(zip_path, workspace)
            if not names:
                print(f"[{i+1}/{len(zips)}] {song}: no audio inside the zip", flush=True)
                continue
            labels = label_song(names, read_headers([workspace / n for n in names]))
            # A file a person has reviewed wins; an untouched guess is re-made, so improvements
            # to the labeller reach songs that were skipped last time.
            reviewed_labels = None
            if label_file.exists():
                candidate = json.loads(label_file.read_text())
                # No "auto" key means the file predates review marking: treat it as a guess, not a review.
                if candidate.get("auto", True) is False and {t["relpath"] for t in candidate["tracks"]} == set(names):
                    reviewed_labels = candidate
            if reviewed_labels:
                labels = reviewed_labels
                reviewed += 1
            else:
                save_json(label_file, {"song": song, "zip": str(zip_path), "auto": True,
                                       "how_to_review": "Correct any 'unknown' roles below, set \"auto\" to "
                                                        "false, and run again. Roles: guitar / other / "
                                                        "guitar_like_other / unknown.",
                                       **labels})
            if labels["status"] != "ready":
                print(f"[{i+1}/{len(zips)}] {song}: needs review -> {'; '.join(labels['problems'])}", flush=True)
                continue
            if args.labels_only:
                guitars = [t["relpath"] for t in labels["tracks"] if t["role"] == "guitar"]
                print(f"[{i+1}/{len(zips)}] {song}: ready, guitar tracks = {', '.join(guitars)}", flush=True)
                continue
            t0 = time.time()
            mixture, guitar = mix_song(workspace, labels)
            estimate = separator.separate_audio(mixture.T, SAMPLE_RATE)["guitar"].T
            row = {"song": song, "source": "cambridge-mt", "tracks": len(labels["tracks"]),
                   "recorded_samplerate": labels["samplerate"],
                   "guitar_tracks": [t["relpath"] for t in labels["tracks"] if t["role"] == "guitar"],
                   "guitar_like_not_counted": labels.get("guitar_like_not_counted", []),
                   "duration_seconds": labels["frames"] / labels["samplerate"], "guitar_types": [],
                   "guitar_has_bleed": False, **score_song(mixture, guitar, estimate),
                   "seconds_elapsed": time.time() - t0}
            save_json(score_file, row)
            rows.append(row)
            print(f"[{i+1}/{len(zips)}] {song}: si-sdr={fmt(row['guitar_si_sdr_db'])} "
                  f"sdr={fmt(row['guitar_sdr_db'])} improvement={fmt(row['guitar_improvement_db'])} "
                  f"({row['seconds_elapsed']:.0f}s)", flush=True)
        finally:
            if not args.keep_audio:
                shutil.rmtree(workspace, ignore_errors=True)    # the audio never lingers

    if args.labels_only:
        print(f"\nLabels written to {out / 'labels'}. Fix any 'unknown' roles there, then run again.")
        return
    if not rows:
        raise SystemExit("Nothing scored. Check the label files for songs marked needs_review.")
    from evaluate_moisesdb import summarise
    summary = {"model": separator.name, "songs": len(rows), "reviewed_label_files": reviewed,
               "peak_memory_gb": peak_memory_gb(), "summary": summarise(rows)}
    save_json(out / "summary.json", summary)
    columns = ["song", "duration_seconds", "tracks", "guitar_tracks", "guitar_si_sdr_db", "guitar_sdr_db",
               "mixture_guitar_si_sdr_db", "guitar_improvement_db", "backing_si_sdr_db", "missed_time_share"]
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "guitar_tracks": ";".join(r["guitar_tracks"])})
    s = summary["summary"]
    lines = ["# htdemucs_6s on Cambridge-MT multitracks (real sessions, no training)", "",
             f"{s['songs_scored']} songs, {s['hours']} h. Audio was unpacked, scored and deleted; "
             "only scores and labels are kept.", "",
             "| | Median | Mean |", "| --- | ---: | ---: |",
             f"| Guitar SI-SDR | {fmt(s['guitar_si_sdr']['median_db'])} | {fmt(s['guitar_si_sdr']['mean_db'])} |",
             f"| Guitar SDR | {fmt(s['guitar_sdr']['median_db'])} | {fmt(s['guitar_sdr']['mean_db'])} |",
             f"| Do-nothing SI-SDR | {fmt(s['mixture_si_sdr']['median_db'])} | {fmt(s['mixture_si_sdr']['mean_db'])} |",
             f"| **Improvement** | **{fmt(s['improvement']['median_db'])}** | **{fmt(s['improvement']['mean_db'])}** |", "",
             f"Helped on {s['improvement']['improved_share']*100:.0f}% of songs; "
             f"{s['songs_made_worse']} made worse. Worst tenth: {fmt(s['improvement_worst_tenth_db'])}.", "",
             "## Which tracks were treated as guitar", "",
             "The last column lists guitar-like instruments (dobro, pedal steel) deliberately left out "
             "of the answer key; if the model puts them in its guitar output they count against it.", "",
             "| Song | Guitar tracks | Guitar-like, not counted |", "| --- | --- | --- |"]
    lines += [f"| {r['song']} | {', '.join(r['guitar_tracks'])} | {', '.join(r['guitar_like_not_counted']) or '-'} |"
              for r in rows]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(s["improvement"], indent=2))


if __name__ == "__main__":
    main()
