"""Read note events out of a Standard MIDI File — no extra packages.

Slakh renders every song from MIDI, so `MIDI/<stem>.mid` tells us exactly which notes the
guitar played and when. That is ground truth for chord and note tagging, which is what
SoundClipper is ultimately for.

    from midi_notes import read_notes
    notes = read_notes("…/MIDI/S08.mid", tempo_source="…/all_src.mid")
    # -> [{"pitch": 52, "start": 1.93, "end": 2.41}, …] with times in seconds

Per-stem files often carry no tempo of their own, so the tempo map is read from the song's
all_src.mid when one is given; otherwise 120 bpm is assumed, as the MIDI standard says.
"""
from __future__ import annotations

import struct
from pathlib import Path

DEFAULT_TEMPO = 500000          # microseconds per quarter note = 120 bpm


def _varint(data: bytes, i: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, i


def parse_midi(path: str | Path) -> tuple[int, list[list[tuple[int, bytes]]]]:
    """Return (ticks per beat, tracks) where each track is a list of (absolute tick, event)."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"Not a MIDI file: {path}")
    (header_len,) = struct.unpack(">I", data[4:8])
    _fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    i = 8 + header_len
    tracks = []
    for _ in range(ntracks):
        if data[i:i + 4] != b"MTrk":
            break
        (length,) = struct.unpack(">I", data[i + 4:i + 8])
        end = i + 8 + length
        j, tick, status, events = i + 8, 0, None, []
        while j < end:
            delta, j = _varint(data, j)
            tick += delta
            byte = data[j]
            if byte & 0x80:
                status, j = byte, j + 1
            if status is None:
                raise ValueError("Running status before any status byte")
            if status == 0xFF:                       # meta event
                meta_type = data[j]
                length_, j = _varint(data, j + 1)
                events.append((tick, bytes([0xFF, meta_type]) + data[j:j + length_]))
                j += length_
            elif status in (0xF0, 0xF7):             # sysex
                length_, j = _varint(data, j)
                j += length_
            else:
                size = 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
                events.append((tick, bytes([status]) + data[j:j + size]))
                j += size
        tracks.append(events)
        i = end
    return division, tracks


def tempo_map(path: str | Path) -> list[tuple[int, int]]:
    """[(tick, microseconds per beat)] from a file's set-tempo events, always starting at 0."""
    _, tracks = parse_midi(path)
    changes = [(tick, int.from_bytes(event[2:5], "big"))
               for track in tracks for tick, event in track
               if event[0] == 0xFF and event[1] == 0x51 and len(event) >= 5]
    changes.sort()
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, DEFAULT_TEMPO))
    return changes


def ticks_to_seconds(tick: int, ticks_per_beat: int, tempos: list[tuple[int, int]]) -> float:
    seconds, last_tick, last_tempo = 0.0, 0, tempos[0][1]
    for change_tick, tempo in tempos:
        if change_tick >= tick:
            break
        seconds += (change_tick - last_tick) / ticks_per_beat * last_tempo / 1e6
        last_tick, last_tempo = change_tick, tempo
    return seconds + (tick - last_tick) / ticks_per_beat * last_tempo / 1e6


def read_notes(path: str | Path, tempo_source: str | Path | None = None) -> list[dict]:
    """Note events with times in seconds, sorted by start."""
    ticks_per_beat, tracks = parse_midi(path)
    tempos = tempo_map(tempo_source) if tempo_source else tempo_map(path)
    notes, sounding = [], {}
    for track in tracks:
        for tick, event in track:
            kind, status = event[0] & 0xF0, event[0]
            if status == 0xFF:
                continue
            if kind == 0x90 and event[2] > 0:                       # note on
                sounding.setdefault((event[0] & 0x0F, event[1]), []).append(tick)
            elif kind == 0x80 or (kind == 0x90 and event[2] == 0):  # note off
                started = sounding.get((event[0] & 0x0F, event[1]))
                if started:
                    start_tick = started.pop(0)
                    notes.append({"pitch": int(event[1]),
                                  "start": ticks_to_seconds(start_tick, ticks_per_beat, tempos),
                                  "end": ticks_to_seconds(tick, ticks_per_beat, tempos)})
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def pitches_at(notes: list[dict], time: float) -> set[int]:
    """Pitch classes (0-11) sounding at a moment — what a chord label is built from."""
    return {n["pitch"] % 12 for n in notes if n["start"] <= time < n["end"]}
