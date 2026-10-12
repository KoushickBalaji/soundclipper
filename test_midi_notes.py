"""Tests for the MIDI reader, built on a MIDI file written by hand in the test."""
import struct
import tempfile
import unittest
from pathlib import Path

from midi_notes import pitches_at, read_notes, ticks_to_seconds, tempo_map


def varint(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def midi_file(path: Path, events, ticks_per_beat=480, tempo=None):
    """events: (delta ticks, raw bytes)."""
    body = b""
    if tempo:
        body += varint(0) + b"\xff\x51\x03" + tempo.to_bytes(3, "big")
    for delta, raw in events:
        body += varint(delta) + raw
    body += varint(0) + b"\xff\x2f\x00"
    track = b"MTrk" + struct.pack(">I", len(body)) + body
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat) + track)


class MidiTests(unittest.TestCase):
    def test_notes_and_times_at_120bpm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "a.mid")
            midi_file(path, [(480, b"\x90\x3c\x64"), (480, b"\x80\x3c\x40")])   # C4 on beat 2, one beat long
            notes = read_notes(path)
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0]["pitch"], 60)
            self.assertAlmostEqual(notes[0]["start"], 0.5, places=6)            # 480 ticks = half a second
            self.assertAlmostEqual(notes[0]["end"], 1.0, places=6)

    def test_note_on_with_zero_velocity_ends_a_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "b.mid")
            midi_file(path, [(0, b"\x90\x40\x50"), (960, b"\x90\x40\x00")])
            notes = read_notes(path)
            self.assertEqual(len(notes), 1)
            self.assertAlmostEqual(notes[0]["end"], 1.0, places=6)

    def test_tempo_changes_are_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "c.mid")
            midi_file(path, [(0, b"\x90\x3c\x64"), (480, b"\x80\x3c\x40")], tempo=250000)  # 240 bpm
            notes = read_notes(path)
            self.assertAlmostEqual(notes[0]["end"], 0.25, places=6)
            self.assertEqual(tempo_map(path)[0][1], 250000)

    def test_tempo_can_come_from_another_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem, source = Path(tmp, "s.mid"), Path(tmp, "all.mid")
            midi_file(stem, [(0, b"\x90\x3c\x64"), (480, b"\x80\x3c\x40")])
            midi_file(source, [], tempo=250000)
            self.assertAlmostEqual(read_notes(stem, tempo_source=source)[0]["end"], 0.25, places=6)

    def test_pitch_classes_at_a_moment(self):
        notes = [{"pitch": 60, "start": 0.0, "end": 2.0}, {"pitch": 64, "start": 0.0, "end": 2.0},
                 {"pitch": 67, "start": 0.0, "end": 1.0}, {"pitch": 72, "start": 3.0, "end": 4.0}]
        self.assertEqual(pitches_at(notes, 0.5), {0, 4, 7})      # C major triad
        self.assertEqual(pitches_at(notes, 1.5), {0, 4})
        self.assertEqual(pitches_at(notes, 2.5), set())

    def test_running_status_and_ticks(self):
        self.assertAlmostEqual(ticks_to_seconds(960, 480, [(0, 500000)]), 1.0, places=6)
        self.assertAlmostEqual(ticks_to_seconds(960, 480, [(0, 500000), (480, 250000)]), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
