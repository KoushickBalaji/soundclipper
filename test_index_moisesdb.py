"""Offline tests for the MoisesDB index (no audio is written except tiny temp files)."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from index_moisesdb import index_song, load_moisesdb_song, wav_header


def write_wav(path, frames, channels=2, rate=44100, float32=False, fill=0.0):
    bits, tag = (32, 3) if float32 else (16, 1)
    align = channels * bits // 8
    data = (np.full(frames * channels, fill, "<f4").tobytes() if float32
            else np.full(frames * channels, int(fill * 32767), "<i2").tobytes())
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * align, align, bits)
    junk = b"JUNK" + struct.pack("<I", 28) + bytes(28)
    body = b"WAVE" + junk + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


class IndexTests(unittest.TestCase):
    def test_header_skips_junk_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_wav(Path(tmp, "a.wav"), 1000, channels=2, float32=True)
            h = wav_header(Path(tmp, "a.wav"))
            self.assertEqual((h["channels"], h["samplerate"], h["bits"], h["frames"]), (2, 44100, 32, 1000))

    def test_song_index_pads_to_longest_and_finds_guitar(self):
        with tempfile.TemporaryDirectory() as tmp:
            song = Path(tmp, "song1")
            meta = {"song": "x", "artist": "y", "genre": "rock", "stems": [
                {"stemName": "guitar", "tracks": [{"trackType": "clean_electric_guitar", "id": "g1", "extension": "wav"},
                                                  {"trackType": "acoustic_guitar", "id": "g2", "extension": "wav", "has_bleed": True}]},
                {"stemName": "drums", "tracks": [{"trackType": "kick", "id": "d1", "extension": "wav"}]}]}
            song.mkdir()
            (song / "data.json").write_text(json.dumps(meta))
            write_wav(song / "guitar" / "g1.wav", 1000, fill=0.25)
            write_wav(song / "guitar" / "g2.wav", 800, fill=0.25)
            write_wav(song / "drums" / "d1.wav", 1000, fill=0.5)
            r = index_song(song)
            self.assertTrue(all(not Path(t["relpath"]).is_absolute() for t in r["tracks"]))
            self.assertEqual(r["status"], "ready")
            self.assertEqual(r["frames"], 1000)
            self.assertAlmostEqual(r["length_gap_seconds"], 200 / 44100)
            self.assertEqual(r["guitar_types"], ["acoustic", "clean electric"])
            self.assertTrue(r["guitar_has_bleed"])
            try:
                import soundfile  # noqa: F401
            except ImportError:
                return  # loader check runs in the Windows environment
            mix, guitar = load_moisesdb_song(r)
            self.assertEqual(mix.shape, (2, 1000))
            np.testing.assert_allclose(guitar[:, 900], 0.25, atol=1e-3)   # g2 has ended: padded with silence
            np.testing.assert_allclose(guitar[:, 100], 0.5, atol=1e-3)
            np.testing.assert_allclose(mix[:, 100], 1.0, atol=1e-3)

    def test_song_without_guitar_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            song = Path(tmp, "s")
            song.mkdir()
            (song / "data.json").write_text(json.dumps({"stems": [{"stemName": "drums", "tracks": [
                {"trackType": "kick", "id": "d1", "extension": "wav"}]}]}))
            write_wav(song / "drums" / "d1.wav", 100)
            self.assertEqual(index_song(song)["problems"], ["no guitar"])


if __name__ == "__main__":
    unittest.main()
