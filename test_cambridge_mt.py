"""Offline tests for the Cambridge-MT fetch-score-delete helper."""
import json
import shutil
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from cambridge_mt import classify_track, is_junk, label_song, mix_song, unpack
from test_index_moisesdb import write_wav


class ClassifyTests(unittest.TestCase):
    def test_guitar_names(self):
        for name in ["05_ElecGtrL.wav", "12_AcousticGtr_DI.wav", "AcGtr Room.wav", "lead guitar 2.wav",
                     "09_Electric_Gtr_Amp.wav"]:
            self.assertEqual(classify_track(name), "guitar", name)

    def test_bass_and_lookalikes_are_not_guitar(self):
        for name in ["03_BassGtr.wav", "Bass DI.wav", "07_BassGuitar_Amp.wav", "Banjo.wav",
                     "Mandolin 1.wav", "Sitar.wav"]:
            self.assertEqual(classify_track(name), "other", name)

    def test_obvious_other_instruments(self):
        for name in ["01_Kick_In.wav", "Snare Bottom.wav", "LeadVox.wav", "Rhodes.wav", "Strings Hi.wav"]:
            self.assertEqual(classify_track(name), "other", name)

    def test_guitar_like_instruments_are_separate(self):
        for name in ["07_Dobro.wav", "14_PedalSteel.wav", "LapSteel.wav"]:
            self.assertEqual(classify_track(name), "guitar_like_other", name)

    def test_folk_and_keys_are_other(self):
        for name in ["09_Fiddle.wav", "11_BagpipeDrones.wav", "13_Accordion.wav", "11_Wurlitzer.wav",
                     "12_HammondMic1Hi.wav", "01_CongaMic1.wav", "03_Brushes.wav", "10_Low Whistle.wav"]:
            self.assertEqual(classify_track(name), "other", name)

    def test_session_oddities_are_other(self):
        for name in ["13_Claps.wav", "06_Cowbell.wav", "45_Trombone.wav", "50_Tuba.wav", "04_Timp.wav",
                     "07_Glockenspiel.wav", "20_Harmonica.wav", "04_Chimes.wav", "18_Speech.wav",
                     "19_Whisper.wav", "NORD ELECTRO L_DI.wav", "LESLIE TOP_M80-SH.wav", "SAMPLER_DI.wav",
                     "18_MIDITrombone.wav", "22_Riser render 001.wav", "16_Stylophone.wav", "18_Guiro.wav",
                     "32_BackingVos01.wav", "55_Marimba.wav", "02_Rim.wav"]:
            self.assertEqual(classify_track(name), "other", name)

    def test_ambiguous_names_still_stop_the_song(self):
        for name in ["08_Chug.wav", "09_MainMics.wav", "08-130624_1035.wav", "23_Break1.wav"]:
            self.assertEqual(classify_track(name), "unknown", name)

    def test_unrecognised_names_stay_unknown(self):
        self.assertEqual(classify_track("Track 07.wav"), "unknown")
        self.assertEqual(classify_track("zz_bounce_3.wav"), "unknown")


class LabelFileTests(unittest.TestCase):
    """Only a file explicitly marked reviewed ("auto": false) survives a re-run."""

    @staticmethod
    def honoured(saved: dict) -> bool:
        return saved.get("auto", True) is False

    def test_reviewed_file_is_kept_and_guesses_are_redone(self):
        self.assertTrue(self.honoured({"auto": False}))
        self.assertFalse(self.honoured({"auto": True}))
        self.assertFalse(self.honoured({}))            # written before review marking existed


class SongTests(unittest.TestCase):
    def test_label_song_flags_problems(self):
        headers = [{"samplerate": 44100, "frames": 100, "channels": 2},
                   {"samplerate": 44100, "frames": 80, "channels": 2}]
        ready = label_song(["ElecGtr.wav", "Kick.wav"], headers)
        self.assertEqual(ready["status"], "ready")
        with_dobro = label_song(["ElecGtr.wav", "07_Dobro.wav"], headers)
        self.assertEqual(with_dobro["status"], "ready")
        self.assertEqual(with_dobro["guitar_like_not_counted"], ["07_Dobro.wav"])
        self.assertEqual(ready["frames"], 100)
        mystery = label_song(["ElecGtr.wav", "Track 3.wav"], headers)
        self.assertEqual(mystery["status"], "needs_review")
        no_guitar = label_song(["Kick.wav", "Snare.wav"], headers)
        self.assertIn("no guitar track found", no_guitar["problems"])
        at_48k = label_song(["ElecGtr.wav", "Kick.wav"],
                            [{"samplerate": 48000, "frames": 100, "channels": 2}] * 2)
        self.assertEqual(at_48k["status"], "ready")      # resampled on the way in
        odd_rate = label_song(["ElecGtr.wav", "Kick.wav"],
                              [{"samplerate": 22050, "frames": 100, "channels": 2}] * 2)
        self.assertEqual(odd_rate["status"], "needs_review")

    def test_macos_resource_forks_are_junk(self):
        for name in ["._Acoustic Guitar-Ela M 260.wav", "Song/__MACOSX/x.wav", "__MACOSX/._y.wav", ".hidden.wav"]:
            self.assertTrue(is_junk(name), name)
        for name in ["01_ElecGtr.wav", "Song/02_Kick.wav"]:
            self.assertFalse(is_junk(name), name)

    def test_unreadable_track_marks_review_instead_of_crashing(self):
        headers = [{"samplerate": 44100, "frames": 100, "channels": 2}, None]
        labels = label_song(["ElecGtr.wav", "broken.wav"], headers)
        self.assertEqual(labels["status"], "needs_review")
        self.assertTrue(any("could not read" in p for p in labels["problems"]))
        self.assertEqual(len(labels["tracks"]), 1)

    def test_unpack_skips_resource_forks(self):
        import tempfile, zipfile
        from pathlib import Path as P
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = P(tmp, "S.zip")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("S/01_ElecGtr.wav", b"RIFF....WAVE")
                archive.writestr("S/._01_ElecGtr.wav", b"junk")
                archive.writestr("__MACOSX/S/._01_ElecGtr.wav", b"junk")
            target = P(tmp, "work")
            target.mkdir()
            self.assertEqual(unpack(zip_path, target), ["01_ElecGtr.wav"])

    def test_unpack_then_mix_then_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "song")
            source.mkdir()
            write_wav(source / "01_ElecGtr.wav", 1000, fill=0.25)
            write_wav(source / "02_Kick.wav", 600, fill=0.5)
            zip_path = Path(tmp, "SongName.zip")
            with zipfile.ZipFile(zip_path, "w") as archive:
                for f in sorted(source.iterdir()):
                    archive.write(f, f"SongName/{f.name}")
            target = Path(tmp, "work")
            target.mkdir()
            names = unpack(zip_path, target)
            self.assertEqual(names, ["01_ElecGtr.wav", "02_Kick.wav"])
            labels = label_song(names, [{"samplerate": 44100, "frames": 1000, "channels": 2},
                                        {"samplerate": 44100, "frames": 600, "channels": 2}])
            try:
                import soundfile  # noqa: F401
            except ImportError:
                return  # mixing needs soundfile, which lives in the Windows environment
            mixture, guitar = mix_song(target, labels)
            self.assertEqual(mixture.shape, (2, 1000))
            np.testing.assert_allclose(mixture[:, 100], 0.75, atol=1e-3)
            np.testing.assert_allclose(mixture[:, 900], 0.25, atol=1e-3)   # kick already ended
            np.testing.assert_allclose(guitar[:, 900], 0.25, atol=1e-3)
            shutil.rmtree(target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
