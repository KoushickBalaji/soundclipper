"""Offline regression tests for channel handling and guitar/backing exports."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from pretrained_separator import GuitarSeparator


class SeparatorTests(unittest.TestCase):
    def setUp(self):
        # No weights or network are needed to exercise preprocessing/export contracts.
        self.separator = object.__new__(GuitarSeparator)
        self.separator.name = "htdemucs_6s"
        self.separator.device = torch.device("cpu")
        self.separator.sample_rate = 44100
        self.separator.channels = 2
        self.separator.segment = 7.8
        self.separator.overlap = 0.25
        self.separator.model = object()
        self.separator.guitar_index = 4

    @staticmethod
    def fake_model(model, mix, **kwargs):
        return torch.stack([mix * factor for factor in (0, 0, 0, 0, .25, .75)], dim=1)

    def test_native_stereo_and_named_guitar_output(self):
        rng = np.random.default_rng(20)
        x = rng.normal(0, .1, (4410, 2)).astype("float32")
        with patch("demucs.apply.apply_model", side_effect=self.fake_model):
            result = self.separator.separate_audio(x, 44100)
        mean = x.mean(axis=1).mean()
        np.testing.assert_allclose(result["guitar"], .25 * (x - mean) + mean,
                                   atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(result["guitar"] + result["backing"], x, atol=1e-7)

    def test_mono_is_resampled_and_duplicated(self):
        x = np.sin(2*np.pi*220*np.arange(1600)/16000).astype("float32")
        with patch("demucs.apply.apply_model", side_effect=self.fake_model):
            result = self.separator.separate_audio(x, 16000)
        self.assertEqual(result["mixture"].shape, (4410, 2))
        np.testing.assert_array_equal(result["mixture"][:, 0], result["mixture"][:, 1])
        np.testing.assert_allclose(result["guitar"] + result["backing"],
                                   result["mixture"], atol=1e-7)

    def test_silence_avoids_model_and_remains_finite(self):
        with patch("demucs.apply.apply_model") as model:
            result = self.separator.separate_audio(np.zeros(1600), 16000)
        model.assert_not_called()
        for values in result.values():
            self.assertTrue(np.isfinite(values).all())
            self.assertFalse(values.any())

    def test_antiphase_stereo_is_not_treated_as_silence(self):
        mono = np.sin(np.arange(4410) * .1).astype("float32")
        x = np.stack((mono, -mono), axis=1)
        with patch("demucs.apply.apply_model", side_effect=self.fake_model) as model:
            result = self.separator.separate_audio(x, 44100)
        model.assert_called_once()
        self.assertGreater(float(abs(result["guitar"]).max()), .1)

    def test_invalid_audio_rejected(self):
        for x in (np.array([]), np.ones((10, 3)), np.full(10, np.nan)):
            with self.subTest(shape=x.shape):
                with self.assertRaises(ValueError):
                    self.separator.separate_audio(x, 44100)

    def test_float_exports_preserve_unclipped_reconstruction(self):
        x = (2 * np.sin(np.arange(4410) * .1)).astype("float32")
        self.separator.model = type("ModelInfo", (), {"sources": ["guitar", "other"]})()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.wav"
            sf.write(src, x, 44100, subtype="FLOAT")
            with patch("demucs.apply.apply_model", side_effect=self.fake_model):
                details = self.separator.separate_file(src, Path(tmp) / "output")
            mix, sr = sf.read(details["files"]["mixture"], dtype="float32", always_2d=True)
            guitar, _ = sf.read(details["files"]["guitar"], dtype="float32", always_2d=True)
            backing, _ = sf.read(details["files"]["backing"], dtype="float32", always_2d=True)
            self.assertEqual(sr, 44100)
            self.assertGreater(float(abs(mix).max()), 1.)
            np.testing.assert_allclose(guitar + backing, mix, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
