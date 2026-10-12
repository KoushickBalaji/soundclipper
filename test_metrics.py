"""Tests for the scoring functions, including known signal-to-noise values."""
import unittest

import numpy as np

from metrics import missed_share, sdr_db, si_sdr_db


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.ref = self.rng.normal(size=44100)

    def test_known_noise_levels(self):
        for target in (20.0, 10.0, 0.0, -5.0):
            noise = self.rng.normal(size=self.ref.size)
            noise *= np.sqrt((self.ref @ self.ref) / (noise @ noise) * 10 ** (-target / 10))
            self.assertAlmostEqual(si_sdr_db(self.ref + noise, self.ref), target, delta=0.15)
            self.assertAlmostEqual(sdr_db(self.ref + noise, self.ref), target, delta=0.15)

    def test_volume_dial_only_affects_sdr(self):
        quiet = 0.5 * self.ref
        self.assertGreater(si_sdr_db(quiet, self.ref), 100)     # perfect shape, wrong volume
        self.assertAlmostEqual(sdr_db(quiet, self.ref), 6.02, delta=0.05)

    def test_silent_reference_has_no_score(self):
        self.assertIsNone(si_sdr_db(self.ref, np.zeros_like(self.ref)))
        self.assertIsNone(sdr_db(self.ref, np.zeros_like(self.ref)))

    def test_missed_share(self):
        sr = 100
        ref = np.ones(10 * sr)
        est = ref.copy()
        est[:4 * sr] = 0.001
        self.assertAlmostEqual(missed_share(est, ref, sr), 0.4)
        ref[5 * sr:] = 0.0
        self.assertAlmostEqual(missed_share(est, ref, sr), 0.8)

    def test_stereo_is_scored_as_one_signal(self):
        left = self.rng.normal(size=22050)
        stereo = np.stack([left, 0.5 * left])
        self.assertGreater(si_sdr_db(stereo, stereo), 60)
        swapped = np.stack([0.5 * left, left])          # same mono average, wrong balance
        self.assertLess(si_sdr_db(swapped, stereo), 15)
        self.assertLess(sdr_db(swapped, stereo), 15)
        self.assertGreater(si_sdr_db(2 * stereo, stereo), 60)      # volume dial still forgives loudness
        self.assertAlmostEqual(sdr_db(2 * stereo, stereo), 0.0, delta=0.1)   # but SDR does not

    def test_silent_estimate_hits_the_floor(self):
        self.assertEqual(si_sdr_db(np.zeros_like(self.ref), self.ref), -100.0)
        self.assertAlmostEqual(sdr_db(np.zeros_like(self.ref), self.ref), 0.0, delta=1e-6)

    def test_matches_the_trainer_si_sdr_where_both_are_defined(self):
        try:
            import torch
            from train import si_sdr
        except ImportError:
            self.skipTest("torch only available in the Windows environment")
        est = 0.4 * self.ref + 0.2 * self.rng.normal(size=self.ref.size)
        theirs = float(si_sdr(torch.from_numpy(est), torch.from_numpy(self.ref)))
        self.assertAlmostEqual(si_sdr_db(est, self.ref), theirs, places=6)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            si_sdr_db(self.ref[:100], self.ref)


if __name__ == "__main__":
    unittest.main()
