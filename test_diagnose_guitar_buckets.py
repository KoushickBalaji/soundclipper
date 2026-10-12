"""Offline tests for the bucket diagnosis helpers."""
import unittest

import numpy as np

from diagnose_guitar_buckets import guitar_share, missed_time


class DiagnoseTests(unittest.TestCase):
    def test_guitar_share_splits_across_buckets(self):
        rng = np.random.default_rng(0)
        real = rng.normal(size=44100)
        self.assertAlmostEqual(guitar_share(0.7 * real, real), 0.7, places=6)
        self.assertAlmostEqual(guitar_share(0.3 * real + 1.0, real), 0.3, places=6)
        self.assertAlmostEqual(guitar_share(rng.normal(size=44100) * 0.01, real), 0.0, places=2)

    def test_missed_time(self):
        sr = 100
        real = np.ones(10 * sr)
        est = real.copy()
        est[:4 * sr] = 0.001          # 4 of 10 playing seconds are nearly silent
        self.assertAlmostEqual(missed_time(est, real, sr), 0.4)
        real[5 * sr:] = 0.0           # guitar stops: only 5 seconds count as playing
        self.assertAlmostEqual(missed_time(est, real, sr), 0.8)
        self.assertIsNone(missed_time(est[:10], real[:10], sr))


if __name__ == "__main__":
    unittest.main()
