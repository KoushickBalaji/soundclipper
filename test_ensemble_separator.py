"""Offline tests for the blend (no models needed)."""
import unittest

import numpy as np

from ensemble_separator import blend
from separators import CHOICES, RUN_SUFFIX


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.a = np.ones((100, 2), dtype=np.float32)
        self.b = np.full((100, 2), 3.0, dtype=np.float32)

    def test_equal_average(self):
        np.testing.assert_allclose(blend([self.a, self.b], [0.5, 0.5]), 2.0, atol=1e-6)

    def test_weight_shifts_towards_one_model(self):
        np.testing.assert_allclose(blend([self.a, self.b], [0.25, 0.75]), 2.5, atol=1e-6)
        np.testing.assert_allclose(blend([self.a, self.b], [1.0, 0.0]), self.a, atol=1e-6)

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            blend([self.a, self.b], [0.5, 0.6])
        with self.assertRaises(ValueError):
            blend([], [])

    def test_tiny_length_difference_is_trimmed_but_a_big_one_is_refused(self):
        self.assertEqual(blend([self.a, self.b[:99]], [0.5, 0.5]).shape, (99, 2))
        with self.assertRaises(ValueError):
            blend([self.a, self.b[:50]], [0.5, 0.5])

    def test_ensemble_is_a_known_model_choice(self):
        self.assertIn("ensemble", CHOICES)
        self.assertEqual(RUN_SUFFIX["ensemble"], "ensemble")


if __name__ == "__main__":
    unittest.main()
