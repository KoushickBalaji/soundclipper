"""Offline tests for the community model's chunking and cross-fade (no torch needed)."""
import unittest

import numpy as np

from roformer_guitar import chunk_starts, fade_window, overlap_add


class ChunkTests(unittest.TestCase):
    def test_starts_cover_the_signal(self):
        self.assertEqual(chunk_starts(100, 40, 20), [0, 20, 40, 60])
        self.assertEqual(chunk_starts(30, 40, 20), [0])           # shorter than one chunk
        self.assertEqual(chunk_starts(101, 40, 20)[-1], 61)       # final chunk reaches the end
        with self.assertRaises(ValueError):
            chunk_starts(100, 20, 40)

    def test_overlap_add_reconstructs_a_signal_including_its_edges(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=(2, 1000)).astype(np.float32)
        chunk, hop = 200, 100
        starts = chunk_starts(signal.shape[1], chunk, hop)
        pieces = [signal[:, s:s + chunk] for s in starts]
        restored = overlap_add(pieces, starts, signal.shape[1], chunk, hop)
        np.testing.assert_allclose(restored, signal, atol=1e-5)
        self.assertGreater(abs(restored[0, 0]), 0)        # first sample survives

    def test_single_chunk_is_not_faded(self):
        ones = [np.ones((2, 200), dtype=np.float32)]
        np.testing.assert_allclose(overlap_add(ones, [0], 200, 200, 100), 1.0, atol=1e-5)

    def test_fade_window_edges(self):
        middle = fade_window(200, 100, first=False, last=False)
        self.assertAlmostEqual(middle[0], 0.0, places=6)
        first = fade_window(200, 100, first=True, last=False)
        np.testing.assert_allclose(first[:100], 1.0)
        last = fade_window(200, 100, first=False, last=True)
        np.testing.assert_allclose(last[100:], 1.0)


if __name__ == "__main__":
    unittest.main()
