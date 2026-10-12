"""Tests for song selection and the blind labelling."""
import unittest

import numpy as np

from make_listening_clips import assign_slots, loudest_window, pick_songs


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = {f"s{i:02d}": {"guitar_improvement_db": float(i)} for i in range(20)}

    def test_picks_span_worst_middle_and_best(self):
        picks = pick_songs(self.rows, counts=(2, 2, 2))
        self.assertIn("s00", picks)          # worst
        self.assertIn("s19", picks)          # best
        self.assertEqual(len(picks), 6)
        self.assertEqual(len(set(picks)), 6)

    def test_slots_are_shuffled_per_song_but_reproducible(self):
        songs = [f"s{i}" for i in range(12)]
        first = assign_slots(songs, ("htdemucs", "ensemble"), seed=7)
        second = assign_slots(songs, ("htdemucs", "ensemble"), seed=7)
        self.assertEqual(first, second)
        for song in songs:                    # each song still has exactly one of each model
            self.assertEqual(sorted(first[song].values()), ["ensemble", "htdemucs"])
        orders = {tuple(first[song].values()) for song in songs}
        self.assertEqual(len(orders), 2, "both A/B orderings should occur across songs")
        flipped = sum(1 for song in songs if first[song]["A"] == "ensemble")
        self.assertGreater(flipped, 0)
        self.assertLess(flipped, len(songs))

    def test_loudest_window_finds_the_active_part(self):
        sr = 100
        guitar = np.zeros((2, 10 * sr), dtype=np.float32)
        guitar[:, 6 * sr:8 * sr] = 1.0
        self.assertEqual(loudest_window(guitar, 2.0, sr), 6 * sr)


if __name__ == "__main__":
    unittest.main()
