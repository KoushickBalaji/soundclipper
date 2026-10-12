"""Offline tests for the MoisesDB scoring helpers (no model needed)."""
import unittest

import numpy as np

from evaluate_moisesdb import score_song, summarise


def row(improvement, bleed=False, types=("acoustic",), missed=0.1):
    return {"guitar_improvement_db": improvement, "guitar_si_sdr_db": improvement,
            "stereo_penalty_db": 0.3,
            "guitar_sdr_db": improvement - 1, "mixture_guitar_si_sdr_db": 0.0,
            "backing_si_sdr_db": 5.0, "missed_time_share": missed, "guitar_has_bleed": bleed,
            "guitar_types": list(types), "duration_seconds": 180.0}


class ScoreTests(unittest.TestCase):
    def test_false_extraction_on_a_guitar_free_song(self):
        from evaluate_moisesdb import false_extraction_db
        rng = np.random.default_rng(11)
        mixture = rng.normal(size=(2, 44100)).astype("float32")
        self.assertLess(false_extraction_db(mixture, mixture * 1e-4), -60)
        self.assertAlmostEqual(false_extraction_db(mixture, mixture), 0.0, delta=1e-6)

    def test_stereo_errors_are_not_hidden(self):
        rng = np.random.default_rng(5)
        guitar = rng.normal(size=(2, 44100)).astype("float32")
        guitar[1] *= 0.4
        mixture = guitar + rng.normal(size=(2, 44100)).astype("float32")
        flipped = guitar[::-1].copy()
        scores = score_song(mixture, guitar, flipped)
        self.assertLess(scores["guitar_si_sdr_db"], scores["guitar_si_sdr_mono_db"])
        self.assertGreater(scores["stereo_penalty_db"], 0)

    def test_score_song_on_a_perfect_and_a_silent_estimate(self):
        rng = np.random.default_rng(3)
        guitar = rng.normal(size=(2, 44100)).astype("float32")
        backing = rng.normal(size=(2, 44100)).astype("float32")
        mixture = guitar + backing
        perfect = score_song(mixture, guitar, guitar)
        self.assertGreater(perfect["guitar_si_sdr_db"], 60)
        self.assertGreater(perfect["guitar_improvement_db"], 60)
        self.assertEqual(perfect["missed_time_share"], 0.0)
        silent = score_song(mixture, guitar, np.zeros_like(guitar))
        self.assertLess(silent["guitar_si_sdr_db"], -60)
        self.assertEqual(silent["missed_time_share"], 1.0)
        self.assertAlmostEqual(perfect["guitar_energy_share"], 0.5, delta=0.05)

    def test_summarise_splits_bleed_and_types(self):
        rows = [row(4.0), row(-2.0, bleed=True), row(6.0, types=("clean electric",)), row(2.0)]
        s = summarise(rows)
        self.assertEqual(s["songs_scored"], 4)
        self.assertAlmostEqual(s["improvement"]["median_db"], 3.0)
        self.assertEqual(s["improvement_clean_guitar_tracks"]["songs"], 3)
        self.assertEqual(s["improvement_guitar_tracks_with_bleed"]["songs"], 1)
        self.assertAlmostEqual(s["missed_time_share_median"], 0.1)
        by_type = {g["guitar_type"]: g for g in s["by_guitar_type"]}
        self.assertEqual(by_type["acoustic"]["contains"]["songs"], 3)
        self.assertEqual(by_type["clean electric"]["only"]["songs"], 1)


if __name__ == "__main__":
    unittest.main()
