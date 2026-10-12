"""Tests for the paired comparison, including the sign test."""
import unittest

from compare_runs import compare, sign_test


def row(song, improvement, si=None):
    return {"song": song, "guitar_improvement_db": improvement,
            "guitar_si_sdr_db": improvement if si is None else si,
            "guitar_sdr_db": improvement - 1, "backing_si_sdr_db": 7.0}


class CompareTests(unittest.TestCase):
    def test_pairs_only_shared_songs(self):
        a = {"x": row("x", 5.0), "y": row("y", 3.0), "onlyA": row("onlyA", 1.0)}
        b = {"x": row("x", 6.0), "y": row("y", 2.0), "onlyB": row("onlyB", 9.0)}
        result = compare(a, b)
        self.assertEqual(result["songs_compared"], 2)
        self.assertEqual(result["only_in_a"], ["onlyA"])
        self.assertEqual(result["only_in_b"], ["onlyB"])
        metric = result["metrics"]["guitar_improvement_db"]
        self.assertAlmostEqual(metric["paired_median_gain_db"], 0.0)   # +1 and -1
        self.assertEqual((metric["b_wins"], metric["a_wins"]), (1, 1))

    def test_clear_win_is_flagged_as_unlikely_chance(self):
        a = {f"s{i}": row(f"s{i}", 0.0) for i in range(12)}
        b = {f"s{i}": row(f"s{i}", 2.0) for i in range(12)}
        metric = compare(a, b)["metrics"]["guitar_improvement_db"]
        self.assertAlmostEqual(metric["paired_median_gain_db"], 2.0)
        self.assertEqual(metric["b_win_share"], 1.0)
        self.assertLess(metric["sign_test_p"], 0.001)

    def test_sign_test_values(self):
        self.assertAlmostEqual(sign_test(5, 5), 1.0)
        self.assertAlmostEqual(sign_test(10, 0), 2 / 1024)
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertLess(sign_test(9, 1), 0.05)

    def test_missing_metric_does_not_break_the_comparison(self):
        a = {"x": {"song": "x", "guitar_improvement_db": 1.0}}
        b = {"x": {"song": "x", "guitar_improvement_db": None}}
        self.assertNotIn("guitar_improvement_db", compare(a, b)["metrics"])


if __name__ == "__main__":
    unittest.main()
