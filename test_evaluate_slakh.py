"""Offline tests for Slakh evaluation helpers (no model, torch or audio files needed)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from evaluate_slakh import group_by_type, guitar_types, loudest_window, stats


class HelperTests(unittest.TestCase):
    def test_guitar_types_from_program_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"stems": {"S00": {"program_num": 28, "inst_class": "Guitar"},
                              "S01": {"program_num": 30, "inst_class": "Guitar"},
                              "S02": {"program_num": 33, "inst_class": "Bass"},
                              "S03": {"program_num": 26, "inst_class": "Guitar"},
                              "S04": {"program_num": 27, "inst_class": "Guitar"},
                              "S05": {"program_num": 80, "inst_class": "Guitar"}}}
            Path(tmp, "metadata.yaml").write_text(yaml.safe_dump(meta))
            record = {"raw_track": tmp, "guitar_stems": ["S00", "S01", "S03", "S04", "S05"]}
            self.assertEqual(guitar_types(record),
                             ["clean electric", "muted electric", "distorted electric", "other guitar"])

    def test_group_contains_versus_only(self):
        rows = [{"guitar_types": ["muted electric"], "guitar_improvement_db": -10.0},
                {"guitar_types": ["muted electric"], "guitar_improvement_db": -2.0},
                {"guitar_types": ["acoustic", "muted electric"], "guitar_improvement_db": 4.0},
                {"guitar_types": ["acoustic"], "guitar_improvement_db": 6.0}]
        groups = {g["guitar_type"]: g for g in group_by_type(rows)}
        self.assertEqual(groups["muted electric"]["contains"]["songs"], 3)
        self.assertEqual(groups["muted electric"]["only"]["songs"], 2)
        self.assertAlmostEqual(groups["muted electric"]["only"]["median_db"], -6.0)
        self.assertAlmostEqual(groups["acoustic"]["contains"]["improved_share"], 1.0)
        self.assertNotIn("clean electric", groups)

    def test_stats_ignores_missing(self):
        s = stats([1.0, None, -1.0, 3.0])
        self.assertEqual(s["songs"], 3)
        self.assertAlmostEqual(s["median_db"], 1.0)
        self.assertEqual(stats([None])["songs"], 0)

    def test_loudest_window(self):
        sr = 100
        x = np.zeros(10 * sr, dtype=np.float32)
        x[6 * sr:8 * sr] = 1.0
        self.assertEqual(loudest_window(x, 2.0, sr), 6 * sr)
        self.assertEqual(loudest_window(x[:sr], 2.0, sr), 0)


if __name__ == "__main__":
    unittest.main()
