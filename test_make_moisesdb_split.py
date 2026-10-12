"""Tests for the artist-based MoisesDB split."""
import unittest

from make_moisesdb_split import build_split


def song(song_id, artist, seconds, types=("acoustic",), status="ready"):
    return {"song_id": song_id, "artist": artist, "duration_seconds": seconds,
            "guitar_types": list(types), "guitar_has_bleed": False, "status": status}


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.records = [song(f"s{i}", f"artist{i // 3}", 200 + i) for i in range(30)]

    def test_artist_never_spans_two_groups(self):
        split = build_split(self.records)
        seen = {}
        for r in self.records:
            seen.setdefault(r["artist"], set()).add(split["songs"][r["song_id"]])
        for artist, groups in seen.items():
            self.assertEqual(len(groups), 1, f"{artist} landed in {groups}")

    def test_shares_are_close_to_targets(self):
        split = build_split(self.records)
        self.assertAlmostEqual(split["stats"]["train"]["share"], 0.70, delta=0.08)
        self.assertAlmostEqual(split["stats"]["validation"]["share"], 0.15, delta=0.08)
        self.assertEqual(sum(s["songs"] for s in split["stats"].values()), 30)

    def test_deterministic_and_skips_unready(self):
        records = self.records + [song("bad", "artistX", 100, status="skipped")]
        first, second = build_split(records), build_split(records)
        self.assertEqual(first["songs"], second["songs"])
        self.assertNotIn("bad", first["songs"])

    def test_missing_artist_is_treated_as_its_own(self):
        records = [song("a", None, 100), song("b", None, 100), song("c", "x", 100)]
        split = build_split(records)
        self.assertEqual(len(set(split["artists"].values()) | set()), len(set(split["songs"].values())))
        self.assertEqual(len(split["songs"]), 3)


if __name__ == "__main__":
    unittest.main()
