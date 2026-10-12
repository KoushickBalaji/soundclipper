"""Tests for the chord labelling, using notes whose chords we know by name."""
import unittest

import numpy as np

from chords import NO_CHORD, chordal_share, label_from_chroma, labels_from_midi, root_of, score_labels


def held(pitches, start=0.0, end=10.0):
    return [{"pitch": p, "start": start, "end": end} for p in pitches]


class ChordTests(unittest.TestCase):
    def test_triads_are_named_correctly(self):
        c_major = np.zeros(12); c_major[[0, 4, 7]] = 1
        a_minor = np.zeros(12); a_minor[[9, 0, 4]] = 1
        g_major = np.zeros(12); g_major[[7, 11, 2]] = 1
        self.assertEqual(label_from_chroma(c_major), "C")
        self.assertEqual(label_from_chroma(a_minor), "Am")
        self.assertEqual(label_from_chroma(g_major), "G")

    def test_silence_and_single_notes_are_no_chord(self):
        self.assertEqual(label_from_chroma(np.zeros(12)), NO_CHORD)
        one = np.zeros(12); one[3] = 1
        self.assertEqual(label_from_chroma(one), NO_CHORD)

    def test_midi_to_labels(self):
        notes = held([60, 64, 67], 0, 2) + held([62, 65, 69], 2, 4)      # C then Dm
        self.assertEqual(labels_from_midi(notes, [1.0, 3.0, 5.0]), ["C", "Dm", NO_CHORD])

    def test_scoring_ignores_frames_with_no_ground_truth_chord(self):
        truth = ["C", "Am", NO_CHORD, "G"]
        estimate = ["C", "A", "C", NO_CHORD]
        result = score_labels(truth, estimate)
        self.assertEqual(result["frames"], 3)
        self.assertAlmostEqual(result["exact"], 1 / 3)
        self.assertAlmostEqual(result["root_only"], 2 / 3)      # Am vs A shares the root
        self.assertAlmostEqual(result["said_no_chord"], 1 / 3)
        self.assertEqual(score_labels([NO_CHORD], ["C"])["frames"], 0)

    def test_root_helper(self):
        self.assertEqual(root_of("F#m"), "F#")
        self.assertEqual(root_of("F#"), "F#")
        self.assertEqual(root_of(NO_CHORD), NO_CHORD)

    def test_chordal_share_separates_lead_lines_from_chord_playing(self):
        chords = held([60, 64, 67], 0, 10)
        lead = [{"pitch": 60 + i, "start": i, "end": i + 0.5} for i in range(10)]
        times = np.arange(0, 10, 0.5)
        self.assertGreater(chordal_share(chords, times), 0.9)
        self.assertLess(chordal_share(lead, times), 0.1)


if __name__ == "__main__":
    unittest.main()
