"""The model choice is a small thing, but a typo here would silently benchmark the wrong model."""
import unittest

from separators import CHOICES, RUN_SUFFIX, make_separator


class SeparatorChoiceTests(unittest.TestCase):
    def test_every_choice_has_a_run_folder_name(self):
        self.assertEqual(set(CHOICES), set(RUN_SUFFIX))
        self.assertEqual(len(set(RUN_SUFFIX.values())), len(CHOICES))

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            make_separator("wav2vec")


if __name__ == "__main__":
    unittest.main()
