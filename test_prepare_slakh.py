"""Small synthetic regressions for label integrity and official split handling."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml

from prepare_slakh import SAMPLE_RATE, RunAlreadyActive, RunLock, discover_tracks, is_guitar, load_prepared_audio, main, prepare_track, scan_dataset


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.out = self.root / "out"
        for split in ("train", "validation", "test", "omitted"):
            (self.raw / split).mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def track(self, split="train", name="Track00001", uuid="unique", gain=0.2):
        path = self.raw / split / name
        (path / "stems").mkdir(parents=True)
        # Source amplitudes ALREADY include Slakh's gain; don't apply it again.
        t = np.arange(4096)
        guitar = (0.1 * np.sin(t * 0.017)).astype(np.float32)
        other = (0.15 * np.cos(t * 0.03)).astype(np.float32)
        sf.write(path / "mix.flac", guitar + other, SAMPLE_RATE, subtype="PCM_16")
        sf.write(path / "stems/S00.flac", guitar, SAMPLE_RATE, subtype="PCM_16")
        sf.write(path / "stems/S01.flac", other, SAMPLE_RATE, subtype="PCM_16")
        metadata = {"UUID": uuid, "overall_gain": gain, "stems": {
            "S00": {"audio_rendered": True, "inst_class": "Guitar", "program_num": 24, "is_drum": False},
            "S01": {"audio_rendered": True, "inst_class": "Piano", "program_num": 0, "is_drum": False},
            "S02": {"audio_rendered": False, "inst_class": "Guitar", "program_num": 25, "is_drum": False},
        }}
        (path / "metadata.yaml").write_text(yaml.safe_dump(metadata))
        return {"split": split, "track_id": name, "uuid": uuid, "raw_track": str(path)}, path

    def test_gain_already_applied_reconstruction_and_stereo_contract(self):
        row, path = self.track()
        result = prepare_track(row, self.out)
        self.assertEqual(result["status"], "ready", result)
        raw_guitar, _ = sf.read(path / "stems/S00.flac", dtype="float32")
        prepared_guitar, _ = sf.read(result["guitar_path"], dtype="float32")
        np.testing.assert_array_equal(raw_guitar, prepared_guitar)
        self.assertEqual(result["additional_gain_applied"], 1.0)
        self.assertEqual(result["unrendered_guitar_stems"], ["S02"])
        mix, targets = load_prepared_audio(result, start_frame=100, frames=500)
        self.assertEqual(mix.shape, (2, 500))
        self.assertEqual(targets.shape, (2, 2, 500))
        self.assertEqual(targets.dtype, np.float32)
        np.testing.assert_array_equal(mix[0], mix[1])
        np.testing.assert_allclose(targets.sum(axis=0), mix, rtol=0, atol=1e-7)
        with self.assertRaises(ValueError):
            load_prepared_audio(result, frames=5000)

    def test_official_splits_preserved_and_omitted_excluded(self):
        for index, split in enumerate(("train", "validation", "test", "omitted")):
            self.track(split, f"Track{index:05d}", f"uuid{index}")
        self.assertEqual([split for split, _ in discover_tracks(self.raw)], ["train", "validation", "test"])
        rows, audit = scan_dataset(self.raw)
        self.assertEqual(audit["omitted_excluded"], 1)
        self.assertEqual(audit["official_counts"], {"train": 1, "validation": 1, "test": 1})
        for row in rows:
            result = prepare_track(row, self.out)
            self.assertEqual(result["status"], "ready", result)
            self.assertEqual(Path(result["guitar_path"]).parent.parent.name, row["split"])

    def test_cross_split_uuid_duplicates_quarantined(self):
        self.track("train", "Track00001", "duplicate")
        self.track("validation", "Track00002", "duplicate")
        rows, audit = scan_dataset(self.raw)
        self.assertEqual(len(audit["cross_split_uuid_duplicates"]), 1)
        for row in rows:
            result = prepare_track(row, self.out)
            self.assertEqual(result["reason"], "cross_split_uuid_duplicate")

    def test_missing_rendered_guitar_is_rejected(self):
        row, path = self.track()
        (path / "stems/S00.flac").unlink()
        self.assertEqual(prepare_track(row, self.out)["reason"], "missing_rendered_stem")

    def test_source_alignment_is_rejected_without_truncation(self):
        row, path = self.track()
        sf.write(path / "stems/S00.flac", np.ones(200), SAMPLE_RATE)
        self.assertEqual(prepare_track(row, self.out)["reason"], "source_alignment")

    def test_incorrect_source_gain_is_rejected(self):
        row, path = self.track()
        source, sr = sf.read(path / "stems/S00.flac")
        sf.write(path / "stems/S00.flac", source * 0.2, sr)
        self.assertEqual(prepare_track(row, self.out)["reason"], "mixture_stem_mismatch")
        self.assertFalse((self.out / "train/Track00001/record.json").exists())

    def test_silent_guitar_is_rejected(self):
        row, path = self.track()
        other, sr = sf.read(path / "stems/S01.flac")
        sf.write(path / "stems/S00.flac", np.zeros(len(other)), sr)
        sf.write(path / "mix.flac", other, sr)
        self.assertEqual(prepare_track(row, self.out)["reason"], "silent_guitar")

    def test_resume_uses_valid_outputs_and_rebuilds_changes(self):
        row, path = self.track()
        initial = prepare_track(row, self.out)
        resumed = prepare_track(row, self.out)
        self.assertTrue(resumed["resumed"])
        Path(initial["guitar_path"]).write_bytes(b"broken")
        rebuilt = prepare_track(row, self.out)
        self.assertEqual(rebuilt["status"], "ready", rebuilt)
        self.assertFalse(rebuilt.get("resumed", False))
        sf.info(rebuilt["guitar_path"])

    def test_drums_not_misclassified_by_program_number(self):
        self.assertFalse(is_guitar({"is_drum": True, "program_num": 24}))
        self.assertTrue(is_guitar({"program_num": 24}))
        self.assertFalse(is_guitar({"program_num": 32}))

    def test_second_writer_cannot_clobber_active_run_status(self):
        self.out.mkdir()
        run_dir = self.root / "run"
        run_dir.mkdir()
        status = run_dir / "status.json"
        sentinel = json.dumps({"state": "running", "pid": 12345})
        status.write_text(sentinel)
        argv = ["prepare_slakh.py", "--input", str(self.raw), "--out", str(self.out), "--run-dir", str(run_dir)]
        with RunLock(self.out / "prepare.lock"), patch("sys.argv", argv):
            with self.assertRaises(RunAlreadyActive):
                main()
        self.assertEqual(status.read_text(), sentinel)


if __name__ == "__main__":
    unittest.main()
