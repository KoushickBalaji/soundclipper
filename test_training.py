"""Regression checks for paired losses, silence handling, and validation scoring."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from train import GuitarDataset, evaluate, si_sdr, source_loss


class EchoSources(torch.nn.Module):
    def forward(self, audio):
        return torch.stack((audio, audio), dim=1)


class TrainingCalculationTests(unittest.TestCase):
    def test_known_signal_to_error_ratio(self):
        ref = torch.tensor([-1., 0., 1., 0.], dtype=torch.float64)
        noise = torch.tensor([0., -1., 0., 1.], dtype=torch.float64)
        estimate = ref + 0.1 * noise
        self.assertAlmostEqual(si_sdr(estimate, ref).item(), 20., places=5)
        self.assertAlmostEqual(si_sdr(estimate * 2, ref).item(), 20., places=5)

    def test_batch_matches_individual_losses_and_gradients(self):
        torch.manual_seed(123)
        ref = torch.randn(3, 1, 128)
        estimate = (ref + 0.01 * torch.randn_like(ref)).requires_grad_()
        batch = source_loss(estimate, ref)
        individual = torch.stack([
            source_loss(estimate[i], ref[i]) for i in range(3)
        ]).mean()
        torch.testing.assert_close(batch, individual)
        grad_batch = torch.autograd.grad(batch, estimate, retain_graph=True)[0]
        grad_individual = torch.autograd.grad(individual, estimate)[0]
        torch.testing.assert_close(grad_batch, grad_individual)
        self.assertLess(batch.item(), -35.)

    def test_broadcasting_is_rejected(self):
        for fn in (si_sdr, source_loss):
            with self.subTest(function=fn.__name__):
                with self.assertRaisesRegex(ValueError, "matching shapes"):
                    fn(torch.ones(3, 128), torch.ones(3, 1, 128))

    def test_silent_scores_are_undefined(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            si_sdr(torch.zeros(128), torch.zeros(128))

    def test_silence_loss_has_finite_gradient_toward_zero(self):
        estimate = torch.full((2, 1, 128), 0.5, requires_grad=True)
        loss = source_loss(estimate, torch.zeros_like(estimate))
        self.assertAlmostEqual(loss.item(), 0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(estimate.grad).all())
        self.assertTrue((estimate.grad > 0).all())
        self.assertEqual(source_loss(torch.zeros(128), torch.zeros(128)).item(), 0.)

    def test_mixed_silent_and_active_batch(self):
        torch.manual_seed(42)
        ref = torch.randn(2, 1, 128)
        ref[1] = 0
        estimate = (ref + 0.1 * torch.randn_like(ref)).requires_grad_()
        loss = source_loss(estimate, ref)
        expected = (-si_sdr(estimate[0], ref[0]).mean()
                    + estimate[1].square().mean()) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(estimate.grad).all())

    def test_bfloat16_energy_sums_use_float32(self):
        torch.manual_seed(1)
        ref = torch.randn(2, 1, 128).bfloat16()
        estimate = ref + (0.1 * torch.randn_like(ref))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            score = si_sdr(estimate, ref)
        self.assertEqual(score.dtype, torch.float32)
        torch.testing.assert_close(score, si_sdr(estimate.float(), ref.float()))

    def test_empty_validation_is_rejected(self):
        valid = SimpleNamespace(tracks=[], sr=8)
        with self.assertRaisesRegex(ValueError, "no tracks"):
            evaluate(EchoSources(), valid, "cpu", 1.)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "valid").mkdir()
            with self.assertRaisesRegex(ValueError, "No tracks"):
                GuitarDataset(tmp, "valid", 8, 1.)

    def test_full_track_scores_are_repeatable_and_exclude_silence(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as tmp:
            tracks = []
            expected_g, expected_o = [], []
            for i in range(2):
                track = Path(tmp) / str(i)
                track.mkdir()
                # 17 samples exercises the final padded chunk (8 + 8 + 1).
                guitar = rng.normal(size=17).astype("float32")
                other = rng.normal(size=17).astype("float32")
                if i == 1:
                    guitar[:] = 0
                # Distinct saved mixture verifies evaluation reads mixture.wav.
                mix = (guitar + other + 0.2 * rng.normal(size=17)).astype("float32")
                for name, values in (("guitar", guitar), ("other", other), ("mixture", mix)):
                    sf.write(track / f"{name}.wav", values, 8, subtype="FLOAT")
                if i == 0:
                    expected_g.append(si_sdr(torch.from_numpy(mix).double(),
                                             torch.from_numpy(guitar).double()).item())
                expected_o.append(si_sdr(torch.from_numpy(mix).double(),
                                         torch.from_numpy(other).double()).item())
                tracks.append(track)
            valid = SimpleNamespace(tracks=tracks, sr=8)
            first = evaluate(EchoSources(), valid, "cpu", 1.)
            second = evaluate(EchoSources(), valid, "cpu", 1.)
            self.assertEqual(first, second)
            self.assertAlmostEqual(first[0], np.mean(expected_g), places=6)
            self.assertAlmostEqual(first[1], np.mean(expected_o), places=6)
            sf.write(tracks[0] / "guitar.wav", np.zeros(17), 8, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "No non-silent guitar"):
                evaluate(EchoSources(), valid, "cpu", 1.)


if __name__ == "__main__":
    unittest.main()
