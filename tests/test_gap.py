"""Numerical checks for the model, measurement reference, windows, and metrics."""

import unittest

import torch

from gap import GAP, GAPConfig, measurement_reference, pose_metrics
from gap.data import GAPDataset, causal_window_indices


class GAPTests(unittest.TestCase):
    def test_zero_initialized_output_is_initial_pose(self):
        model = GAP(GAPConfig(channels=8)).eval()
        poses = torch.randn(2, 9, 22, 3)
        references = torch.randn(2, 9, 3)
        with torch.no_grad():
            prediction = model(poses, references)
        self.assertTrue(torch.equal(prediction["pred_pose_absolute"], poses[:, -1]))
        self.assertEqual(torch.count_nonzero(prediction["joint_residual"]).item(), 0)

    def test_median_ignores_padding_and_preserves_even_sample_convention(self):
        points = torch.tensor([[1., 5., 3., 0.], [3., 1., 7., 0.], [0., 0., 0., 0.]])
        reference, used = measurement_reference(points, torch.ones(3))
        self.assertTrue(bool(used))
        self.assertTrue(torch.equal(reference, torch.tensor([1., 1., 3.])))
        reference, used = measurement_reference(torch.zeros(0, 4), torch.ones(3))
        self.assertFalse(bool(used))
        self.assertTrue(torch.equal(reference, torch.ones(3)))

    def test_window_order_boundaries_and_gaps(self):
        ids = torch.tensor([[1, 2, 8], [1, 2, 1], [2, 2, 0], [1, 2, 4]])
        windows = causal_window_indices(ids, 3)
        self.assertTrue(torch.equal(windows[0], torch.tensor([1, 3, 0])))
        self.assertTrue(torch.equal(windows[1], torch.tensor([1, 1, 1])))
        self.assertTrue(torch.equal(windows[2], torch.tensor([2, 2, 2])))
        with self.assertRaises(ValueError):
            causal_window_indices(torch.tensor([[0, 0, 1], [0, 0, 1]]))

    def test_pelvis_translation_and_relative_configuration_are_distinct(self):
        target = torch.zeros(2, 22, 3)
        pred = target + torch.tensor([0.1, 0., 0.])
        metrics = pose_metrics(pred, target)
        self.assertAlmostEqual(metrics["absolute_mpjpe_mm"], 100., places=4)
        self.assertAlmostEqual(metrics["root_error_mm"], 100., places=4)
        self.assertEqual(metrics["root_relative_mpjpe_mm"], 0.)
        pred = target.clone()
        pred[:, 1, 0] = 0.22
        metrics = pose_metrics(pred, target)
        self.assertAlmostEqual(metrics["absolute_mpjpe_mm"], 10., places=4)
        self.assertAlmostEqual(metrics["root_relative_mpjpe_mm"], 10., places=4)
        self.assertEqual(metrics["root_error_mm"], 0.)

    def test_all_samples_have_equal_weight(self):
        target = torch.zeros(3, 22, 3)
        pred = target.clone()
        pred[:, :, 0] = torch.tensor([0.01, 0.02, 0.09])[:, None]
        self.assertAlmostEqual(pose_metrics(pred, target)["absolute_mpjpe_mm"], 40., places=4)

    def test_unit_conversion_and_pose_root_inputs(self):
        ids = torch.tensor([[0, 0, 0], [0, 0, 1]])
        pose = torch.ones(2, 22, 3) * 1000
        ds = GAPDataset(
            {"indicator": ids, "coarse_pose": pose},
            {"indicator": ids, "target_pose": pose},
            units="mm",
            reference="pose-root",
        )
        self.assertTrue(torch.equal(ds[0]["coarse_pose_seq"], torch.ones(9, 22, 3)))
        self.assertTrue(torch.equal(ds[0]["measurement_reference_seq"], torch.ones(9, 3)))


if __name__ == "__main__":
    unittest.main()
