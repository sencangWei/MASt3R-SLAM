import csv

import lietorch
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mast3r_slam.metric_keyframe_prior import MetricKeyframePrior


def test_metric_targets_align_first_valid_keyframe_without_extrapolating(tmp_path):
    path = tmp_path / "camera.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("input_index", "valid", "x", "y", "z", "qx", "qy", "qz", "qw"))
        writer.writerow((0, 0, "", "", "", "", "", "", ""))
        writer.writerow((1, 1, 1, 0, 0, 0, 0, 0, 1))
        writer.writerow((2, 1, 2, 0, 0, 0, 0, 0, 1))
    prior = MetricKeyframePrior(path)
    yaw = Rotation.from_euler("z", 90, degrees=True).as_quat()
    visual = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0.2, 0.3, 0.4, *yaw, 1],
        [0.5, 0.6, 0.7, *yaw, 0.4],
    ])
    targets, valid = prior.targets([0, 1, 2], visual)
    np.testing.assert_array_equal(valid, [False, True, True])
    np.testing.assert_allclose(targets[1, :3], visual[1, :3], atol=1e-7)
    np.testing.assert_allclose(targets[2, :3], visual[1, :3] + [0, 1, 0], atol=1e-7)
    np.testing.assert_array_equal(targets[:, 3], [1, 1, 1])


def test_metric_position_and_scale_jacobian_matches_left_sim3_retraction():
    pose = lietorch.Sim3.exp(torch.tensor([[0.01, -0.02, 0.03, 0.03, 0.02, -0.01, -0.1]]))
    x, y, z = pose.data[0, :3]
    jacobian = torch.tensor([
        [1, 0, 0, 0, z, -y, x],
        [0, 1, 0, -z, 0, x, y],
        [0, 0, 1, y, -x, 0, z],
    ])
    eps = 1e-4
    for axis in range(7):
        step = torch.zeros((1, 7))
        step[0, axis] = eps
        plus = pose.retr(step).data[0]
        minus = pose.retr(-step).data[0]
        numerical = (plus[:3] - minus[:3]) / (2 * eps)
        assert torch.allclose(jacobian[:, axis], numerical, atol=3e-4)
        scale_derivative = (torch.log(plus[7]) - torch.log(minus[7])) / (2 * eps)
        assert abs(float(scale_derivative) - float(axis == 6)) < 3e-4
