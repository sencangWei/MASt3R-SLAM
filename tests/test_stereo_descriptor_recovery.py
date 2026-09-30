import numpy as np
from scipy.spatial.transform import Rotation

from mast3r_slam.stereo_descriptor_recovery import (
    imu_rotation_agrees,
    metric_descriptor_pose,
)


def test_metric_descriptor_pose_accepts_consistent_stereo_motion():
    height, width = 80, 120
    yy, xx = np.indices((height, width))
    depth = (0.25 + 0.0005 * xx + 0.0003 * yy).astype(np.float32)
    source_xy = np.column_stack((xx[::2, ::2].ravel(), yy[::2, ::2].ravel()))
    source_xy = source_xy[(source_xy[:, 0] < width - 5)]
    target_xy = source_xy + np.array([2, 0])
    target_depth = np.full_like(depth, np.nan)
    target_depth[target_xy[:, 1], target_xy[:, 0]] = depth[
        source_xy[:, 1], source_xy[:, 0]
    ]
    K = np.array([[200.0, 0.0, width / 2],
                  [0.0, 200.0, height / 2], [0.0, 0.0, 1.0]])
    pose, scale, report = metric_descriptor_pose(
        source_xy, target_xy, depth, target_depth,
        depth / 0.4, np.full_like(depth, 2.0), K,
    )
    assert report["accepted"]
    assert report["inlier_points"] >= 100
    assert np.isclose(scale, 0.4, atol=1e-5)
    assert pose[0, 3] > 0.001


def test_metric_descriptor_pose_rejects_missing_depth():
    depth = np.full((40, 40), np.nan)
    xy = np.column_stack(np.indices((40, 40)).reshape(2, -1))
    pose, scale, report = metric_descriptor_pose(
        xy, xy, depth, depth, depth, depth,
        np.array([[200.0, 0, 20], [0, 200.0, 20], [0, 0, 1]]),
    )
    assert pose is None and scale is None
    assert report["reason"] == "too_few_stereo_matches"


def test_imu_gate_uses_inverse_pnp_rotation():
    source_to_target = np.eye(4)
    source_to_target[:3, :3] = Rotation.from_euler("z", 1, degrees=True).as_matrix()
    prior = Rotation.from_euler("z", -1, degrees=True).as_quat()
    accepted, disagreement = imu_rotation_agrees(source_to_target, prior)
    assert accepted and disagreement < 1e-6
    rejected, disagreement = imu_rotation_agrees(
        source_to_target, Rotation.from_euler("z", 3, degrees=True).as_quat()
    )
    assert not rejected and disagreement > 3.0
