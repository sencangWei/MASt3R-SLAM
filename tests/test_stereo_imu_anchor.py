import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mast3r_slam.stereo_imu_anchor import integrate_imu_rotations, propose_anchor_pose


def test_imu_preintegration_uses_keyframe_relative_right_composition():
    steps = [Rotation.identity(), Rotation.from_euler("x", 30, degrees=True),
             Rotation.from_euler("z", 20, degrees=True)]
    result = integrate_imu_rotations([step.as_quat() for step in steps])
    expected = steps[1]*steps[2]
    assert (result[2].inv()*expected).magnitude() < 1e-12
    assert (result[2].inv()*(steps[2]*steps[1])).magnitude() > 0.1
    with pytest.raises(ValueError, match="first"):
        integrate_imu_rotations([steps[1].as_quat(), steps[2].as_quat()])


def test_anchor_pose_requires_stereo_pnp_and_imu_agreement():
    height = width = 120
    depth = np.full((height, width), 0.3, np.float32)
    K = np.array([[100, 0, 60], [0, 100, 60], [0, 0, 1]], dtype=float)
    pose = np.r_[np.zeros(3), Rotation.from_euler("z", 1.5, degrees=True).as_quat(), 1.0]
    valid = np.ones(height*width, bool)
    mapping = np.arange(height*width)
    learned_keyframe = np.zeros((height*width, 3))
    learned_keyframe[:, 2] = 2.4
    learned_current = learned_keyframe.copy()
    learned_current[:, 2] = 1.2
    corrected, report = propose_anchor_pose(
        pose, Rotation.identity(), valid, mapping, learned_keyframe,
        learned_current, K, depth, depth)
    assert report["reason"] == "accepted"
    np.testing.assert_allclose(corrected[:3], 0, atol=1e-9)
    assert Rotation.from_quat(corrected[3:7]).magnitude() < 1e-9
    assert corrected[7] == pytest.approx(2.0)
    assert report["pnp_translation_delta_mm"] < 1e-6
    unchanged = pose.copy()
    unchanged[3:7] = Rotation.from_euler("z", 0.5, degrees=True).as_quat()
    rejected, report = propose_anchor_pose(
        unchanged, Rotation.identity(), valid, mapping, learned_keyframe,
        learned_current, K, depth, depth)
    assert rejected is None and report["reason"] == "visual_imu_within_guard"


def test_anchor_pose_rejects_pnp_translation_disagreement(monkeypatch):
    from mast3r_slam import stereo_imu_anchor

    def inconsistent_pnp(*_args):
        transform = np.eye(4)
        transform[0, 3] = 0.01
        return transform, dict(inlier_ratio=0.9, reprojection_p95_px=1.0)

    monkeypatch.setattr(stereo_imu_anchor, "solve_metric_keyframe_pnp", inconsistent_pnp)
    depth = np.full((120, 120), 0.3, np.float32)
    points = np.zeros((120*120, 3))
    points[:, 2] = 2.4
    K = np.array([[100, 0, 60], [0, 100, 60], [0, 0, 1]], dtype=float)
    pose = np.r_[np.zeros(3), Rotation.from_euler("z", 1.5, degrees=True).as_quat(), 1.0]
    corrected, report = propose_anchor_pose(
        pose, Rotation.identity(), np.ones(120*120, bool),
        np.arange(120*120), points, points, K, depth, depth)
    assert corrected is None
    assert report["reason"] == "stereo_pnp_translation_inconsistent"
