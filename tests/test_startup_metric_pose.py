import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mast3r_slam.startup_metric_pose import solve_startup_metric_pose


def _translated_scene(dx=4, dy=0):
    height, width = 240, 320
    previous = np.zeros((height, width), np.uint8)
    current = np.zeros_like(previous)
    for y in range(16, height - 16, 12):
        for x in range(16, width - 16, 12):
            cv2.circle(previous, (x, y), 2, 255, -1)
            cv2.circle(current, (x + dx, y + dy), 2, 255, -1)
    depth = np.full((height, width), 0.3, np.float32)
    K = np.array(
        [[200.0, 0.0, width / 2], [0.0, 200.0, height / 2], [0.0, 0.0, 1.0]]
    )
    imu = np.eye(3)
    expected_tx = dx * 0.3 / K[0, 0]
    return previous, current, depth, K, imu, expected_tx


def test_startup_metric_pose_accepts_translated_synthetic_images():
    previous, current, depth, K, imu, expected_tx = _translated_scene()
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is not None, report
    assert report["accepted"]
    assert report["lk_bidirectional_points"] >= 100
    assert report["metric_points"] >= 100
    assert report["translation_m"] == pytest.approx(abs(expected_tx), abs=0.002)
    assert pose[0, 3] == pytest.approx(expected_tx, abs=0.002)
    assert report["imu_rotation_disagreement_deg"] <= 1.5


def test_startup_metric_pose_passes_production_pnp_defaults(monkeypatch):
    import mast3r_slam.startup_metric_pose as module

    previous, current, depth, K, imu, _ = _translated_scene()
    seen = {}

    def capture_pnp(*_args, **kwargs):
        seen.update(kwargs)
        return np.eye(4), {"accepted": True, "reprojection_p95_px": 0.2}

    monkeypatch.setattr(module, "solve_metric_keyframe_pnp", capture_pnp)
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is not None, report
    assert seen == {
        "minimum_points": 100,
        "minimum_inlier_ratio": 0.4,
        "reprojection_error_px": 2.0,
    }


@pytest.mark.parametrize(
    ("imu", "reason"),
    [
        (None, "missing_imu_prior"),
        (np.eye(4), "invalid_imu_prior"),
    ],
)
def test_startup_metric_pose_requires_valid_imu_prior(imu, reason):
    previous, current, depth, K, _, _ = _translated_scene()
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is None
    assert report["reason"] == reason


@pytest.mark.parametrize(
    "imu",
    [
        np.diag([1.0, 1.0, 0.98]),
        np.diag([1.0, 1.0, -1.0]),
    ],
)
def test_startup_metric_pose_rejects_non_rotation_imu_matrix(imu):
    previous, current, depth, K, _, _ = _translated_scene()
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is None
    assert report["reason"] == "invalid_imu_prior"


def test_startup_metric_pose_rejects_bad_intrinsics_and_shape():
    previous, current, depth, K, imu, _ = _translated_scene()
    bad_K = K.copy()
    bad_K[0, 0] = np.nan
    pose, report = solve_startup_metric_pose(previous, current, depth, bad_K, imu, {})
    assert pose is None
    assert report["reason"] == "invalid_intrinsics"

    bad_K = K.copy()
    bad_K[0, 1] = 0.01
    pose, report = solve_startup_metric_pose(previous, current, depth, bad_K, imu, {})
    assert pose is None
    assert report["reason"] == "invalid_intrinsics"

    bad_K = K.copy()
    bad_K[2, 2] = 0.99
    pose, report = solve_startup_metric_pose(previous, current, depth, bad_K, imu, {})
    assert pose is None
    assert report["reason"] == "invalid_intrinsics"

    pose, report = solve_startup_metric_pose(previous, current[:-1], depth, K, imu, {})
    assert pose is None
    assert report["reason"] == "invalid_image_shape"


def test_startup_metric_pose_rejects_excessive_motion(monkeypatch):
    import mast3r_slam.startup_metric_pose as module

    previous, current, depth, K, imu, _ = _translated_scene()

    def excessive_pnp(*_args, **_kwargs):
        pose = np.eye(4)
        pose[0, 3] = 0.031
        return pose, {"accepted": True, "reprojection_p95_px": 0.2}

    monkeypatch.setattr(module, "solve_metric_keyframe_pnp", excessive_pnp)
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is None
    assert report["reason"] == "translation_too_large"
    assert report["translation_m"] > 0.03


def test_startup_metric_pose_rejects_high_pnp_residual(monkeypatch):
    import mast3r_slam.startup_metric_pose as module

    previous, current, depth, K, imu, _ = _translated_scene()

    def noisy_pnp(*_args, **_kwargs):
        return np.eye(4), {"accepted": True, "reprojection_p95_px": 2.1}

    monkeypatch.setattr(module, "solve_metric_keyframe_pnp", noisy_pnp)
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is None
    assert report["reason"] == "metric_pnp_reprojection_high"


def test_startup_metric_pose_rejects_invalid_pnp_rotation(monkeypatch):
    import mast3r_slam.startup_metric_pose as module

    previous, current, depth, K, imu, _ = _translated_scene()

    def bad_rotation_pnp(*_args, **_kwargs):
        pose = np.eye(4)
        pose[:3, :3] = np.diag([1.0, 1.0, 0.98])
        return pose, {"accepted": True, "reprojection_p95_px": 0.2}

    monkeypatch.setattr(module, "solve_metric_keyframe_pnp", bad_rotation_pnp)
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is None
    assert report["reason"] == "invalid_metric_pnp_rotation"


def test_startup_metric_pose_rejects_imu_rotation_disagreement():
    previous, current, depth, K, _, _ = _translated_scene()
    imu = Rotation.from_euler("z", 3.0, degrees=True).as_matrix()
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is None
    assert report["reason"] == "imu_rotation_disagreement"
    assert report["imu_rotation_disagreement_deg"] > 1.5


def test_startup_metric_pose_masks_do_not_index_out_of_bounds(monkeypatch):
    previous, current, depth, K, imu, _ = _translated_scene()

    def features(*_args, **_kwargs):
        return np.array([[[10.0, 10.0]], [[319.0, 239.0]], [[20.0, 20.0]]], np.float32)

    def lk(src, dst, pts, *_args, **_kwargs):
        if src is previous:
            tracked = np.array([[[12.0, 10.0]], [[400.0, 239.0]], [[22.0, 20.0]]], np.float32)
        else:
            tracked = np.array([[[10.0, 10.0]], [[319.0, 239.0]], [[40.0, 20.0]]], np.float32)
        return tracked, np.ones((3, 1), np.uint8), None

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", features)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lk)
    pose, report = solve_startup_metric_pose(previous, current, depth, K, imu, {})
    assert pose is None
    assert report["reason"] == "insufficient_metric_correspondences"
    assert report["lk_bidirectional_points"] == 1
