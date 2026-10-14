"""CPU-only metric startup pose proposal from existing stereo depth and IMU."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mast3r_slam.stereo_depth import solve_metric_keyframe_pnp


def _gray_image(image):
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0 or array.dtype != np.uint8:
        return None
    return array


def _valid_intrinsics(K):
    camera = np.asarray(K, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        return None
    if camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0:
        return None
    if camera[0, 1] != 0.0 or camera[1, 0] != 0.0:
        return None
    if not np.allclose(camera[2], [0.0, 0.0, 1.0], atol=1e-12):
        return None
    return camera


def _valid_rotation(matrix):
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        return None
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        return None
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        return None
    return Rotation.from_matrix(rotation)


def _backproject(xy, z, K):
    return np.column_stack(
        (
            (xy[:, 0] - K[0, 2]) * z / K[0, 0],
            (xy[:, 1] - K[1, 2]) * z / K[1, 1],
            z,
        )
    )


def solve_startup_metric_pose(
    previous_gray,
    current_gray,
    depth,
    K,
    imu_previous_to_current,
    cfg,
):
    """Return a source-camera -> current-camera metric pose proposal.

    This helper only measures local two-view geometry. It does not decide
    startup/steady-state policy, interpolate GT, compute stereo depth, or write
    graph/map state.
    """
    cfg = {} if cfg is None else cfg
    report = {"accepted": False}
    camera = _valid_intrinsics(K)
    if camera is None:
        return None, dict(report, reason="invalid_intrinsics")

    previous = _gray_image(previous_gray)
    current = _gray_image(current_gray)
    if previous is None or current is None or previous.shape != current.shape:
        return None, dict(report, reason="invalid_image_shape")
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != previous.shape:
        return None, dict(report, reason="invalid_depth_shape")

    if imu_previous_to_current is None:
        return None, dict(report, reason="missing_imu_prior")
    imu_rotation = _valid_rotation(imu_previous_to_current)
    if imu_rotation is None:
        return None, dict(report, reason="invalid_imu_prior")

    maximum_points = int(cfg.get("stereo_keyframe_pnp_max_points", 5000))
    features = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=maximum_points,
        qualityLevel=0.01,
        minDistance=8,
    )
    detected = 0 if features is None else int(len(features))
    report["lk_detected_points"] = detected
    if features is None or detected == 0:
        return None, dict(report, reason="no_lk_features")

    tracked, status_fwd, _ = cv2.calcOpticalFlowPyrLK(previous, current, features, None)
    if tracked is None or status_fwd is None:
        return None, dict(report, reason="lk_forward_failed")
    retracked, status_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, tracked, None)
    if retracked is None or status_back is None:
        return None, dict(report, reason="lk_backward_failed")

    source_xy = features.reshape(-1, 2)
    current_xy = tracked.reshape(-1, 2)
    retracked_xy = retracked.reshape(-1, 2)
    height, width = previous.shape
    fb_error = np.linalg.norm(source_xy - retracked_xy, axis=1)
    valid_lk = (
        (status_fwd.reshape(-1) == 1)
        & (status_back.reshape(-1) == 1)
        & np.isfinite(source_xy).all(axis=1)
        & np.isfinite(current_xy).all(axis=1)
        & np.isfinite(retracked_xy).all(axis=1)
        & (fb_error <= 1.0)
        & (current_xy[:, 0] >= 0.0)
        & (current_xy[:, 0] < width)
        & (current_xy[:, 1] >= 0.0)
        & (current_xy[:, 1] < height)
    )
    report["lk_bidirectional_points"] = int(np.count_nonzero(valid_lk))
    if not np.any(valid_lk):
        return None, dict(report, reason="no_bidirectional_lk_tracks")

    source_xy = source_xy[valid_lk]
    current_xy = current_xy[valid_lk]
    source_cells = np.rint(source_xy).astype(np.int64)
    inside_source = (
        (source_cells[:, 0] >= 0)
        & (source_cells[:, 0] < width)
        & (source_cells[:, 1] >= 0)
        & (source_cells[:, 1] < height)
    )
    source_xy = source_xy[inside_source]
    current_xy = current_xy[inside_source]
    source_cells = source_cells[inside_source]

    z = depth[source_cells[:, 1], source_cells[:, 0]]
    valid_depth = np.isfinite(z) & (z >= 0.15) & (z <= 0.65)
    source_xy = source_xy[valid_depth]
    current_xy = current_xy[valid_depth]
    z = z[valid_depth]
    report["metric_points"] = int(len(z))

    object_points = _backproject(source_xy, z.astype(np.float64), camera)
    reprojection_error_px = float(cfg.get("stereo_keyframe_pnp_reprojection_error_px", 2.0))
    pose, pnp = solve_metric_keyframe_pnp(
        object_points,
        current_xy,
        camera,
        minimum_points=int(cfg.get("stereo_keyframe_pnp_min_points", 100)),
        minimum_inlier_ratio=float(cfg.get("stereo_keyframe_pnp_min_inlier_ratio", 0.4)),
        reprojection_error_px=reprojection_error_px,
    )
    report.update({"pnp": pnp})
    if pose is None:
        return None, dict(report, reason=pnp.get("reason", "metric_pnp_failed"))
    if np.asarray(pose).shape != (4, 4) or not np.isfinite(pose).all():
        return None, dict(report, reason="invalid_metric_pnp_pose")
    pnp_rotation = _valid_rotation(pose[:3, :3])
    if pnp_rotation is None:
        return None, dict(report, reason="invalid_metric_pnp_rotation")
    if float(pnp.get("reprojection_p95_px", np.inf)) > reprojection_error_px:
        return None, dict(report, reason="metric_pnp_reprojection_high")

    translation_m = float(np.linalg.norm(pose[:3, 3]))
    report["translation_m"] = translation_m
    if translation_m > 0.03:
        return None, dict(report, reason="translation_too_large")

    disagreement_deg = float(
        np.rad2deg((pnp_rotation.inv() * imu_rotation).magnitude())
    )
    report["imu_rotation_disagreement_deg"] = disagreement_deg
    if disagreement_deg > 1.5:
        return None, dict(report, reason="imu_rotation_disagreement")

    return pose, dict(report, accepted=True, reason="accepted")
