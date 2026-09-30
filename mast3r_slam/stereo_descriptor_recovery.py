"""Opt-in metric recovery when pointmap-gated temporal tracking loses a frame.

Only onboard stereo images and the MASt3R descriptor decoder are used here.
The caller must still validate whole-trajectory scale and pose-graph quality.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mast3r_slam.stereo_depth import (
    robust_pointmap_metric_scale,
    solve_metric_keyframe_pnp,
)


def metric_descriptor_pose(source_xy, target_xy, source_depth, target_depth,
                           source_pointmap_z, source_confidence, camera_matrix,
                           minimum_confidence=1.5):
    """Return source-to-target metric SE(3), model-to-meter scale, and gates."""
    source_xy = np.asarray(source_xy, dtype=np.int32).reshape(-1, 2)
    target_xy = np.asarray(target_xy, dtype=np.int32).reshape(-1, 2)
    source_depth = np.asarray(source_depth)
    target_depth = np.asarray(target_depth)
    source_pointmap_z = np.asarray(source_pointmap_z)
    source_confidence = np.asarray(source_confidence)
    K = np.asarray(camera_matrix, dtype=np.float64)
    if source_xy.shape != target_xy.shape or source_depth.shape != target_depth.shape:
        raise ValueError("descriptor matches and stereo depth shapes must agree")
    if source_depth.shape != source_pointmap_z.shape or source_depth.shape != source_confidence.shape:
        raise ValueError("pointmap and metric depth grids must agree")
    height, width = source_depth.shape
    inside = (
        (source_xy[:, 0] >= 0) & (source_xy[:, 0] < width)
        & (source_xy[:, 1] >= 0) & (source_xy[:, 1] < height)
        & (target_xy[:, 0] >= 0) & (target_xy[:, 0] < width)
        & (target_xy[:, 1] >= 0) & (target_xy[:, 1] < height)
    )
    source_xy, target_xy = source_xy[inside], target_xy[inside]
    z = source_depth[source_xy[:, 1], source_xy[:, 0]]
    good = np.isfinite(z) & (z >= 0.15) & (z <= 0.65)
    good &= source_confidence[source_xy[:, 1], source_xy[:, 0]] >= minimum_confidence
    source_xy, target_xy, z = source_xy[good], target_xy[good], z[good]
    report = {"descriptor_matches": int(len(inside)), "metric_matches": int(len(z))}
    if len(z) < 100:
        return None, None, dict(report, accepted=False, reason="too_few_stereo_matches")

    xyz = np.column_stack((
        (source_xy[:, 0] - K[0, 2]) * z / K[0, 0],
        (source_xy[:, 1] - K[1, 2]) * z / K[1, 1],
        z,
    ))
    pose, pnp = solve_metric_keyframe_pnp(
        xyz, target_xy, K, minimum_points=100,
        minimum_inlier_ratio=0.65, reprojection_error_px=2.0,
    )
    report.update(pnp)
    if pose is None:
        return None, None, report
    projected = xyz @ pose[:3, :3].T + pose[:3, 3]
    uv = np.column_stack((
        K[0, 0] * projected[:, 0] / projected[:, 2] + K[0, 2],
        K[1, 1] * projected[:, 1] / projected[:, 2] + K[1, 2],
    ))
    target_z = target_depth[target_xy[:, 1], target_xy[:, 0]]
    reprojection = np.linalg.norm(uv - target_xy, axis=1)
    depth_valid = (np.isfinite(target_z) & (target_z >= 0.15)
                   & (target_z <= 0.65) & (reprojection <= 2.0)
                   & (projected[:, 2] > 0))
    depth_error = np.abs(projected[depth_valid, 2] - target_z[depth_valid])
    report["depth_consistent_points"] = int(len(depth_error))
    if len(depth_error) < 100:
        return None, None, dict(report, accepted=False, reason="too_few_depth_checks")
    report["depth_median_m"] = float(np.median(depth_error))
    report["depth_p95_m"] = float(np.percentile(depth_error, 95))
    if report["reprojection_p95_px"] > 1.5 or report["depth_median_m"] > 0.006 or report["depth_p95_m"] > 0.025:
        return None, None, dict(report, accepted=False, reason="inconsistent_stereo_geometry")

    scale = robust_pointmap_metric_scale(
        source_pointmap_z, source_confidence, source_depth,
        minimum_points=500, minimum_depth_m=0.15,
        maximum_depth_m=0.65, minimum_confidence=minimum_confidence,
    )
    report["pointmap_scale"] = scale
    if not scale.get("accepted") or scale["relative_mad"] > 0.15:
        return None, None, dict(report, accepted=False, reason="uncertain_pointmap_scale")
    return pose, scale["scale"], dict(report, accepted=True)


def imu_rotation_agrees(source_to_target, prior_quaternion_xyzw, maximum_deg=1.5):
    """Check metric visual rotation against the onboard gyro increment."""
    if prior_quaternion_xyzw is None:
        return False, float("inf")
    # The CSV prior is the right-multiplied camera increment (source <- target).
    # PnP returns the opposite transform (target <- source).
    visual = Rotation.from_matrix(source_to_target[:3, :3]).inv()
    prior = Rotation.from_quat(prior_quaternion_xyzw)
    disagreement = float(np.rad2deg((visual * prior.inv()).magnitude()))
    return disagreement <= maximum_deg, disagreement
