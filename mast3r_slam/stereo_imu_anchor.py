"""Opt-in, source-only stereo/IMU cross-check for a frontend anchor pose."""

import numpy as np
from scipy.spatial.transform import Rotation

from mast3r_slam.stereo_depth import solve_metric_keyframe_pnp


def integrate_imu_rotations(increments):
    increments = np.asarray(increments, dtype=np.float64)
    if increments.ndim != 2 or increments.shape[1] != 4 or len(increments) == 0:
        raise ValueError("expected Nx4 incremental IMU rotations")
    if not np.all(np.isfinite(increments)) or not np.allclose(
            np.linalg.norm(increments, axis=1), 1.0, atol=1e-5):
        raise ValueError("invalid incremental IMU rotations")
    if Rotation.from_quat(increments[0]).magnitude() > 1e-5:
        raise ValueError("first IMU rotation increment must be identity")
    current = Rotation.identity()
    result = []
    for increment in increments:
        current = current * Rotation.from_quat(increment)
        result.append(current)
    return result


def _backproject(ids, depth, K, width):
    x, y = ids % width, ids // width
    return np.column_stack(((x-K[0, 2])*depth/K[0, 0],
                            (y-K[1, 2])*depth/K[1, 1], depth))


def propose_anchor_pose(relative_pose, imu_rotation, valid, current_ids,
                        keyframe_points, current_points, K,
                        depth_keyframe, depth_current):
    """Return a new relative Sim3 vector only with three-sensor agreement.

    No external GT enters this decision. The visual pose is left untouched on
    every rejected frame; thresholds encode the fixed diagnostic witness.
    """
    pose = np.asarray(relative_pose, dtype=np.float64).reshape(8)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    depth_k, depth_f = np.asarray(depth_keyframe), np.asarray(depth_current)
    if depth_k.shape != depth_f.shape or depth_k.ndim != 2:
        raise ValueError("stereo depth shape mismatch")
    h, w = depth_k.shape
    mask = np.asarray(valid).reshape(-1).astype(bool)
    mapping = np.asarray(current_ids).reshape(-1)
    points = np.asarray(keyframe_points)
    current = np.asarray(current_points)
    if (mask.size != h*w or mapping.size != h*w
            or points.shape != (h*w, 3) or current.shape != (h*w, 3)):
        raise ValueError("matched point/image shape mismatch")
    visual_rotation = Rotation.from_quat(pose[3:7])
    visual_imu_deg = (visual_rotation.inv()*imu_rotation).magnitude()*180/np.pi
    if visual_imu_deg <= 1.0:
        return None, dict(reason="visual_imu_within_guard", visual_imu_deg=float(visual_imu_deg))
    ids = np.flatnonzero(mask)
    mapped = mapping[ids].astype(np.int64)
    if np.any(mapped < 0) or np.any(mapped >= h*w):
        raise ValueError("matched current pixel outside image")
    dk, df = depth_k.ravel()[ids], depth_f.ravel()[mapped]
    stereo_valid = np.isfinite(dk) & np.isfinite(df) & (dk > 0) & (df > 0)
    ids_pair, mapped_pair, dk_pair, df_pair = (array[stereo_valid] for array in
                                             (ids, mapped, dk, df))
    if len(ids_pair) < 10000:
        return None, dict(reason="insufficient_stereo_pairs", points=int(len(ids_pair)))
    candidates = _backproject(ids_pair, dk_pair, K, w) - imu_rotation.apply(
        _backproject(mapped_pair, df_pair, K, w))
    metric_t = np.median(candidates, axis=0)
    quadrant = (ids_pair // w >= h//2).astype(int)*2 + (ids_pair % w >= w//2).astype(int)
    tiles = [np.median(candidates[quadrant == index], axis=0)
             for index in range(4) if np.count_nonzero(quadrant == index) >= 100]
    if len(tiles) != 4:
        return None, dict(reason="insufficient_spatial_coverage", tiles=len(tiles))
    tile_spread_mm = float(np.max(np.linalg.norm(np.asarray(tiles)-metric_t, axis=1))*1000)
    if tile_spread_mm >= 2.0:
        return None, dict(reason="stereo_spatial_inconsistent", tile_spread_mm=tile_spread_mm)

    anchor_valid = np.flatnonzero(np.isfinite(dk) & (dk > 0))
    selected = anchor_valid[np.linspace(0, len(anchor_valid)-1,
                                        min(5000, len(anchor_valid))).astype(int)]
    object_points = _backproject(ids[selected], dk[selected], K, w)
    image_points = np.column_stack((mapped[selected] % w, mapped[selected] // w)).astype(np.float32)
    keyframe_to_current, pnp = solve_metric_keyframe_pnp(
        object_points, image_points, K, 100, 0.3, 2.0)
    if keyframe_to_current is None:
        return None, dict(reason="stereo_pnp_failed", pnp=pnp)
    pnp_rotation = Rotation.from_matrix(keyframe_to_current[:3, :3]).inv()
    pnp_imu_deg = float((pnp_rotation.inv()*imu_rotation).magnitude()*180/np.pi)
    pnp_visual_deg = float((pnp_rotation.inv()*visual_rotation).magnitude()*180/np.pi)
    if not (pnp_imu_deg < 0.6 and pnp_visual_deg > 0.8
            and pnp["inlier_ratio"] > 0.8 and pnp["reprojection_p95_px"] < 1.5):
        return None, dict(reason="stereo_pnp_not_imu_consensus", pnp=pnp,
                          pnp_imu_deg=pnp_imu_deg, pnp_visual_deg=pnp_visual_deg)
    pnp_metric_t = -keyframe_to_current[:3, :3].T @ keyframe_to_current[:3, 3]
    pnp_translation_delta_mm = float(np.linalg.norm(pnp_metric_t-metric_t)*1000)
    if pnp_translation_delta_mm >= 5.0:
        return None, dict(reason="stereo_pnp_translation_inconsistent",
                          pnp_translation_delta_mm=pnp_translation_delta_mm, pnp=pnp)
    scale_valid = (np.isfinite(dk) & (dk > 0) & np.isfinite(df) & (df > 0)
                   & np.isfinite(points[ids, 2]) & (points[ids, 2] > 0)
                   & np.isfinite(current[ids, 2]) & (current[ids, 2] > 0))
    if np.count_nonzero(scale_valid) < 1000:
        return None, dict(reason="insufficient_metric_scale_points")
    alpha_k = float(np.median(dk[scale_valid]/points[ids[scale_valid], 2]))
    alpha_f = float(np.median(df[scale_valid]/current[ids[scale_valid], 2]))
    if not (np.isfinite(alpha_k) and alpha_k > 0
            and np.isfinite(alpha_f) and alpha_f > 0):
        raise ValueError("invalid stereo/pointmap scales")
    updated = pose.copy()
    updated[:3] = metric_t/alpha_k
    updated[3:7] = imu_rotation.as_quat()
    updated[7] = alpha_f/alpha_k
    return updated, dict(reason="accepted", visual_imu_deg=float(visual_imu_deg),
                         pnp_imu_deg=pnp_imu_deg, pnp_visual_deg=pnp_visual_deg,
                         metric_translation_delta_mm=float(np.linalg.norm(alpha_k*pose[:3]-metric_t)*1000),
                         tile_spread_mm=tile_spread_mm, stereo_points=int(len(ids_pair)),
                         metric_per_keyframe_unit=alpha_k,
                         metric_per_current_unit=alpha_f,
                         pnp_translation_delta_mm=pnp_translation_delta_mm, pnp=pnp)
