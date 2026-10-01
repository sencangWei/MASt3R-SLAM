"""Experimental stereo conditioning of a weak MASt3R 3D pair.

The two MASt3R pointmaps are expressed in the first camera frame. Stereo
depth in the second image is transported into that frame before correction.
This module never reads external reference poses.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def spatial_depth_correction(model, metric, valid, cell_size=16):
    """Estimate a smooth 3D residual from alternating image cells.

    The complementary cells are held out to check whether the correction
    generalizes spatially; it is not accepted merely for fitting its inputs.
    """
    height, width = valid.shape
    if model.shape != metric.shape or model.shape != (height, width, 3):
        raise ValueError("pointmaps and validity mask must share an image grid")
    rows, cols = np.indices((height, width))
    training = valid & (((rows // cell_size + cols // cell_size) % 2) == 0)
    holdout = valid & ~training
    n_rows = (height + cell_size - 1) // cell_size
    n_cols = (width + cell_size - 1) // cell_size
    field = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
    known = np.zeros((n_rows, n_cols), dtype=bool)
    difference = metric - model
    for row in range(n_rows):
        for col in range(n_cols):
            ys = slice(row * cell_size, (row + 1) * cell_size)
            xs = slice(col * cell_size, (col + 1) * cell_size)
            patch = training[ys, xs]
            if np.count_nonzero(patch) >= 8:
                field[row, col] = np.median(difference[ys, xs][patch], axis=0)
                known[row, col] = True
    if np.count_nonzero(known) < 10 or np.count_nonzero(holdout) < 100:
        return model, {"accepted": False, "reason": "insufficient_spatial_support"}
    nearest = distance_transform_edt(~known, return_distances=False, return_indices=True)
    field = gaussian_filter(field[tuple(nearest)], sigma=(1.0, 1.0, 0.0))
    corrected = model + cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
    before = np.linalg.norm((model - metric)[holdout], axis=1)
    after = np.linalg.norm((corrected - metric)[holdout], axis=1)
    report = {
        "accepted": True,
        "training_cells": int(np.count_nonzero(known)),
        "holdout_points": int(np.count_nonzero(holdout)),
        "before_p50_mm": float(np.median(before) * 1000),
        "after_p50_mm": float(np.median(after) * 1000),
        "after_p95_mm": float(np.percentile(after, 95) * 1000),
    }
    return corrected, report


def condition_pair(pointmaps, confidences, depths, source_to_target, scale, K,
                   source_xy, target_xy):
    """Return conditioned maps after spatial holdout and matched-3D checks."""
    source_depth, target_depth = (np.asarray(value) for value in depths)
    if source_depth.shape != target_depth.shape:
        raise ValueError("stereo depth grids must have equal shape")
    height, width = source_depth.shape
    if pointmaps.shape != (2, height, width, 3):
        raise ValueError("MASt3R pointmap geometry differs from stereo depth")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("model-to-metric scale must be positive")
    rows, cols = np.indices((height, width))
    K = np.asarray(K, dtype=np.float64)
    transform = np.asarray(source_to_target, dtype=np.float64)
    corrected = []
    reports = []
    for index, depth in enumerate((source_depth, target_depth)):
        finite_depth = np.nan_to_num(depth, nan=0.0)
        metric = np.stack((
            (cols - K[0, 2]) * finite_depth / K[0, 0],
            (rows - K[1, 2]) * finite_depth / K[1, 1],
            finite_depth,
        ), axis=-1)
        if index:
            metric = (metric - transform[:3, 3]) @ transform[:3, :3]
        model = scale * np.asarray(pointmaps[index])
        valid = (
            np.isfinite(depth) & (depth >= 0.15) & (depth <= 0.65)
            & np.isfinite(model).all(axis=-1) & (model[..., 2] > 0)
            & (np.asarray(confidences[index]) >= 1.0)
        )
        candidate, report = spatial_depth_correction(model, metric, valid)
        corrected.append(candidate / scale)
        reports.append(report)
    if not all(report["accepted"] and report["after_p50_mm"] < 10
               and report["after_p50_mm"] < report["before_p50_mm"]
               for report in reports):
        return None, {"accepted": False, "reason": "heldout_geometry_failed", "views": reports}
    source_xy = np.asarray(source_xy, dtype=np.int64).reshape(-1, 2)
    target_xy = np.asarray(target_xy, dtype=np.int64).reshape(-1, 2)
    inside = (
        (source_xy[:, 0] >= 0) & (source_xy[:, 0] < width)
        & (source_xy[:, 1] >= 0) & (source_xy[:, 1] < height)
        & (target_xy[:, 0] >= 0) & (target_xy[:, 0] < width)
        & (target_xy[:, 1] >= 0) & (target_xy[:, 1] < height)
    )
    source_xy, target_xy = source_xy[inside], target_xy[inside]
    source_z = source_depth[source_xy[:, 1], source_xy[:, 0]]
    target_z = target_depth[target_xy[:, 1], target_xy[:, 0]]
    valid_pair = (
        np.isfinite(source_z) & (source_z >= 0.15) & (source_z <= 0.65)
        & np.isfinite(target_z) & (target_z >= 0.15) & (target_z <= 0.65)
    )
    source_xy, target_xy = source_xy[valid_pair], target_xy[valid_pair]
    if len(source_xy) < 100:
        return None, {"accepted": False, "reason": "insufficient_3d_pairs", "views": reports}
    source_points = corrected[0][source_xy[:, 1], source_xy[:, 0]]
    target_points = corrected[1][target_xy[:, 1], target_xy[:, 0]]
    distances = scale * np.linalg.norm(source_points - target_points, axis=1)
    pair_p50_mm = float(np.median(distances) * 1000)
    pair_p95_mm = float(np.percentile(distances, 95) * 1000)
    report = {"views": reports, "pair_points": len(distances),
              "pair_p50_mm": pair_p50_mm, "pair_p95_mm": pair_p95_mm}
    if pair_p50_mm >= 6 or pair_p95_mm >= 25:
        return None, dict(report, accepted=False, reason="matched_3d_geometry_failed")
    return np.stack(corrected).astype(np.float32), dict(report, accepted=True)
