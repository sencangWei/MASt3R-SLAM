from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch


def force_unit_sim3_scale(pose):
    data = pose.data.clone() if hasattr(pose.data, "clone") else pose.data.copy()
    data[..., 7] = 1.0
    return type(pose)(data)


def pose_optimization_jacobian(jacobian, fixed_scale):
    return jacobian[..., :6] if fixed_scale else jacobian


def embed_pose_increment(increment, fixed_scale):
    if not fixed_scale:
        return increment
    return torch.cat(
        (increment, torch.zeros_like(increment[..., :1])), dim=-1
    )


def solve_metric_keyframe_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    minimum_points: int,
    minimum_inlier_ratio: float,
    reprojection_error_px: float,
) -> tuple[np.ndarray | None, dict]:
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if object_points.shape[0] != image_points.shape[0]:
        raise ValueError("PnP object/image point counts differ")
    if object_points.shape[0] < minimum_points:
        return None, {
            "accepted": False,
            "reason": "insufficient_metric_correspondences",
            "candidate_points": int(object_points.shape[0]),
        }
    # OpenCV's RANSAC sampler uses process-global RNG state. Resetting it makes
    # repeated offline runs bitwise reproducible instead of occasionally
    # selecting a planar degenerate sample.
    cv2.setRNGSeed(0)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        np.asarray(camera_matrix, dtype=np.float64),
        None,
        iterationsCount=100,
        reprojectionError=float(reprojection_error_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_count = 0 if inliers is None else int(len(inliers))
    inlier_ratio = inlier_count / max(object_points.shape[0], 1)
    if not success or inlier_count < minimum_points or inlier_ratio < minimum_inlier_ratio:
        return None, {
            "accepted": False,
            "reason": "metric_pnp_inliers_low",
            "candidate_points": int(object_points.shape[0]),
            "inlier_points": inlier_count,
            "inlier_ratio": inlier_ratio,
        }
    inlier_indices = inliers[:, 0]
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_indices],
            image_points[inlier_indices],
            np.asarray(camera_matrix, dtype=np.float64),
            None,
            rvec,
            tvec,
        )
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        object_points[inlier_indices],
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        None,
    )
    reprojection_error = np.linalg.norm(
        projected.reshape(-1, 2) - image_points[inlier_indices], axis=1
    )
    keyframe_to_current = np.eye(4, dtype=np.float64)
    keyframe_to_current[:3, :3] = rotation
    keyframe_to_current[:3, 3] = np.asarray(tvec).reshape(3)
    return keyframe_to_current, {
        "accepted": True,
        "candidate_points": int(object_points.shape[0]),
        "inlier_points": inlier_count,
        "inlier_ratio": inlier_ratio,
        "reprojection_median_px": float(np.median(reprojection_error)),
        "reprojection_p95_px": float(np.percentile(reprojection_error, 95)),
    }


def refine_metric_translation_3d3d(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_to_target: np.ndarray,
    minimum_points: int,
    minimum_inlier_ratio: float,
    maximum_residual_m: float,
) -> tuple[np.ndarray | None, dict]:
    """Refine only translation from paired metric 3D points at fixed rotation."""
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(source_to_target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("3D source/target point counts differ")
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < minimum_points:
        return None, {
            "accepted": False,
            "reason": "insufficient_metric_3d_correspondences",
            "candidate_points": int(source.shape[0]),
        }

    rotation = transform[:3, :3]
    translation_samples = target - source @ rotation.T
    translation = np.median(translation_samples, axis=0)
    residual = np.linalg.norm(translation_samples - translation, axis=1)
    residual_median = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_median)))
    robust_limit = max(0.002, residual_median + 3.0 * 1.4826 * residual_mad)
    threshold = min(float(maximum_residual_m), robust_limit)
    inliers = residual <= threshold
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / source.shape[0]
    if inlier_count < minimum_points or inlier_ratio < minimum_inlier_ratio:
        return None, {
            "accepted": False,
            "reason": "metric_3d_translation_inliers_low",
            "candidate_points": int(source.shape[0]),
            "inlier_points": inlier_count,
            "inlier_ratio": inlier_ratio,
            "residual_threshold_m": threshold,
        }

    translation = np.median(translation_samples[inliers], axis=0)
    refined_residual = np.linalg.norm(
        translation_samples[inliers] - translation, axis=1
    )
    refined = transform.copy()
    refined[:3, 3] = translation
    return refined, {
        "accepted": True,
        "candidate_points": int(source.shape[0]),
        "inlier_points": inlier_count,
        "inlier_ratio": inlier_ratio,
        "residual_threshold_m": threshold,
        "residual_p95_m": float(np.percentile(refined_residual, 95)),
    }


def metric_depth_anchor_mask(
    primary_scaled: torch.Tensor,
    confidence: torch.Tensor,
    metric_depth: np.ndarray | None,
    cfg: dict,
) -> torch.Tensor:
    point_count = primary_scaled.reshape(-1, 3).shape[0]
    empty = torch.zeros(point_count, dtype=torch.bool, device=primary_scaled.device)
    if metric_depth is None or not bool(cfg.get("stereo_pointmap_depth_anchor", False)):
        return empty
    metric = np.asarray(metric_depth, dtype=np.float32).reshape(-1)
    confidence_flat = confidence.detach().reshape(-1)
    primary_flat = primary_scaled.reshape(-1, 3)
    if point_count != metric.size or confidence_flat.numel() != metric.size:
        return empty
    metric_tensor = torch.as_tensor(
        metric, device=primary_flat.device, dtype=primary_flat.dtype
    )
    valid = torch.isfinite(metric_tensor)
    valid &= metric_tensor >= float(cfg.get("stereo_scale_min_depth_m", 0.15))
    valid &= metric_tensor <= float(cfg.get("stereo_scale_max_depth_m", 0.65))
    valid &= confidence_flat >= float(cfg.get("stereo_scale_min_confidence", 1.5))
    valid &= primary_flat[:, 2] > 1e-6
    relative_residual = torch.full_like(metric_tensor, float("inf"))
    relative_residual[valid] = torch.abs(
        metric_tensor[valid] / primary_flat[valid, 2] - 1.0
    )
    return valid & (
        relative_residual
        <= float(cfg.get("stereo_depth_anchor_max_relative_residual", 0.5))
    )


def compute_stereo_depth(
    left: np.ndarray,
    right: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
    num_disparities: int = 128,
    consistency_tolerance_px: float = 1.0,
) -> np.ndarray:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("stereo images must be matching grayscale arrays")
    num_disparities = max(16, int(np.ceil(num_disparities / 16.0)) * 16)
    common = dict(
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity_left = cv2.StereoSGBM_create(minDisparity=0, **common).compute(
        left, right
    ).astype(np.float32) / 16.0
    disparity_right = cv2.StereoSGBM_create(
        minDisparity=-num_disparities, **common
    ).compute(right, left).astype(np.float32) / 16.0

    height, width = left.shape
    y, x = np.indices((height, width))
    right_x = np.rint(x - disparity_left).astype(np.int32)
    inside = (right_x >= 0) & (right_x < width) & (disparity_left > 0.5)
    sampled_right = np.full(left.shape, np.nan, dtype=np.float32)
    sampled_right[inside] = disparity_right[y[inside], right_x[inside]]
    valid = inside & (
        np.abs(disparity_left + sampled_right) <= consistency_tolerance_px
    )

    depth = np.full(left.shape, np.nan, dtype=np.float32)
    depth[valid] = focal_length_px * baseline_m / disparity_left[valid]
    valid &= (depth >= minimum_depth_m) & (depth <= maximum_depth_m)
    depth[~valid] = np.nan
    return depth


def robust_pointmap_metric_scale(
    pointmap_z: np.ndarray,
    confidence: np.ndarray,
    metric_depth: np.ndarray,
    minimum_points: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    minimum_confidence: float,
) -> dict:
    pointmap_z = np.asarray(pointmap_z, dtype=np.float64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    metric_depth = np.asarray(metric_depth, dtype=np.float64).reshape(-1)
    if not (pointmap_z.size == confidence.size == metric_depth.size):
        raise ValueError("pointmap, confidence, and metric depth sizes differ")

    valid = np.isfinite(metric_depth) & np.isfinite(pointmap_z)
    valid &= (metric_depth >= minimum_depth_m) & (metric_depth <= maximum_depth_m)
    valid &= pointmap_z > 1e-6
    valid &= confidence >= minimum_confidence
    candidate_points = int(np.count_nonzero(valid))
    if candidate_points < minimum_points:
        return {
            "accepted": False,
            "reason": "insufficient_metric_depth_points",
            "candidate_points": candidate_points,
        }

    ratios = metric_depth[valid] / pointmap_z[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios >= 0.02) & (ratios <= 20.0)]
    if ratios.size < minimum_points:
        return {
            "accepted": False,
            "reason": "insufficient_scale_ratio_points",
            "candidate_points": candidate_points,
            "ratio_points": int(ratios.size),
        }
    center = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - center)))
    threshold = max(3.0 * 1.4826 * mad, 0.03 * center)
    inliers = np.abs(ratios - center) <= threshold
    if int(np.count_nonzero(inliers)) < minimum_points:
        return {
            "accepted": False,
            "reason": "insufficient_robust_scale_inliers",
            "candidate_points": candidate_points,
            "ratio_points": int(ratios.size),
            "inlier_points": int(np.count_nonzero(inliers)),
        }
    scale = float(np.median(ratios[inliers]))
    return {
        "accepted": True,
        "scale": scale,
        "candidate_points": candidate_points,
        "inlier_points": int(np.count_nonzero(inliers)),
        "relative_mad": mad / max(center, 1e-12),
    }


def scale_pointmaps_with_metric_depth(
    pointmaps: tuple[torch.Tensor, ...],
    confidence: torch.Tensor,
    metric_depth: np.ndarray | None,
    cfg: dict,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    if metric_depth is None:
        return pointmaps, {"accepted": False, "reason": "metric_depth_unavailable"}
    primary = pointmaps[0]
    result = robust_pointmap_metric_scale(
        primary[..., 2].detach().cpu().numpy(),
        confidence.detach().cpu().numpy(),
        metric_depth,
        minimum_points=int(cfg.get("stereo_scale_min_points", 500)),
        minimum_depth_m=float(cfg.get("stereo_scale_min_depth_m", 0.15)),
        maximum_depth_m=float(cfg.get("stereo_scale_max_depth_m", 0.65)),
        minimum_confidence=float(cfg.get("stereo_scale_min_confidence", 1.5)),
    )
    if not result["accepted"]:
        return pointmaps, result
    scale = primary.new_tensor(result["scale"])
    scaled = tuple(pointmap * scale for pointmap in pointmaps)
    if not bool(cfg.get("stereo_pointmap_depth_anchor", False)):
        return scaled, result

    primary_scaled = scaled[0].clone().reshape(-1, 3)
    metric_tensor = torch.as_tensor(
        np.asarray(metric_depth, dtype=np.float32).reshape(-1),
        device=primary_scaled.device,
        dtype=primary_scaled.dtype,
    )
    valid = metric_depth_anchor_mask(scaled[0], confidence, metric_depth, cfg)
    maximum_residual = float(cfg.get("stereo_depth_anchor_max_relative_residual", 0.5))
    if valid.any():
        ray_scale = metric_tensor[valid] / primary_scaled[valid, 2]
        primary_scaled[valid] *= ray_scale[:, None]

    anchored = (primary_scaled.reshape_as(scaled[0]), *scaled[1:])
    return anchored, dict(
        result,
        depth_anchor_points=int(valid.sum().item()),
        depth_anchor_max_relative_residual=maximum_residual,
    )


class StereoDepthProvider:
    def __init__(self, dataset_path: Path, metadata: dict):
        self.dataset_path = dataset_path
        self.right_directory = dataset_path / metadata["right_directory"]
        self.focal_length_px = float(metadata["left_focal_length_px"])
        self.baseline_m = float(metadata["baseline_m"])
        self.minimum_depth_m = float(metadata.get("minimum_depth_m", 0.15))
        self.maximum_depth_m = float(metadata.get("maximum_depth_m", 0.65))

    @classmethod
    def from_dataset(cls, dataset_path: Path) -> StereoDepthProvider | None:
        manifest_path = dataset_path / "dataset_manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stereo = manifest.get("stereo_depth_source")
        return cls(dataset_path, stereo) if stereo else None

    def get_depth(self, left_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_path = self.right_directory / left_path.name
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise FileNotFoundError(f"missing stereo pair for {left_path.name}")
        depth = compute_stereo_depth(
            left,
            right,
            self.focal_length_px,
            self.baseline_m,
            self.minimum_depth_m,
            self.maximum_depth_m,
        )
        target_height, target_width = target_shape
        if depth.shape != target_shape:
            depth = cv2.resize(
                depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST
            )
        return depth
