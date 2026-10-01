import numpy as np

from mast3r_slam.stereo_pointmap_recovery import (
    condition_pair, spatial_depth_correction,
)


def test_spatial_correction_generalizes_to_heldout_image_cells():
    model = np.zeros((128, 128, 3), dtype=np.float32)
    metric = model.copy()
    metric[..., 2] = 0.02
    corrected, report = spatial_depth_correction(
        model, metric, np.ones((128, 128), dtype=bool)
    )
    assert report["accepted"]
    assert report["holdout_points"] == 8192
    assert abs(report["before_p50_mm"] - 20.0) < 1e-4
    assert report["after_p50_mm"] < 0.01
    np.testing.assert_allclose(corrected[..., 2], 0.02, atol=1e-6)


def test_pair_conditioning_keeps_model_units_and_requires_both_views():
    height = width = 128
    depth = np.full((height, width), 0.4, dtype=np.float32)
    K = np.array([[500.0, 0.0, 64.0], [0.0, 500.0, 64.0], [0.0, 0.0, 1.0]])
    y, x = np.indices(depth.shape)
    metric = np.stack(((x - 64) * depth / 500, (y - 64) * depth / 500, depth), -1)
    pointmaps = np.stack(((metric + [0.0, 0.0, 0.02]) / 0.1,) * 2).astype(np.float32)
    confidence = np.ones((2, height, width), dtype=np.float32)
    pixels = np.column_stack((x.ravel()[::8], y.ravel()[::8]))
    conditioned, report = condition_pair(
        pointmaps, confidence, (depth, depth), np.eye(4), 0.1, K,
        pixels, pixels,
    )
    assert report["accepted"]
    np.testing.assert_allclose(conditioned[..., 2], 4.0, atol=1e-5)

    shifted_depth = depth + 0.001
    _, metric_report = condition_pair(
        pointmaps, confidence, (depth, shifted_depth), np.eye(4), 0.1, K,
        pixels, pixels,
    )
    assert metric_report["accepted"]
    assert 0.9 < metric_report["pair_p50_mm"] < 1.1

    missing_depth = np.full_like(depth, np.nan)
    rejected, report = condition_pair(
        pointmaps, confidence, (depth, missing_depth), np.eye(4), 0.1, K,
        pixels, pixels,
    )
    assert rejected is None
    assert report["reason"] == "heldout_geometry_failed"
