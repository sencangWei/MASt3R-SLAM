import torch
import lietorch

from mast3r_slam.tracker import (
    FrameTracker,
    anchored_camera_local_translation_target,
    metric_translation_prior_terms,
)


def test_metric_translation_prior_jacobian_matches_left_retraction():
    pose = lietorch.Sim3.exp(
        torch.tensor([[0.012, -0.008, 0.018, 0.08, -0.04, 0.03, 0.05]])
    )
    target = torch.tensor([0.010, -0.005, 0.015])
    residual, jacobian, info = metric_translation_prior_terms(
        pose, target, sigma_m=0.004
    )
    assert residual.shape == (1, 3)
    assert jacobian.shape == (1, 3, 7)
    assert torch.allclose(info, torch.full((1, 3), 250.0))

    eps = 1e-4
    for axis in range(7):
        increment = torch.zeros((1, 7))
        increment[0, axis] = eps
        plus = pose.retr(increment)
        minus = pose.retr(-increment)
        numerical = -(plus.data[0, :3] - minus.data[0, :3]) / (2 * eps)
        assert torch.allclose(jacobian[0, :, axis], numerical, atol=2e-4)


def test_metric_translation_prior_rejects_nonpositive_sigma():
    pose = lietorch.Sim3.Identity(1)
    try:
        metric_translation_prior_terms(pose, torch.zeros(3), sigma_m=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero sigma must be rejected")


def test_metric_translation_prior_uses_keyframe_world_scale():
    pose = lietorch.Sim3.exp(
        torch.tensor([[0.015, -0.006, 0.011, 0.04, 0.02, -0.03, -0.2]])
    )
    target = torch.tensor([0.020, 0.001, -0.005])
    world_scale = 0.35
    residual, jacobian, _ = metric_translation_prior_terms(
        pose, target, sigma_m=0.004, world_scale=world_scale
    )
    assert torch.allclose(
        residual[0], target - world_scale * pose.data[0, :3]
    )
    eps = 1e-4
    for axis in range(7):
        increment = torch.zeros((1, 7))
        increment[0, axis] = eps
        plus = pose.retr(increment)
        minus = pose.retr(-increment)
        numerical = -world_scale * (plus.data[0, :3] - minus.data[0, :3]) / (2 * eps)
        assert torch.allclose(jacobian[0, :, axis], numerical, atol=2e-4)


def test_correlated_visual_rows_are_normalized_after_robust_weighting():
    tracker = FrameTracker.__new__(FrameTracker)
    tracker.cfg = {"huber": 1000.0}
    jacobian = torch.zeros((11, 3, 7))
    residual = torch.zeros((11, 3))
    for row in range(4):
        jacobian[row, 0, 0] = -1.0
        residual[row, 0] = 1.0
    for axis in range(1, 7):
        jacobian[axis + 3, 0, axis] = -1.0
    jacobian[-1, 0, 0] = -1.0
    sqrt_info = torch.ones_like(residual)

    unnormalized, _ = tracker.solve(sqrt_info, residual, jacobian)
    normalized, _ = tracker.solve(
        sqrt_info, residual, jacobian,
        visual_row_count=10, visual_effective_count=10,
    )
    assert torch.allclose(unnormalized[0, 0], torch.tensor(0.8), atol=1e-5)
    assert torch.allclose(normalized[0, 0], torch.tensor(4.0 / 14.0), atol=1e-5)


def test_vins_anchor_bridges_a_keyframe_before_vins_initialization():
    keyframe = torch.tensor([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4])
    visual_anchor = torch.tensor(
        [0.3, 0.1, 0.0, 0.0, 0.0, 0.70710678, 0.70710678, 1.0]
    )
    vins_anchor = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    vins_current = torch.tensor([2.1, 0.0, 0.0])
    target = anchored_camera_local_translation_target(
        keyframe, visual_anchor, vins_anchor, vins_current
    )
    # The VINS +X motion becomes visual-world +Y, then subtract the
    # pre-initialization visual keyframe's world position.
    assert torch.allclose(
        torch.as_tensor(target), torch.tensor([0.2, 0.4, 0.0], dtype=torch.float64), atol=1e-6
    )
