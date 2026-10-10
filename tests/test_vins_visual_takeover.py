from types import SimpleNamespace

import lietorch
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mast3r_slam.tracker import FrameTracker, VisualSolveRejected
from mast3r_slam.vins_takeover import propagate_vins_relative_pose


def pose(position=(0, 0, 0), rotation=None, scale=None):
    q = Rotation.identity().as_quat() if rotation is None else rotation.as_quat()
    return np.r_[position, q] if scale is None else np.r_[position, q, scale]


def test_relative_vins_motion_preserves_visual_world_origin_rotation_and_scale():
    visual = pose((8, 4, -1), Rotation.from_euler("z", 90, degrees=True), .4)
    vins_previous = pose((100, 20, 30), Rotation.from_euler("z", 30, degrees=True))
    step_local = np.array([.01, -.002, .003])
    vins_current = pose(
        vins_previous[:3] + Rotation.from_quat(vins_previous[3:7]).apply(step_local),
        Rotation.from_quat(vins_previous[3:7]) * Rotation.from_euler("y", 2, degrees=True),
    )
    result = propagate_vins_relative_pose(visual, vins_previous, vins_current)
    assert np.allclose(result[:3], visual[:3] + Rotation.from_quat(visual[3:7]).apply(step_local))
    assert result[7] == .4
    expected_rotation = Rotation.from_quat(visual[3:7]) * Rotation.from_euler("y", 2, degrees=True)
    assert (Rotation.from_quat(result[3:7]).inv() * expected_rotation).magnitude() < 1e-10


def test_stationary_vins_outputs_the_same_pose_not_a_missing_frame():
    visual = pose((1, 2, 3), scale=.7)
    vins = pose((100, 200, 300))
    assert np.allclose(propagate_vins_relative_pose(visual, vins, vins), visual)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_vins_position_is_rejected(bad):
    vins = pose((bad, 0, 0))
    with pytest.raises(ValueError):
        propagate_vins_relative_pose(pose(scale=1), pose(), vins)


def test_zero_scale_or_quaternion_is_rejected():
    with pytest.raises(ValueError):
        propagate_vins_relative_pose(pose(scale=0), pose(), pose())
    bad = pose()
    bad[3:7] = 0
    with pytest.raises(ValueError):
        propagate_vins_relative_pose(pose(scale=1), pose(), bad)


def row(index, x=0, valid="1"):
    return dict(input_index=str(index), valid=valid, x=str(x), y="0", z="0",
                qx="0", qy="0", qz="0", qw="1")


def tracker_and_frames():
    tracker = FrameTracker.__new__(FrameTracker)
    tracker.cfg = {"vins_visual_takeover": True, "vins_translation_prior_sigma_m": .004}
    tracker.vins_camera_poses = [row(0), row(1, .002)]
    tracker.vins_degraded = False
    tracker.vins_recovery_count = 0
    tracker.vins_recovery_frame_id = None
    tracker.vins_degraded_frames = 0
    tracker.vins_translation_prior_sigma_m = .004
    tracker.idx_f2k = torch.ones(1)
    tracker.last_takeover_report = None
    tracker.previous_frame = SimpleNamespace(frame_id=0, T_WC=lietorch.Sim3(torch.tensor(pose(scale=.5)[None], dtype=torch.float32)))
    frame = SimpleNamespace(frame_id=1, T_WC=tracker.previous_frame.T_WC)
    return tracker, frame


def test_pose_only_takeover_keeps_scale_and_resets_visual_match_cache():
    tracker, frame = tracker_and_frames()
    assert tracker.recover_vins(frame, "test_failed_solver")
    assert torch.allclose(frame.T_WC.data[0, :3], torch.tensor([.002, 0, 0]))
    assert frame.T_WC.data[0, 7] == .5
    assert frame.pose_only
    assert tracker.idx_f2k is None
    assert tracker.vins_degraded
    assert tracker.last_takeover_report["reason"] == "test_failed_solver"


@pytest.mark.parametrize("failure", ["invalid", "gap", "jump"])
def test_unusable_vins_is_not_reported_as_successful_takeover(failure):
    tracker, frame = tracker_and_frames()
    original = frame.T_WC.data.clone()
    if failure == "invalid":
        tracker.vins_camera_poses[1]["valid"] = "0"
    elif failure == "gap":
        tracker.previous_frame.frame_id = -1
    else:
        tracker.vins_camera_poses[1]["x"] = "1"
    assert not tracker.recover_vins(frame, "visual_failure")
    assert torch.equal(frame.T_WC.data, original)
    assert not getattr(frame, "pose_only", False)


def test_good_visual_pose_does_not_exit_degraded_mode_until_two_checks():
    tracker, frame = tracker_and_frames()
    tracker.vins_degraded = True
    candidate = tracker.predict_vins_pose(frame)
    assert candidate is not None
    assert not tracker.accept_visual_recovery(candidate, frame)
    assert tracker.vins_degraded
    assert not tracker.accept_visual_recovery(candidate, frame)
    assert tracker.vins_recovery_count == 1
    tracker.previous_frame = frame
    tracker.vins_camera_poses.append(row(2, .004))
    frame = SimpleNamespace(frame_id=2, T_WC=candidate)
    candidate = tracker.predict_vins_pose(frame)
    assert tracker.accept_visual_recovery(candidate, frame)
    assert not tracker.vins_degraded


def test_disagreeing_visual_pose_resets_recovery_confirmation():
    tracker, frame = tracker_and_frames()
    tracker.vins_degraded = True
    good = tracker.predict_vins_pose(frame)
    assert not tracker.accept_visual_recovery(good, frame)
    bad_data = good.data.clone()
    bad_data[0, 0] += .1
    assert not tracker.accept_visual_recovery(lietorch.Sim3(bad_data), frame)
    assert tracker.vins_recovery_count == 0


def test_missing_startup_vins_does_not_prohibit_healthy_visual_initialization():
    tracker, frame = tracker_and_frames()
    tracker.vins_camera_poses[0]["valid"] = "0"
    assert tracker.accept_visual_recovery(frame.T_WC, frame)
    tracker.vins_degraded = True
    assert not tracker.accept_visual_recovery(frame.T_WC, frame)


def test_disabled_takeover_does_not_modify_pose():
    tracker, frame = tracker_and_frames()
    tracker.cfg["vins_visual_takeover"] = False
    original = frame.T_WC.data.clone()
    assert not tracker.recover_vins(frame, "unreliable_geometry")
    assert torch.equal(frame.T_WC.data, original)


def test_recovery_reference_is_independent_of_shared_graph_and_metric_scale(monkeypatch):
    import mast3r_slam.tracker as module
    tracker, frame = tracker_and_frames()
    tracker.metric_pointmap_scale = 4.0
    tracker.recovery_reference = None
    tracker.recovery_reference_scale = None
    frame.metric_depth = np.ones((1, 3))
    frame.K = torch.eye(3)
    raw = torch.ones((3, 3))
    conf = torch.full((3, 1), 2.0)
    monkeypatch.setattr(module, "scale_pointmaps_with_metric_depth",
                        lambda *args: ((raw * 2,), {"accepted": True, "scale": 2.0}))
    monkeypatch.setattr(module, "metric_depth_anchor_mask",
                        lambda *args: torch.ones(3, dtype=torch.bool))
    assert tracker.seed_recovery_reference(frame, raw, conf, frame.K)
    anchor = tracker.recovery_reference
    assert anchor is not frame
    assert anchor.frame_id == frame.frame_id
    assert anchor.K is not frame.K
    assert tracker.metric_pointmap_scale == 4.0
    assert tracker.recovery_reference_scale == 2.0
    frame.T_WC.data[0, 0] = 99
    conf.fill_(0)
    assert anchor.T_WC.data[0, 0] != 99
    assert torch.all(anchor.C == 2)
    assert torch.all(anchor.X_canon == 2)


def test_bad_self_geometry_cannot_replace_the_recovery_reference(monkeypatch):
    import mast3r_slam.tracker as module
    tracker, frame = tracker_and_frames()
    original = object()
    tracker.recovery_reference = original
    tracker.recovery_reference_scale = 2.0
    frame.metric_depth = None
    monkeypatch.setattr(module, "scale_pointmaps_with_metric_depth",
                        lambda *args: (args[0], {"accepted": False}))
    assert not tracker.seed_recovery_reference(frame, torch.ones((3, 3)), torch.ones((3, 1)), None)
    assert tracker.recovery_reference is original
    assert tracker.recovery_reference_scale == 2.0


def test_takeover_tracks_fresh_reference_instead_of_stale_graph_keyframe(monkeypatch):
    import mast3r_slam.tracker as module
    tracker, frame = tracker_and_frames()
    tracker.spatial_recovery_rotations = None
    tracker.metric_pointmap_scale = 4.0
    tracker.model = None
    shadow = SimpleNamespace(K=torch.eye(3))
    tracker.recovery_reference = shadow
    tracker.recovery_reference_scale = 2.0
    tracker.keyframes = SimpleNamespace(last_keyframe=lambda: object())
    def match(model, current, reference, **kwargs):
        assert reference is shadow
        raise RuntimeError("selected_fresh_reference")
    monkeypatch.setattr(module, "mast3r_match_asymmetric", match)
    with pytest.raises(RuntimeError, match="selected_fresh_reference"):
        tracker.track(frame)
    assert tracker.metric_pointmap_scale == 4.0


def test_confirmed_shadow_visual_pose_never_writes_or_inserts_graph_reference(monkeypatch):
    import mast3r_slam.tracker as module
    tracker, frame = tracker_and_frames()
    tracker.cfg.update(C_conf=0., Q_conf=1.5, min_match_frac=.05,
                       stereo_scale_min_points=2, C_weight_floor=1.)
    tracker.vins_degraded = True
    tracker.spatial_recovery_rotations = None
    tracker.stereo_imu_anchor = None
    tracker.metric_pointmap_scale = 4.0
    tracker.recovery_reference_scale = 2.0
    tracker.model = None
    points = torch.tensor([[0., 0., .3], [.01, 0., .3], [0., .01, .3]])
    confidence = torch.full((3, 1), 2.)
    shadow = SimpleNamespace(K=torch.eye(3), X_canon=points, C=confidence,
                             frame_id=0, T_WC=tracker.previous_frame.T_WC)
    tracker.recovery_reference = shadow
    tracker.keyframes = object()  # Any graph read/write/append would fail.
    frame.metric_depth = np.full((1, 3), .3)
    frame.img = torch.ones((3, 1, 3))
    frame.update_pointmap = lambda *args: None
    tracker.last_metric_anchor_mask = torch.ones(3, dtype=torch.bool)
    monkeypatch.setitem(module.config, "use_calib", False)
    monkeypatch.setattr(module, "mast3r_match_asymmetric", lambda *args, **kwargs: (
        torch.arange(3)[None], torch.ones((1, 3, 1), dtype=torch.bool),
        points, confidence, confidence, points, confidence, confidence,
    ))
    tracker.scale_pointmaps = lambda *args: ((points, points), {"accepted": True})
    monkeypatch.setattr(module, "scale_pointmaps_with_metric_depth",
                        lambda *args: ((points,), {"accepted": True, "scale": 2.}))
    monkeypatch.setattr(module, "metric_depth_anchor_mask", lambda *args: torch.ones(3, dtype=torch.bool))
    tracker.get_points_poses = lambda *args: (
        points, points, frame.T_WC, shadow.T_WC, confidence, confidence, None, None)
    tracker.opt_pose_ray_dist_sim3 = lambda *args: (tracker.predict_vins_pose(frame), lietorch.Sim3.Identity(1))
    assert tracker.track(frame) == (False, [], False)
    assert frame.pose_only
    tracker.previous_frame = frame
    tracker.vins_camera_poses.append(row(2, .004))
    frame.frame_id = 2
    # Do not alias the previous-frame index while simulating SharedStates.
    tracker.previous_frame = SimpleNamespace(frame_id=1, T_WC=frame.T_WC)
    assert tracker.track(frame) == (False, [], False)
    assert not frame.pose_only
    assert not tracker.vins_degraded
    assert tracker.recovery_reference.frame_id == 2
    assert tracker.metric_pointmap_scale == 4.0


@pytest.mark.parametrize("geometry_accepted,reference_conf", [(False, 2.0), (True, 1.0)])
def test_unreliable_geometry_returns_before_any_reference_update(monkeypatch, geometry_accepted, reference_conf):
    import mast3r_slam.tracker as module
    tracker, frame = tracker_and_frames()
    tracker.cfg.update(stereo_pointmap_scale_prior=True, stereo_scale_min_points=2)
    tracker.spatial_recovery_rotations = None
    tracker.metric_pointmap_scale = 2.0
    tracker.last_metric_anchor_mask = None
    tracker.model = None
    keyframe = SimpleNamespace(K=None)
    tracker.keyframes = SimpleNamespace(last_keyframe=lambda: keyframe)
    frame.metric_depth = None
    frame.update_pointmap = lambda *args: None
    points = torch.zeros((3, 3))
    confidence = torch.full((3, 1), reference_conf)
    monkeypatch.setattr(module, "mast3r_match_asymmetric", lambda *args, **kwargs: (
        torch.arange(3)[None], torch.ones((1, 3, 1), dtype=torch.bool),
        points, confidence, confidence, points, confidence, confidence,
    ))

    def scale(*args):
        tracker.metric_pointmap_scale = 9.0
        return (points, points), {"accepted": geometry_accepted}

    tracker.scale_pointmaps = scale
    # The stub has no update_pointmap/indexing; proceeding into reference writes
    # would fail instead of silently satisfying this assertion.
    assert tracker.track(frame) == (False, [], False)
    assert frame.pose_only
    assert tracker.metric_pointmap_scale == 2.0


@pytest.mark.parametrize("costs,delta_norm,expected", [
    ([1., 2.], 1e-8, "uphill_visual_solve"),
    ([1., .8], 1e-8, "visual_solve_max_iterations"),
    ([float("nan")], 1e-8, "nonfinite_visual_solve"),
])
def test_bad_visual_solve_is_rejected(monkeypatch, costs, delta_norm, expected):
    tracker, _ = tracker_and_frames()
    tracker.cfg.update(max_iters=len(costs), sigma_pixel=1., sigma_depth=1.,
                       pixel_border=0, depth_eps=1e-6, rel_error=0., delta_norm=delta_norm)
    values = iter(costs)
    tracker.solve_pose_increment = lambda *args, **kwargs: (
        torch.tensor([[.01, 0., 0., 0., 0., 0., 0.]]), next(values)
    )
    _call_visual_solver(tracker, expected)


def _call_visual_solver(tracker, expected=None):
    points = torch.tensor([[0., 0., 1.], [.1, .1, 1.]])
    one = torch.ones((2, 1))
    valid = one.bool()
    camera = torch.tensor([[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]])
    kwargs = dict(Xf=points, Xk=points, T_WCf=lietorch.Sim3.Identity(1),
                  T_WCk=lietorch.Sim3.Identity(1), Qk=one, valid=valid,
                  conf_w=one, meas_k=torch.zeros((2, 3)), valid_meas_k=valid,
                  K=camera, img_size=(10, 10), metric_translation_target=torch.zeros(3))
    if expected:
        with pytest.raises(RuntimeError, match=expected):
            tracker.opt_pose_calib_sim3(**kwargs)
    else:
        return tracker.opt_pose_calib_sim3(**kwargs)


def test_terminal_increment_is_not_published_without_evaluating_its_cost():
    tracker, _ = tracker_and_frames()
    tracker.cfg.update(max_iters=2, sigma_pixel=1., sigma_depth=1.,
                       pixel_border=0, depth_eps=1e-6, rel_error=.01, delta_norm=1e-8)
    values = iter([1., .999])
    tracker.solve_pose_increment = lambda *args, **kwargs: (
        torch.tensor([[.01, 0., 0., 0., 0., 0., 0.]]), next(values)
    )
    world, _ = _call_visual_solver(tracker)
    assert float(world.data[0, 0]) == pytest.approx(.01)


def test_unexpected_visual_exception_is_not_hidden_by_vins_takeover():
    tracker, frame = tracker_and_frames()
    with pytest.raises(ValueError, match="wrong_tensor_shape"):
        tracker.recover_visual_solve_failure(frame, ValueError("wrong_tensor_shape"))
    assert not getattr(frame, "pose_only", False)
    assert tracker.recover_visual_solve_failure(frame, VisualSolveRejected("uphill_visual_solve"))


def test_vins_rotation_is_not_rejected_when_no_independent_gyro_prior_is_enabled():
    tracker, frame = tracker_and_frames()
    rotation = Rotation.from_euler("y", 2, degrees=True).as_quat()
    for key, value in zip(("qx", "qy", "qz", "qw"), rotation):
        tracker.vins_camera_poses[1][key] = str(value)
    assert tracker.predict_vins_pose(frame) is not None
    tracker.cfg["imu_rotation_prior"] = True
    assert tracker.predict_vins_pose(frame) is None
