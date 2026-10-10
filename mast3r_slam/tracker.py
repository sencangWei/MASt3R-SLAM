import os
import atexit
import csv
import json
import math
import copy

import torch
import lietorch
import numpy as np
from scipy.spatial.transform import Rotation
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import (
    mast3r_asymmetric_inference,
    mast3r_match_asymmetric,
    mast3r_match_symmetric,
)
from mast3r_slam.stereo_depth import (
    embed_pose_increment,
    force_unit_sim3_scale,
    metric_depth_anchor_mask,
    pose_optimization_jacobian,
    refine_metric_translation_3d3d,
    robust_pointmap_metric_scale,
    scale_pointmaps_with_metric_depth,
    solve_metric_keyframe_pnp,
)
from mast3r_slam.stereo_imu_anchor import integrate_imu_rotations, propose_anchor_pose
from mast3r_slam.stereo_descriptor_recovery import (
    descriptor_relative_translation,
    imu_rotation_agrees,
    metric_descriptor_pose,
)
from mast3r_slam.vins_takeover import propagate_vins_relative_pose


_MATCH_LOG = os.environ.get("MAST3R_MATCH_LOG", "")


class VisualSolveRejected(RuntimeError):
    """Expected numerical rejection, not an arbitrary visual-path defect."""


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


_ROBUST_LOG = os.environ.get("MAST3R_ROBUST_LOG", "")
_FRONTEND_LOG = os.environ.get("MAST3R_FRONTEND_LOG", "")
_robust_samples = {}  # (width, part) -> list[Tensor]


def _column_groups(w):
    """把 r 的列映射到语义分组（列序见各 optimizer 里 sqrt_info 的拼接顺序）。

    `opt_pose_ray_dist_sim3` 是 [光线(3), 距离(1)]，`opt_pose_calib_sim3` 是
    [u, v, log_z]。两组的白化尺度差几个数量级，混在一起统计会把结论带偏
    （第一版就是因为混在一起，才发现不了 pixel/depth 的权重关系 → §28.3）。
    """
    if w == 4:
        return {"ray": slice(0, 3), "dist": slice(3, 4)}
    if w == 3:
        return {"pixel": slice(0, 2), "depth": slice(2, 3)}
    return {"all": slice(0, w)}


def log_robust_sample(whitened_r):
    """记录白化残差 |whitened_r| 的分布（仅在 MAST3R_ROBUST_LOG 设了路径时生效）。"""
    if not _ROBUST_LOG:
        return
    a = whitened_r.detach().abs()
    for part, sl in _column_groups(a.shape[-1]).items():
        v = a[..., sl].reshape(-1)
        if v.numel() > 5000:  # 均匀抽样，避免长跑把内存吃光（每帧每组上限 5000）
            idx = torch.randint(0, v.numel(), (5000,), device=v.device)
            v = v[idx]
        _robust_samples.setdefault((a.shape[-1], part), []).append(v.cpu())


def _dump_robust_log():
    if not _ROBUST_LOG or not _robust_samples:
        return
    import numpy as np

    with open(_ROBUST_LOG, "w") as f:
        f.write("part,width,n,p50,p90,p99,max,frac<1.345,frac<10\n")
        for (w, part), chunks in sorted(_robust_samples.items(), key=str):
            v = torch.cat(chunks).numpy().ravel()
            f.write("%s,%d,%d,%.4g,%.4g,%.4g,%.4g,%.5f,%.5f\n" % (
                part, w, v.size, *np.percentile(v, [50, 90, 99]).tolist(), v.max(),
                float((v < 1.345).mean()), float((v < 10.0).mean())))


atexit.register(_dump_robust_log)


def log_match_stats(frame_id, match_k, valid_kf, valid_opt):
    """逐帧记录匹配数（仅在 MAST3R_MATCH_LOG 设了路径时生效，默认零行为影响）。"""
    if not _MATCH_LOG:
        return
    new = not os.path.exists(_MATCH_LOG)
    with open(_MATCH_LOG, "a") as f:
        if new:
            f.write("frame_id,n_match,n_match_Q,n_opt,n_total\n")
        f.write(f"{frame_id},{int(match_k.sum())},{int(valid_kf.sum())},"
                f"{int(valid_opt.sum())},{int(valid_opt.numel())}\n")


def rotation_disagreement_deg(T_visual, T_prior):
    """Return the shortest angular distance between two Sim3 rotations."""
    q_visual = torch.nn.functional.normalize(T_visual.data[..., 3:7], dim=-1)
    q_prior = torch.nn.functional.normalize(T_prior.data[..., 3:7], dim=-1)
    dot = torch.sum(q_visual * q_prior, dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def blend_sim3_rotation(T_visual, T_prior, weight):
    """Slerp only the rotation from visual pose toward the IMU prior pose."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("rotation prior weight must be within [0, 1]")
    if weight == 0.0:
        return T_visual

    visual_data = T_visual.data.clone()
    q_visual = torch.nn.functional.normalize(visual_data[..., 3:7], dim=-1)
    q_prior = torch.nn.functional.normalize(T_prior.data[..., 3:7], dim=-1)
    dot = torch.sum(q_visual * q_prior, dim=-1, keepdim=True)
    q_prior = torch.where(dot < 0.0, -q_prior, q_prior)
    dot = torch.sum(q_visual * q_prior, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = torch.nn.functional.normalize(
        (1.0 - weight) * q_visual + weight * q_prior, dim=-1
    )
    spherical = (
        torch.sin((1.0 - weight) * theta) / sin_theta * q_visual
        + torch.sin(weight * theta) / sin_theta * q_prior
    )
    quaternion = torch.where(sin_theta.abs() < 1e-6, linear, spherical)
    visual_data[..., 3:7] = torch.nn.functional.normalize(quaternion, dim=-1)
    return lietorch.Sim3(visual_data)


def motion_keyframe_trigger(T_CkCf, T_prior, T_keyframe, cfg, frame_gap=0):
    """Request a keyframe only after independently bounded motion."""
    translation_limit = float(cfg.get("motion_keyframe_translation", 0.0))
    rotation_limit_deg = float(cfg.get("motion_keyframe_rotation_deg", 0.0))
    maximum_frame_gap = int(cfg.get("motion_keyframe_max_gap_frames", 0))
    aged_translation_limit = float(
        cfg.get("motion_keyframe_aged_translation", 0.0)
    )
    translation = float(torch.linalg.vector_norm(T_CkCf.data[..., :3]).item())
    rotation_deg = float(rotation_disagreement_deg(T_prior, T_keyframe).item())
    aged_motion = (
        maximum_frame_gap > 0
        and frame_gap >= maximum_frame_gap
        and aged_translation_limit > 0.0
        and translation >= aged_translation_limit
    )
    triggered = (
        translation_limit > 0.0 and translation >= translation_limit
    ) or (
        rotation_limit_deg > 0.0 and rotation_deg >= rotation_limit_deg
    ) or aged_motion
    return triggered, translation, rotation_deg


def metric_translation_prior_terms(pose, target_m, sigma_m, world_scale=1.0):
    """Linearize a metric camera-i translation target for left Sim(3) updates."""
    if sigma_m <= 0.0:
        raise ValueError("metric translation sigma must be positive")
    if not np.isfinite(world_scale) or world_scale <= 0.0:
        raise ValueError("keyframe world scale must be finite and positive")
    translation = pose.data[0, :3]
    target = torch.as_tensor(target_m, device=translation.device, dtype=translation.dtype)
    residual = (target - world_scale * translation).reshape(1, 3)
    jacobian = translation.new_zeros((1, 3, 7))
    jacobian[0, :, :3] = -torch.eye(3, device=translation.device, dtype=translation.dtype)
    x, y, z = translation.unbind()
    jacobian[0, :, 3:6] = torch.stack(
        (torch.stack((translation.new_zeros(()), -z, y)),
         torch.stack((z, translation.new_zeros(()), -x)),
         torch.stack((-y, x, translation.new_zeros(())))),
    )
    jacobian[0, :, 6] = -translation
    jacobian *= world_scale
    sqrt_info = torch.full_like(residual, 1.0 / sigma_m)
    return residual, jacobian, sqrt_info


def anchored_camera_local_translation_target(
    keyframe_visual, anchor_visual, anchor_vins, current_vins_position
):
    """Bridge a visual keyframe that predates VINS without VINS extrapolation."""
    keyframe_visual = np.asarray(keyframe_visual, dtype=float)
    anchor_visual = np.asarray(anchor_visual, dtype=float)
    anchor_vins = np.asarray(anchor_vins, dtype=float)
    current_vins_position = np.asarray(current_vins_position, dtype=float)
    world_alignment = (
        Rotation.from_quat(anchor_visual[3:7])
        * Rotation.from_quat(anchor_vins[3:7]).inv()
    )
    target_world = anchor_visual[:3] + world_alignment.apply(
        current_vins_position - anchor_vins[:3]
    )
    return Rotation.from_quat(keyframe_visual[3:7]).inv().apply(
        target_world - keyframe_visual[:3]
    )


class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device
        self.metric_pointmap_scale = None
        self.last_metric_anchor_mask = None
        self.vins_camera_poses = None
        self.vins_visual_anchor = None
        self.vins_pose_anchor = None
        self.previous_frame = None
        self.vins_degraded = False
        self.vins_recovery_count = 0
        self.vins_recovery_frame_id = None
        self.vins_degraded_frames = 0
        self.last_takeover_report = None
        self.recovery_reference = None
        self.recovery_reference_scale = None
        self.vins_window_visual_anchor = None
        self.vins_window_pose_anchor = None
        self.vins_window_frame_id = None
        self.last_reentry_probe_report = None
        self.last_visual_consistency_report = None
        self.in_reentry_probe = False
        self.stereo_imu_anchor = None
        self.spatial_recovery_rotations = None
        self.vins_translation_prior_sigma_m = float(
            self.cfg.get("vins_translation_prior_sigma_m", 0.0)
        )
        if self.vins_translation_prior_sigma_m > 0.0:
            if not self.cfg.get("stereo_pointmap_scale_prior", False):
                raise ValueError("VINS metric translation prior requires metric pointmaps")
            path = os.environ.get("MAST3R_VINS_CAMERA_POSES")
            if not path:
                raise ValueError("MAST3R_VINS_CAMERA_POSES is required")
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows or [int(row["input_index"]) for row in rows] != list(range(len(rows))):
                raise ValueError("VINS camera pose prior indices must be contiguous")
            self.vins_camera_poses = rows

        if self.cfg.get("vins_visual_takeover", False) and self.vins_camera_poses is None:
            raise ValueError("VINS visual takeover requires metric camera pose priors")

        self.reset_idx_f2k()

    def reset_vins_recovery_window(self):
        self.vins_window_visual_anchor = None
        self.vins_window_pose_anchor = None
        self.vins_window_frame_id = None

    def _has_vins_recovery_window(self):
        return (
            self.vins_window_visual_anchor is not None
            and self.vins_window_pose_anchor is not None
            and self.vins_window_frame_id is not None
        )

    def predict_vins_pose(self, frame):
        previous = self.previous_frame
        rows = self.vins_camera_poses
        if previous is None or rows is None or previous.frame_id != frame.frame_id - 1:
            return None
        if not 0 <= previous.frame_id < frame.frame_id < len(rows):
            return None
        before, after = rows[previous.frame_id], rows[frame.frame_id]
        if before["valid"] != "1" or after["valid"] != "1":
            return None
        try:
            before_pose, after_pose = self.vins_pose_from_row(before), self.vins_pose_from_row(after)
            # Same short-frame physical sanity bound as stereo descriptor recovery.
            if np.linalg.norm(after_pose[:3] - before_pose[:3]) > 0.03:
                return None
            data = propagate_vins_relative_pose(
                previous.T_WC.data[0].detach().cpu().numpy(), before_pose, after_pose
            )
            predicted = lietorch.Sim3(torch.as_tensor(
                data, dtype=frame.T_WC.data.dtype, device=frame.T_WC.data.device
            ).reshape(1, 8))
            # The incoming frame is initialized with the independent gyro prior.
            if (self.cfg.get("imu_rotation_prior", False)
                    and float(rotation_disagreement_deg(predicted, frame.T_WC).item()) > 1.5):
                return None
            return predicted
        except (ValueError, KeyError, TypeError):
            return None

    def _predict_vins_pose_from_window_anchor(self, frame):
        rows = self.vins_camera_poses
        if (
            rows is None
            or self.vins_window_visual_anchor is None
            or self.vins_window_pose_anchor is None
            or self.vins_window_frame_id is None
            or not 0 <= self.vins_window_frame_id < frame.frame_id < len(rows)
        ):
            return None
        current = rows[frame.frame_id]
        if current["valid"] != "1":
            return None
        try:
            current_pose = self.vins_pose_from_row(current)
            data = propagate_vins_relative_pose(
                self.vins_window_visual_anchor, self.vins_window_pose_anchor, current_pose
            )
            return lietorch.Sim3(torch.as_tensor(
                data, dtype=frame.T_WC.data.dtype, device=frame.T_WC.data.device
            ).reshape(1, 8))
        except (ValueError, KeyError, TypeError):
            return None

    def visual_vins_consistency_report(self, visual_pose, frame):
        predicted = self.predict_vins_pose(frame)
        report = {
            "accepted": False,
            "reason": "vins_unavailable",
            "frame_id": int(frame.frame_id),
            "window_anchor_frame_id": self.vins_window_frame_id,
        }
        if predicted is None:
            return False, report
        translation_error = float(torch.linalg.vector_norm(
            visual_pose.data[0, :3] - predicted.data[0, :3]
        ).item())
        scale_change = abs(float(visual_pose.data[0, 7] / predicted.data[0, 7]) - 1.0)
        report.update(
            adjacent_translation_error_m=translation_error,
            adjacent_rotation_error_deg=float(
                rotation_disagreement_deg(visual_pose, predicted).item()
            ),
            adjacent_scale_change=scale_change,
        )
        adjacent_limit = 3 * float(self.cfg["vins_translation_prior_sigma_m"])
        if not bool(torch.isfinite(visual_pose.data).all()):
            report["reason"] = "nonfinite_visual_pose"
            return False, report
        if translation_error > adjacent_limit:
            report["reason"] = "adjacent_vins_translation_disagreement"
            return False, report
        if report["adjacent_rotation_error_deg"] > 1.5:
            report["reason"] = "adjacent_vins_rotation_disagreement"
            return False, report
        if scale_change > float(self.cfg.get("stereo_scale_max_relative_jump", 0.25)):
            report["reason"] = "visual_scale_jump"
            return False, report

        window_predicted = (
            self._predict_vins_pose_from_window_anchor(frame)
            if self.cfg.get("vins_visual_safe_reentry", False) else None
        )
        if window_predicted is not None:
            window_error = float(torch.linalg.vector_norm(
                visual_pose.data[0, :3] - window_predicted.data[0, :3]
            ).item())
            steps = max(int(frame.frame_id - self.vins_window_frame_id), 1)
            sigma = float(self.cfg["vins_translation_prior_sigma_m"])
            max_sigma = float(self.cfg.get("vins_reentry_window_max_sigma", 3.0))
            window_limit = sigma * max_sigma * math.sqrt(steps)
            cap = float(self.cfg.get("vins_reentry_window_max_error_m", 0.012))
            if cap > 0.0:
                window_limit = min(window_limit, cap)
            report.update(
                window_translation_error_m=window_error,
                window_translation_limit_m=window_limit,
            )
            if window_error > window_limit:
                report["reason"] = "window_vins_translation_disagreement"
                return False, report
        report["accepted"] = True
        report["reason"] = "visual_vins_consistent"
        return True, report

    def anchor_vins_recovery_window(self, visual_pose, frame):
        rows = self.vins_camera_poses
        if rows is None or not 0 <= frame.frame_id < len(rows):
            return
        row = rows[frame.frame_id]
        if row["valid"] != "1":
            return
        self.vins_window_visual_anchor = (
            visual_pose.data[0].detach().cpu().numpy().copy()
        )
        self.vins_window_pose_anchor = self.vins_pose_from_row(row)
        self.vins_window_frame_id = frame.frame_id

    def _emit_takeover_report(self, report):
        report = dict(report, diagnostic_probe=bool(getattr(self, "in_reentry_probe", False)))
        self.last_takeover_report = _json_ready(report)
        path = os.environ.get("MAST3R_VINS_TAKEOVER_LOG")
        if path:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.last_takeover_report, sort_keys=True) + "\n")

    def log_takeover(self, frame, reason, accepted, **details):
        report = dict(
            frame_id=frame.frame_id, reason=reason, accepted=accepted,
            pose_only=bool(getattr(frame, "pose_only", False)),
            recovery_count=self.vins_recovery_count,
            degraded_frames=self.vins_degraded_frames,
            recovery_reference_id=(None if getattr(self, "recovery_reference", None) is None
                                   else getattr(self.recovery_reference, "frame_id", None)),
            window_anchor_frame_id=self.vins_window_frame_id,
            external_ground_truth_used=False,
        )
        report.update(details)
        self._emit_takeover_report(report)

    def recover_vins(self, frame, reason, *, recovery_pending=False):
        if not self.cfg.get("vins_visual_takeover", False):
            return False
        predicted = (
            self._predict_vins_pose_from_window_anchor(frame)
            if (
                self.cfg.get("vins_visual_safe_reentry", False)
                and self._has_vins_recovery_window()
            ) else None
        )
        if predicted is None:
            predicted = self.predict_vins_pose(frame)
        if predicted is None:
            self.log_takeover(frame, "vins_unavailable:" + reason, False)
            return False
        frame.T_WC = predicted
        frame.pose_only = True
        self.vins_degraded = True
        self.vins_degraded_frames += 1
        if not recovery_pending:
            self.vins_recovery_count = 0
            self.vins_recovery_frame_id = None
            if not (
                self.cfg.get("vins_visual_safe_reentry", False)
                and self._has_vins_recovery_window()
            ):
                self.reset_vins_recovery_window()
        self.reset_idx_f2k()
        self.log_takeover(frame, reason, True)
        return True

    def recover_visual_solve_failure(self, frame, error):
        if not isinstance(error, (VisualSolveRejected, torch.linalg.LinAlgError)):
            raise error
        return self.recover_vins(frame, "visual_solve_failed:" + str(error))

    def seed_recovery_reference(self, frame, raw_points, confidence, K):
        """Keep a physically scaled local reference outside the global graph.

        A rejected registration must not overwrite the old graph pointmap.
        Self geometry can nevertheless seed the next short-baseline attempt.
        Copies also keep SharedStates.set_frame from changing this reference.
        """
        (points,), report = scale_pointmaps_with_metric_depth(
            (raw_points,), confidence, frame.metric_depth, self.cfg
        )
        if not report.get("accepted") or not torch.isfinite(points).all():
            return False
        anchor = copy.copy(frame)
        anchor.T_WC = lietorch.Sim3(frame.T_WC.data.clone())
        anchor.X_canon = points.clone()
        anchor.C = confidence.clone()
        anchor.N = anchor.N_updates = 1
        anchor.K = None if K is None else K.clone()
        anchor.metric_anchor_mask = metric_depth_anchor_mask(
            points, confidence, frame.metric_depth, self.cfg
        )
        self.recovery_reference = anchor
        self.recovery_reference_scale = float(report["scale"])
        self.reset_idx_f2k()
        return True

    def accept_visual_recovery(self, visual_pose, frame):
        predicted = self.predict_vins_pose(frame)
        if predicted is None:
            self.vins_recovery_count = 0
            self.vins_recovery_frame_id = None
            # VINS initializes later than the first image. Do not prohibit a
            # healthy visual start merely because its prior is not ready yet.
            rows = self.vins_camera_poses
            previous = self.previous_frame
            if (
                not self.vins_degraded and rows is not None and previous is not None
                and 0 <= previous.frame_id < frame.frame_id < len(rows)
                and (rows[previous.frame_id]["valid"] != "1"
                     or rows[frame.frame_id]["valid"] != "1")
            ):
                return bool(torch.isfinite(visual_pose.data).all())
            return False
        consistent, report = self.visual_vins_consistency_report(visual_pose, frame)
        self.last_visual_consistency_report = report
        if not consistent:
            self.vins_recovery_count = 0
            self.vins_recovery_frame_id = None
            if not (
                self.cfg.get("vins_visual_safe_reentry", False)
                and self._has_vins_recovery_window()
            ):
                self.reset_vins_recovery_window()
            self._emit_takeover_report(dict(report, external_ground_truth_used=False))
            return False
        if self.vins_degraded:
            if (
                self.cfg.get("vins_visual_safe_reentry", False)
                and self.vins_window_visual_anchor is None
            ):
                self.anchor_vins_recovery_window(visual_pose, frame)
            if self.vins_recovery_frame_id == frame.frame_id:
                return False
            if self.vins_recovery_frame_id != frame.frame_id - 1:
                self.vins_recovery_count = 0
            self.vins_recovery_frame_id = frame.frame_id
            self.vins_recovery_count += 1
            required = int(self.cfg.get("vins_reentry_min_confirmations", 2))
            if self.vins_recovery_count < required:
                return False
        self.vins_degraded = False
        self.vins_recovery_count = 0
        self.vins_recovery_frame_id = None
        self.vins_degraded_frames = 0
        if self.recovery_reference is None:
            self.reset_vins_recovery_window()
        return True

    def _copy_probe_frame(self, frame):
        probe = copy.copy(frame)
        probe.T_WC = lietorch.Sim3(frame.T_WC.data.clone())
        for attr in ("X_canon", "C", "feat", "pos", "metric_anchor_mask"):
            value = getattr(frame, attr, None)
            if torch.is_tensor(value):
                setattr(probe, attr, value.clone())
        return probe

    def _graph_keyframe_count(self):
        try:
            return len(self.keyframes)
        except TypeError:
            length = getattr(self.keyframes, "__len__", None)
            if callable(length):
                try:
                    return int(length())
                except TypeError:
                    return int(length(self.keyframes))
            return 0

    def _snapshot_reentry_state(self):
        return {
            "idx_f2k": None if self.idx_f2k is None else self.idx_f2k.clone(),
            "vins_degraded": self.vins_degraded,
            "vins_recovery_count": self.vins_recovery_count,
            "vins_recovery_frame_id": self.vins_recovery_frame_id,
            "vins_degraded_frames": self.vins_degraded_frames,
            "vins_window_visual_anchor": None if self.vins_window_visual_anchor is None
            else self.vins_window_visual_anchor.copy(),
            "vins_window_pose_anchor": None if self.vins_window_pose_anchor is None
            else self.vins_window_pose_anchor.copy(),
            "vins_window_frame_id": self.vins_window_frame_id,
            "last_takeover_report": self.last_takeover_report,
            "last_visual_consistency_report": self.last_visual_consistency_report,
            "in_reentry_probe": self.in_reentry_probe,
            "metric_pointmap_scale": self.metric_pointmap_scale,
            "last_metric_anchor_mask": self.last_metric_anchor_mask,
            "vins_visual_anchor": None if self.vins_visual_anchor is None
            else self.vins_visual_anchor.copy(),
            "vins_pose_anchor": None if self.vins_pose_anchor is None
            else self.vins_pose_anchor.copy(),
        }

    def _restore_reentry_state(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def _probe_graph_reentry(self, frame, diagnostic_depth):
        graph_count = self._graph_keyframe_count()
        self.last_reentry_probe_report = None
        if graph_count <= 0:
            self.last_reentry_probe_report = {"accepted": False, "reason": "no_graph_anchor"}
            return None
        graph_anchor = self.keyframes[graph_count - 1]
        probe = self._copy_probe_frame(frame)
        state = self._snapshot_reentry_state()
        try:
            self.reset_idx_f2k()
            self.in_reentry_probe = True
            _, _, try_reloc = self.track(
                probe,
                diagnostic_depth=diagnostic_depth,
                reference_keyframe_index=graph_count - 1,
                update_reference=False,
                reentry_probe=True,
            )
            self.in_reentry_probe = False
            if try_reloc or bool(getattr(probe, "pose_only", False)):
                self.last_reentry_probe_report = {
                    "accepted": False,
                    "reason": "graph_probe_tracking_failed",
                    "try_reloc": bool(try_reloc),
                    "pose_only": bool(getattr(probe, "pose_only", False)),
                }
                return None
            accepted, report = self.metric_bridge_gate(graph_anchor, probe)
            probe.metric_bridge_report = report
            if not accepted:
                self.last_reentry_probe_report = report
                return None
            return probe
        finally:
            self._restore_reentry_state(state)

    def metric_bridge_gate(self, graph_anchor, frame):
        report = {
            "accepted": False,
            "reason": "not_evaluated",
            "first_frame_id": int(graph_anchor.frame_id),
            "second_frame_id": int(frame.frame_id),
            "geometry_source": "mandatory_bidirectional_d405_stereo_depth",
        }
        if not config["use_calib"]:
            report["reason"] = "calibration_required"
            return False, report
        if (
            getattr(graph_anchor, "metric_depth", None) is None
            or getattr(frame, "metric_depth", None) is None
            or not torch.isfinite(torch.as_tensor(graph_anchor.metric_depth)).any()
            or not torch.isfinite(torch.as_tensor(frame.metric_depth)).any()
        ):
            report["reason"] = "stereo_depth_required"
            return False, report
        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model,
            graph_anchor.feat,
            graph_anchor.pos,
            frame.feat,
            frame.pos,
            [graph_anchor.img_true_shape],
            [frame.img_true_shape],
        )
        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[:, None].repeat(
            1, idx_i2j.shape[1]
        )
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)
        valid_j = valid_match_j & (Qj > self.cfg["Q_conf"])
        valid_i = valid_match_i & (Qi > self.cfg["Q_conf"])
        match_frac_j = valid_j.float().mean()
        match_frac_i = valid_i.float().mean()
        min_match_frac = float(
            self.cfg.get("vins_reentry_bridge_min_match_frac", self.cfg["min_match_frac"])
        )
        min_match_frac = max(
            min_match_frac,
            float(config.get("local_opt", {}).get("min_match_frac", min_match_frac)),
        )
        report.update(
            match_frac_j=float(match_frac_j.item()),
            match_frac_i=float(match_frac_i.item()),
            min_match_frac=min_match_frac,
        )
        if min(float(match_frac_j.item()), float(match_frac_i.item())) < min_match_frac:
            report["reason"] = "bridge_match_fraction_low"
            return False, report

        from mast3r_slam.global_opt import FactorGraph

        graph_for_gate = copy.copy(graph_anchor)
        frame_for_gate = copy.copy(frame)
        for candidate in (graph_for_gate, frame_for_gate):
            if getattr(candidate, "metric_anchor_mask", None) is None:
                depth = torch.as_tensor(candidate.metric_depth, device=idx_i2j.device)
                candidate.metric_anchor_mask = torch.isfinite(depth.reshape(-1)) & (
                    depth.reshape(-1) > 0.0
                )

        gate = FactorGraph.__new__(FactorGraph)
        gate.cfg = self.cfg
        gate.K = graph_for_gate.K
        gate.device = idx_i2j.device
        if gate.K is None:
            report["reason"] = "bridge_intrinsics_required"
            return False, report
        accepted, metric_report = gate.metric_loop_gate(
            graph_for_gate,
            frame_for_gate,
            idx_i2j[0],
            valid_j[0],
            idx_j2i[0],
            valid_i[0],
        )
        report["metric_loop"] = metric_report
        if not accepted:
            report["reason"] = "metric_bridge_rejected"
            return False, report
        report["accepted"] = True
        report["reason"] = "metric_bridge_accepted"
        return True, report

    def try_safe_graph_reentry(self, frame, diagnostic_depth):
        if not self.cfg.get("vins_visual_safe_reentry", False):
            self.log_takeover(frame, "reentry_disabled", False)
            return False
        probe = self._probe_graph_reentry(frame, diagnostic_depth)
        if probe is None:
            report = self.last_reentry_probe_report or {
                "accepted": False,
                "reason": "reentry_probe_failed",
            }
            self._emit_takeover_report(dict(
                frame_id=frame.frame_id,
                reason="reentry_" + str(report.get("reason", "probe_failed")),
                accepted=False,
                reentry_probe=report,
                external_ground_truth_used=False,
            ))
            return False
        bridge_report = getattr(probe, "metric_bridge_report", {})
        if not bridge_report.get("accepted", False):
            self._emit_takeover_report(dict(
                frame_id=frame.frame_id,
                reason="reentry_metric_bridge_missing",
                accepted=False,
                metric_bridge=bridge_report,
                external_ground_truth_used=False,
            ))
            return False
        consistent, report = self.visual_vins_consistency_report(probe.T_WC, probe)
        self.last_visual_consistency_report = report
        if not consistent:
            self._emit_takeover_report(dict(
                report,
                frame_id=frame.frame_id,
                reason="reentry_" + report["reason"],
                external_ground_truth_used=False,
            ))
            return False
        for attr in ("T_WC", "X_canon", "C", "feat", "pos", "N", "N_updates",
                     "metric_anchor_mask"):
            if hasattr(probe, attr):
                setattr(frame, attr, getattr(probe, attr))
        frame.pose_only = False
        frame.requires_metric_bridge = True
        self.recovery_reference = None
        self.recovery_reference_scale = None
        self.vins_degraded = False
        self.vins_degraded_frames = 0
        self.vins_recovery_count = 0
        self.vins_recovery_frame_id = None
        self.reset_vins_recovery_window()
        self.log_takeover(
            frame,
            "visual_reentry_accepted",
            True,
            metric_bridge=bridge_report,
            consistency_report=report,
        )
        return True

    def configure_stereo_imu_anchor(self, increments, depth_getter):
        if depth_getter is None:
            raise ValueError("stereo/IMU anchor correction requires independent stereo depth")
        self.stereo_imu_anchor = dict(
            rotations=integrate_imu_rotations(increments),
            depth_getter=depth_getter,
            depth_cache={},
        )

    def configure_spatial_pointmap_recovery(self, increments):
        self.spatial_recovery_rotations = integrate_imu_rotations(increments)

    @torch.inference_mode()
    def recover_stereo_descriptor(self, frame, previous_frame, camera_matrix,
                                  rotation_prior):
        """Rescue one failed frame with independent descriptor/stereo geometry.

        The result is a pose-only observation, not a new graph keyframe or a
        replacement for the existing MASt3R pointmap factors.
        """
        from mast3r.fast_nn import fast_reciprocal_NNs

        if previous_frame.metric_depth is None or frame.metric_depth is None:
            return {"accepted": False, "reason": "stereo_depth_unavailable"}
        X, C, D, _ = mast3r_asymmetric_inference(
            self.model, previous_frame, frame
        )
        source_xy, target_xy = fast_reciprocal_NNs(
            D[0], D[1], subsample_or_initxy1=8,
            device=self.device, dist="dot", block_size=8192,
        )
        source_depth = np.asarray(previous_frame.metric_depth)
        target_depth = np.asarray(frame.metric_depth)
        pose, scale, report = metric_descriptor_pose(
            source_xy, target_xy, source_depth, target_depth,
            X[0, ..., 2].detach().cpu().numpy(),
            C[0].detach().cpu().numpy(), camera_matrix,
        )
        if pose is None and report.get("reason") == "too_few_stereo_matches":
            # D405 IR can drive pointmap confidence to its 1.0 floor while
            # reciprocal 2D matches remain geometrically valid. The relaxed
            # pass still has to clear PnP, two-view depth, scale, and IMU gates.
            pose, scale, report = metric_descriptor_pose(
                source_xy, target_xy, source_depth, target_depth,
                X[0, ..., 2].detach().cpu().numpy(),
                C[0].detach().cpu().numpy(), camera_matrix,
                minimum_confidence=1.0,
            )
            report["low_confidence_geometry_retry"] = True
        if pose is None:
            return report
        rotation_ok, rotation_error = imu_rotation_agrees(
            pose, rotation_prior
        )
        report["imu_rotation_disagreement_deg"] = rotation_error
        displacement = float(np.linalg.norm(pose[:3, 3]))
        report["step_translation_m"] = displacement
        if not rotation_ok or displacement > 0.03:
            return dict(report, accepted=False, reason="motion_prior_disagreement")
        relative = np.linalg.inv(pose)
        quaternion = Rotation.from_matrix(relative[:3, :3]).as_quat()
        metric_world = bool(self.cfg.get("stereo_pointmap_scale_prior", False))
        world_scale = float(previous_frame.T_WC.data[0, 7].item())
        relative_translation = descriptor_relative_translation(
            relative[:3, 3], pointmap_scale=scale,
            previous_world_scale=world_scale, metric_world=metric_world,
        )
        report["translation_unit_policy"] = "metric_world_pose_scale" if metric_world else "model_pointmap_scale"
        report["previous_world_scale"] = world_scale
        pose_data = frame.T_WC.data.new_zeros((1, 8))
        pose_data[0, :3] = torch.as_tensor(
            relative_translation, device=self.device, dtype=pose_data.dtype
        )
        pose_data[0, 3:7] = torch.as_tensor(
            quaternion, device=self.device, dtype=pose_data.dtype
        )
        pose_data[0, 7] = 1.0
        frame.T_WC = previous_frame.T_WC * lietorch.Sim3(pose_data)
        self.reset_idx_f2k()
        return report

    def correct_stereo_imu_anchor(self, frame, keyframe, relative_pose, valid,
                                  current_ids, keyframe_points, current_points,
                                  K, shape):
        state = self.stereo_imu_anchor
        if state is None:
            return None
        rotations = state["rotations"]
        imu_relative = rotations[keyframe.frame_id].inv() * rotations[frame.frame_id]
        visual_relative = relative_pose.data[0].detach().cpu().numpy()
        visual_rotation = Rotation.from_quat(visual_relative[3:7])
        if (visual_rotation.inv()*imu_relative).magnitude()*180/np.pi <= 1.0:
            return None
        cache = state["depth_cache"]
        for index in (keyframe.frame_id, frame.frame_id):
            if index not in cache:
                cache[index] = state["depth_getter"](index, tuple(shape))
        if len(cache) > 4:
            for index in list(cache):
                if index not in (keyframe.frame_id, frame.frame_id):
                    cache.pop(index)
        updated, report = propose_anchor_pose(
            visual_relative, imu_relative, valid.detach().cpu().numpy(),
            current_ids.detach().cpu().numpy(),
            keyframe_points.detach().cpu().numpy(),
            current_points.detach().cpu().numpy(), K.detach().cpu().numpy(),
            cache[keyframe.frame_id], cache[frame.frame_id])
        if updated is None:
            return None
        print("Stereo/IMU anchor correction", frame.frame_id, keyframe.frame_id, report)
        tensor = torch.as_tensor(updated, dtype=relative_pose.data.dtype,
                                 device=relative_pose.data.device).reshape(1, 8)
        return lietorch.Sim3(tensor)

    def scale_pointmaps(self, pointmaps, confidence, metric_depth):
        scaled, report = scale_pointmaps_with_metric_depth(
            pointmaps, confidence, metric_depth, self.cfg
        )
        self.last_metric_anchor_mask = metric_depth_anchor_mask(
            scaled[0], confidence, metric_depth, self.cfg
        )
        if report["accepted"]:
            scale = float(report["scale"])
            maximum_jump = float(self.cfg.get("stereo_scale_max_relative_jump", 0.25))
            if maximum_jump < 0.0:
                raise ValueError("tracking.stereo_scale_max_relative_jump must be non-negative")
            if self.metric_pointmap_scale is not None:
                relative_jump = abs(scale / self.metric_pointmap_scale - 1.0)
                if relative_jump > maximum_jump:
                    fallback = pointmaps[0].new_tensor(self.metric_pointmap_scale)
                    report = dict(
                        report,
                        accepted=False,
                        reason="metric_scale_jump_rejected",
                        relative_jump=relative_jump,
                        fallback_scale=self.metric_pointmap_scale,
                    )
                    return tuple(pointmap * fallback for pointmap in pointmaps), report
            self.metric_pointmap_scale = scale
            return scaled, report
        if self.metric_pointmap_scale is not None:
            fallback = pointmaps[0].new_tensor(self.metric_pointmap_scale)
            report = dict(report, fallback_scale=self.metric_pointmap_scale)
            return tuple(pointmap * fallback for pointmap in pointmaps), report
        return pointmaps, report

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def reset_metric_scale_state(self):
        """Drop the scale carried by the previous keyframe anchor."""
        self.metric_pointmap_scale = None

    def solve_metric_pose(
        self,
        Xf,
        Xk,
        Cf,
        Ck,
        Qk,
        valid_match_k,
        idx_f2k,
        keyframe_anchor_mask,
        current_anchor_mask,
        K,
        img_size,
    ):
        metric_valid = (
            valid_match_k[:, 0]
            & (Cf[:, 0] > self.cfg["C_conf"])
            & (Ck[:, 0] > self.cfg["C_conf"])
            & (Qk[:, 0] > self.cfg["Q_conf"])
            & keyframe_anchor_mask
            & current_anchor_mask[idx_f2k]
        )
        candidate_indices = torch.where(metric_valid)[0]
        maximum_points = int(self.cfg.get("stereo_keyframe_pnp_max_points", 5000))
        if candidate_indices.numel() > maximum_points:
            sampled = torch.linspace(
                0,
                candidate_indices.numel() - 1,
                maximum_points,
                device=candidate_indices.device,
            ).long()
            candidate_indices = candidate_indices[sampled]
        width = int(img_size[1])
        current_indices = idx_f2k[candidate_indices]
        image_points = torch.stack(
            (current_indices % width, current_indices // width), dim=-1
        ).float()
        keyframe_to_current, report = solve_metric_keyframe_pnp(
            Xk[candidate_indices].detach().cpu().numpy(),
            image_points.detach().cpu().numpy(),
            K.detach().cpu().numpy(),
            minimum_points=int(self.cfg.get("stereo_keyframe_pnp_min_points", 100)),
            minimum_inlier_ratio=float(
                self.cfg.get("stereo_keyframe_pnp_min_inlier_ratio", 0.4)
            ),
            reprojection_error_px=float(
                self.cfg.get("stereo_keyframe_pnp_reprojection_error_px", 2.0)
            ),
        )
        if keyframe_to_current is None:
            return None, report

        current_to_keyframe = np.linalg.inv(keyframe_to_current)
        if bool(self.cfg.get("stereo_keyframe_3d_translation_refine", False)):
            refined_pose, refinement_report = refine_metric_translation_3d3d(
                Xf[candidate_indices].detach().cpu().numpy(),
                Xk[candidate_indices].detach().cpu().numpy(),
                current_to_keyframe,
                minimum_points=int(
                    self.cfg.get("stereo_keyframe_3d_min_points", 100)
                ),
                minimum_inlier_ratio=float(
                    self.cfg.get("stereo_keyframe_3d_min_inlier_ratio", 0.3)
                ),
                maximum_residual_m=float(
                    self.cfg.get("stereo_keyframe_3d_max_residual_m", 0.02)
                ),
            )
            report = dict(report, translation_3d_refinement=refinement_report)
            if refined_pose is not None:
                current_to_keyframe = refined_pose

        quaternion = Rotation.from_matrix(current_to_keyframe[:3, :3]).as_quat()
        pose_data = Xf.new_zeros((1, 8))
        pose_data[:, :3] = torch.as_tensor(
            current_to_keyframe[:3, 3],
            device=pose_data.device,
            dtype=pose_data.dtype,
        )
        pose_data[:, 3:7] = torch.as_tensor(
            quaternion, device=pose_data.device, dtype=pose_data.dtype
        )
        pose_data[:, 7] = 1.0
        return lietorch.Sim3(pose_data), report

    def track(
        self,
        frame: Frame,
        diagnostic_depth=None,
        reference_keyframe_index=None,
        update_reference=True,
        reentry_probe=False,
    ):
        takeover = bool(self.cfg.get("vins_visual_takeover", False))
        shadow = getattr(self, "recovery_reference", None) if takeover else None
        using_shadow = (
            shadow is not None and reference_keyframe_index is None and not reentry_probe
        )
        keyframe = shadow if using_shadow else (
            self.keyframes.last_keyframe()
            if reference_keyframe_index is None
            else self.keyframes[reference_keyframe_index]
        )
        rotation_prior_pose = frame.T_WC
        previous_metric_scale = self.metric_pointmap_scale

        imu_relative = None
        if self.spatial_recovery_rotations is not None:
            rotations = self.spatial_recovery_rotations
            imu_relative = (
                rotations[frame.frame_id].inv() * rotations[keyframe.frame_id]
            ).as_quat()
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k,
            metric_distance_m=float(self.cfg.get("metric_match_distance_m", 0.0)),
            weak_stereo_recovery=self.spatial_recovery_rotations is not None,
            camera_matrix=keyframe.K,
            imu_relative_quaternion=imu_relative,
        )
        raw_current_points = Xff
        if using_shadow:
            self.metric_pointmap_scale = self.recovery_reference_scale
        try:
            (Xff, Xkf), stereo_scale_report = self.scale_pointmaps(
                (Xff, Xkf), Cff,
                frame.metric_depth
                if self.cfg.get("stereo_pointmap_scale_prior", False) else None,
            )
        finally:
            if using_shadow:
                self.metric_pointmap_scale = previous_metric_scale
        frame.metric_anchor_mask = self.last_metric_anchor_mask
        if (
            self.cfg.get("stereo_pointmap_scale_prior", False)
            and frame.frame_id % 30 == 0
        ):
            print("Stereo pointmap scale", frame.frame_id, stereo_scale_report)
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        if takeover:
            minimum = int(self.cfg.get("stereo_scale_min_points", 500))
            confidence = float(self.cfg.get("stereo_scale_min_confidence", 1.5))
            reference_support = int((Ckf >= confidence).sum().item())
            if not stereo_scale_report.get("accepted") or reference_support < minimum:
                self.metric_pointmap_scale = previous_metric_scale
                accepted = self.recover_vins(frame, "unreliable_pointmap_geometry")
                if accepted:
                    self.seed_recovery_reference(frame, raw_current_points, Cff, keyframe.K)
                return False, [], not accepted

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        diagnostic_scale = None
        if _FRONTEND_LOG and diagnostic_depth is not None:
            diagnostic_scale = robust_pointmap_metric_scale(
                Xff[..., 2].detach().cpu().numpy(),
                Cff.detach().cpu().numpy(),
                diagnostic_depth,
                minimum_points=int(self.cfg.get("stereo_scale_min_points", 500)),
                minimum_depth_m=float(self.cfg.get("stereo_scale_min_depth_m", 0.15)),
                maximum_depth_m=float(self.cfg.get("stereo_scale_max_depth_m", 0.65)),
                minimum_confidence=float(self.cfg.get("stereo_scale_min_confidence", 1.5)),
            )

        # Confidence as a *proportional* weight, not just a hard gate.
        # Upstream only ever uses C at `valid_Cf/valid_Ck` above, and with the
        # shipped `C_conf: 0.0` that gate is always open, so C never reaches the
        # normal equations (the residuals are whitened by `valid * sqrt(Qk)`
        # only). Measured C is confined to [1.0, 1.5] with 36% of matches at
        # C<=1.02, and those currently pull with the same weight as good ones.
        # `C_weight_floor: 1.0` (default) reproduces the upstream weighting
        # exactly -- 1.0 + 0.0*t is exactly 1.0, and multiplying by 1.0 is exact.
        w_floor = float(self.cfg.get("C_weight_floor", 1.0))
        c_lo = float(self.cfg.get("C_weight_lo", 1.0))
        c_hi = float(self.cfg.get("C_weight_hi", 1.5))
        conf_w = w_floor + (1.0 - w_floor) * (
            ((Cf + Ck) * 0.5 - c_lo) / max(c_hi - c_lo, 1e-6)
        ).clamp(0.0, 1.0)

        match_frac = valid_opt.sum() / valid_opt.numel()
        log_match_stats(frame.frame_id, valid_match_k, valid_kf, valid_opt)
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            if takeover:
                self.metric_pointmap_scale = previous_metric_scale
                accepted = self.recover_vins(frame, "insufficient_visual_matches")
                if accepted:
                    self.seed_recovery_reference(frame, raw_current_points, Cff, keyframe.K)
                return False, [], not accepted
            return False, [], True

        metric_pose = None
        metric_pose_report = None
        if bool(self.cfg.get("stereo_keyframe_pnp", False)) and use_calib:
            metric_pose, metric_pose_report = self.solve_metric_pose(
                Xf,
                Xk,
                Cf,
                Ck,
                Qk,
                valid_match_k,
                idx_f2k,
                keyframe.metric_anchor_mask,
                frame.metric_anchor_mask,
                K,
                img_size,
            )

            lookback = int(self.cfg.get("stereo_keyframe_pnp_lookback", 1))
            if lookback > 1 and len(self.keyframes) > 1:
                current_report = metric_pose_report
                current_world_pose = (
                    None if metric_pose is None else T_WCk * metric_pose
                )
                previous_keyframe = self.keyframes[len(self.keyframes) - 2]
                (
                    previous_idx,
                    previous_valid_match,
                    previous_Xff,
                    previous_Cff,
                    previous_Qff,
                    _,
                    _,
                    previous_Qkf,
                ) = mast3r_match_asymmetric(self.model, frame, previous_keyframe)
                (previous_Xff,), previous_scale_report = (
                    scale_pointmaps_with_metric_depth(
                        (previous_Xff,), previous_Cff, frame.metric_depth, self.cfg
                    )
                )
                previous_current_anchor = metric_depth_anchor_mask(
                    previous_Xff,
                    previous_Cff,
                    frame.metric_depth,
                    self.cfg,
                )
                previous_idx = previous_idx[0]
                previous_valid_match = previous_valid_match[0]
                previous_current_points = min(
                    previous_Qff.shape[0],
                    previous_Xff.shape[0],
                    previous_Cff.shape[0],
                    previous_current_anchor.numel(),
                )
                safe_previous_idx = previous_idx.clamp(
                    min=0, max=previous_current_points - 1
                )
                previous_Qk = torch.sqrt(
                    previous_Qff[safe_previous_idx] * previous_Qkf
                )
                previous_Xf = constrain_points_to_ray(
                    img_size, previous_Xff[None], K
                ).squeeze(0)[safe_previous_idx]
                previous_Xk = constrain_points_to_ray(
                    img_size, previous_keyframe.X_canon[None], K
                ).squeeze(0)
                previous_pose, previous_report = self.solve_metric_pose(
                    previous_Xf,
                    previous_Xk,
                    previous_Cff[safe_previous_idx],
                    previous_keyframe.get_average_conf(),
                    previous_Qk,
                    previous_valid_match,
                    safe_previous_idx,
                    previous_keyframe.metric_anchor_mask,
                    previous_current_anchor,
                    K,
                    img_size,
                )
                metric_pose_report = dict(
                    current_report or {},
                    previous_keyframe=previous_report,
                    previous_scale=previous_scale_report,
                )
                if previous_pose is not None:
                    previous_world_pose = previous_keyframe.T_WC * previous_pose
                    translation_disagreement = float(
                        torch.linalg.vector_norm(
                            previous_world_pose.data[..., :3]
                            - current_world_pose.data[..., :3]
                        ).item()
                    ) if current_world_pose is not None else 0.0
                    rotation_disagreement = float(
                        rotation_disagreement_deg(
                            previous_world_pose, current_world_pose
                        ).item()
                    ) if current_world_pose is not None else 0.0
                    metric_pose_report["previous_world_disagreement_m"] = (
                        translation_disagreement
                    )
                    metric_pose_report["previous_world_disagreement_deg"] = (
                        rotation_disagreement
                    )
                    maximum_translation = float(
                        self.cfg.get(
                            "stereo_keyframe_pnp_history_max_disagreement_m", 0.03
                        )
                    )
                    maximum_rotation = float(
                        self.cfg.get(
                            "stereo_keyframe_pnp_history_max_disagreement_deg", 3.0
                        )
                    )
                    current_reprojection = float(
                        (current_report or {}).get("reprojection_p95_px", float("inf"))
                    )
                    previous_reprojection = float(
                        previous_report.get("reprojection_p95_px", float("inf"))
                    )
                    if (
                        translation_disagreement <= maximum_translation
                        and rotation_disagreement <= maximum_rotation
                        and previous_reprojection < current_reprojection
                    ):
                        metric_pose = previous_world_pose
                        metric_pose_report["selected_keyframe_age"] = (
                            frame.frame_id - previous_keyframe.frame_id
                        )

        try:
            # Track
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt, conf_w
                )
            else:
                metric_translation_target = self.vins_translation_target(
                    keyframe, frame.frame_id
                )
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    conf_w,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                    metric_translation_target,
                    float(T_WCk.data[0, 7].item()),
                )
        except Exception as e:
            if takeover:
                self.metric_pointmap_scale = previous_metric_scale
                accepted = self.recover_visual_solve_failure(frame, e)
                if accepted:
                    self.seed_recovery_reference(frame, raw_current_points, Cff, keyframe.K)
                return False, [], not accepted
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        if metric_pose is not None:
            metric_pose_is_world = int(
                self.cfg.get("stereo_keyframe_pnp_lookback", 1)
            ) > 1 and (
                metric_pose_report or {}
            ).get("selected_keyframe_age") is not None
            metric_world_pose = metric_pose if metric_pose_is_world else T_WCk * metric_pose
            translation_disagreement = float(
                torch.linalg.vector_norm(
                    metric_world_pose.data[..., :3] - T_WCf.data[..., :3]
                ).item()
            )
            rotation_disagreement = float(
                rotation_disagreement_deg(metric_world_pose, T_WCf).item()
            )
            metric_pose_report = dict(
                metric_pose_report or {},
                visual_translation_disagreement_m=translation_disagreement,
                visual_rotation_disagreement_deg=rotation_disagreement,
            )
            maximum_translation = float(
                self.cfg.get("stereo_keyframe_pnp_max_visual_disagreement_m", 0.05)
            )
            maximum_rotation = float(
                self.cfg.get("stereo_keyframe_pnp_max_visual_disagreement_deg", 5.0)
            )
            if (
                translation_disagreement <= maximum_translation
                and rotation_disagreement <= maximum_rotation
            ):
                T_WCf = metric_world_pose
                T_CkCf = T_WCk.inv() * T_WCf
            else:
                metric_pose_report["accepted"] = False
                metric_pose_report["reason"] = "metric_pnp_visual_disagreement"
        if frame.frame_id % 30 == 0 and metric_pose_report is not None:
            print("Stereo keyframe PnP", frame.frame_id, metric_pose_report)

        prior_weight = float(self.cfg.get("imu_rotation_constraint_weight", 0.0))
        prior_gate_deg = float(self.cfg.get("imu_rotation_constraint_gate_deg", 0.0))
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError("tracking.imu_rotation_constraint_weight must be within [0, 1]")
        if prior_gate_deg < 0.0:
            raise ValueError("tracking.imu_rotation_constraint_gate_deg must be non-negative")
        disagreement_deg = float(
            rotation_disagreement_deg(T_WCf, rotation_prior_pose).item()
        )
        if prior_weight > 0.0 and disagreement_deg > prior_gate_deg:
            gated_weight = prior_weight * (
                1.0 - prior_gate_deg / max(disagreement_deg, 1e-6)
            )
            T_WCf = blend_sim3_rotation(T_WCf, rotation_prior_pose, gated_weight)
            T_CkCf = T_WCk.inv() * T_WCf

        if bool(self.cfg.get("stereo_fix_pose_scale", False)):
            T_WCf = force_unit_sim3_scale(T_WCf)
            T_CkCf = T_WCk.inv() * T_WCf

        if self.stereo_imu_anchor is not None:
            corrected = self.correct_stereo_imu_anchor(
                frame, keyframe, T_CkCf, valid_opt, idx_f2k, Xk, Xf, K, img_size)
            if corrected is not None:
                T_CkCf = corrected
                T_WCf = T_WCk * T_CkCf

        if takeover:
            if not self.accept_visual_recovery(T_WCf, frame):
                self.metric_pointmap_scale = previous_metric_scale
                accepted = self.recover_vins(
                    frame, "visual_vins_consistency_or_recovery_pending",
                    recovery_pending=self.vins_recovery_count > 0,
                )
                if accepted:
                    self.seed_recovery_reference(frame, raw_current_points, Cff, keyframe.K)
                return False, [], not accepted
            frame.pose_only = False
            details = {}
            if self.cfg.get("vins_visual_safe_reentry", False):
                details["consistency_report"] = self.last_visual_consistency_report
            self.log_takeover(frame, "visual_accepted", True, **details)

        frame.T_WC = T_WCf
        if using_shadow:
            if self.try_safe_graph_reentry(frame, diagnostic_depth):
                return True, [], False
            self.seed_recovery_reference(frame, raw_current_points, Cff, keyframe.K)
            # Recovery observations stay out of SharedKeyframes/retrieval and
            # cannot introduce an unchecked consecutive edge to the stale map.
            return False, [], False
        if self.vins_camera_poses is not None and self.vins_visual_anchor is None:
            prior_row = self.vins_camera_poses[frame.frame_id]
            if prior_row["valid"] == "1":
                self.vins_visual_anchor = T_WCf.data[0].detach().cpu().numpy().copy()
                self.vins_pose_anchor = self.vins_pose_from_row(prior_row)

        if _FRONTEND_LOG:
            valid_depth = valid_opt.squeeze(-1) & (Xf[:, 2] > 0) & (Xk[:, 2] > 0)
            zf = (
                float(Xf[valid_depth, 2].median().item())
                if valid_depth.any() else float("nan")
            )
            zk = (
                float(Xk[valid_depth, 2].median().item())
                if valid_depth.any() else float("nan")
            )
            new_log = not os.path.exists(_FRONTEND_LOG)
            with open(_FRONTEND_LOG, "a") as stream:
                if new_log:
                    stream.write(
                        "frame_id,keyframe_id,n_opt,n_total,"
                        "pointmap_z_current,pointmap_z_keyframe,"
                        "metric_pointmap_scale,metric_relative_mad,metric_points,"
                        "relative_translation,relative_scale\n"
                    )
                report = diagnostic_scale or {}
                stream.write(
                    f"{frame.frame_id},{keyframe.frame_id},{int(valid_opt.sum())},{valid_opt.numel()},"
                    f"{zf},{zk},{report.get('scale', float('nan'))},"
                    f"{report.get('relative_mad', float('nan'))},"
                    f"{report.get('inlier_points', 0)},"
                    f"{float(torch.linalg.vector_norm(T_CkCf.data[..., :3]).item())},"
                    f"{float(T_CkCf.data[..., 7].item())}\n"
                )

        # A short-window retry may use an older keyframe only to recover this
        # frame's pose. It must not rewrite that keyframe or create graph edges.
        if not update_reference:
            return False, [], False

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        preserve_metric_keyframe = bool(
            self.cfg.get("stereo_pointmap_depth_anchor", False)
            and self.cfg.get("stereo_preserve_keyframe_pointmap", False)
        )
        if not preserve_metric_keyframe:
            keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]
        motion_triggered, relative_translation, relative_rotation_deg = (
            motion_keyframe_trigger(
                T_CkCf,
                rotation_prior_pose,
                T_WCk,
                self.cfg,
                frame.frame_id - keyframe.frame_id,
            )
        )
        if motion_triggered and not new_kf:
            print(
                "Motion keyframe"
                f" {frame.frame_id}: translation={relative_translation:.4f},"
                f" rotation={relative_rotation_deg:.3f}deg"
            )
            new_kf = True

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()
            self.reset_metric_scale_state()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J, visual_row_count=0, visual_effective_count=1):
        whitened_r = sqrt_info * r
        log_robust_sample(whitened_r)
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        if visual_row_count:
            robust_sqrt_info[:visual_row_count] /= np.sqrt(visual_effective_count)
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def solve_pose_increment(
        self, sqrt_info, r, J, visual_row_count=0, visual_effective_count=1
    ):
        fixed_scale = bool(self.cfg.get("stereo_fix_pose_scale", False))
        tau, cost = self.solve(
            sqrt_info, r, pose_optimization_jacobian(J, fixed_scale),
            visual_row_count=visual_row_count,
            visual_effective_count=visual_effective_count,
        )
        return embed_pose_increment(tau, fixed_scale), cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid, conf_w):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * conf_w * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * conf_w * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve_pose_increment(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    @staticmethod
    def vins_pose_from_row(row):
        return np.asarray(
            [float(row[axis]) for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        )

    def vins_translation_target(self, keyframe, frame_id):
        if self.vins_camera_poses is None:
            return None
        if frame_id >= len(self.vins_camera_poses):
            raise ValueError("VINS camera pose priors end before the tracked frame")
        keyframe_row = self.vins_camera_poses[keyframe.frame_id]
        current = self.vins_camera_poses[frame_id]
        if current["valid"] != "1":
            return None
        current_pose = self.vins_pose_from_row(current)
        if keyframe_row["valid"] == "1":
            keyframe_pose = self.vins_pose_from_row(keyframe_row)
            return Rotation.from_quat(keyframe_pose[3:7]).inv().apply(
                current_pose[:3] - keyframe_pose[:3]
            )
        if self.vins_visual_anchor is None:
            return None
        return anchored_camera_local_translation_target(
            keyframe.T_WC.data[0].detach().cpu().numpy(),
            self.vins_visual_anchor,
            self.vins_pose_anchor,
            current_pose[:3],
        )

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, conf_w, meas_k, valid_meas_k, K, img_size,
        metric_translation_target=None,
        metric_world_scale=1.0,
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * conf_w * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * conf_w * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf
            visual_row_count = 0
            visual_effective_count = 1

            if metric_translation_target is not None:
                if self.cfg.get("vins_translation_prior_normalize_visual", False):
                    visual_row_count = r.shape[0]
                    visual_effective_count = max(int(valid2.sum().item()), 1)
                prior_r, prior_J, prior_info = metric_translation_prior_terms(
                    T_CkCf, metric_translation_target,
                    self.vins_translation_prior_sigma_m,
                    world_scale=metric_world_scale,
                )
                r = torch.cat((r, prior_r), dim=0)
                J = torch.cat((J, prior_J), dim=0)
                sqrt_info2 = torch.cat((sqrt_info2, prior_info), dim=0)

            tau_ij_sim3, new_cost = self.solve_pose_increment(
                sqrt_info2, r, J,
                visual_row_count=visual_row_count,
                visual_effective_count=visual_effective_count,
            )
            if self.cfg.get("vins_visual_takeover", False) and metric_translation_target is not None:
                if not math.isfinite(new_cost) or not bool(torch.isfinite(tau_ij_sim3).all()):
                    raise VisualSolveRejected("nonfinite_visual_solve")
                if new_cost > old_cost * (1.0 + 1e-6):
                    raise VisualSolveRejected("uphill_visual_solve")
                # Return only a state whose cost has actually been evaluated.
                # The legacy loop applied an unevaluated terminal increment.
                if check_convergence(
                    step, self.cfg["rel_error"], self.cfg["delta_norm"],
                    old_cost, new_cost, tau_ij_sim3,
                ):
                    break
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                if self.cfg.get("vins_visual_takeover", False) and metric_translation_target is not None:
                    raise VisualSolveRejected("visual_solve_max_iterations")
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
