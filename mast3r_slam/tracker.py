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
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.stereo_depth import (
    embed_pose_increment,
    force_unit_sim3_scale,
    metric_depth_anchor_mask,
    pose_optimization_jacobian,
    refine_metric_translation_3d3d,
    scale_pointmaps_with_metric_depth,
    solve_metric_keyframe_pnp,
)


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


class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device
        self.metric_pointmap_scale = None
        self.last_metric_anchor_mask = None

        self.reset_idx_f2k()

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

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()
        rotation_prior_pose = frame.T_WC

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        (Xff, Xkf), stereo_scale_report = self.scale_pointmaps(
            (Xff, Xkf), Cff, frame.metric_depth
        )
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

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
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
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
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

        frame.T_WC = T_WCf

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

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def solve_pose_increment(self, sqrt_info, r, J):
        fixed_scale = bool(self.cfg.get("stereo_fix_pose_scale", False))
        tau, cost = self.solve(
            sqrt_info, r, pose_optimization_jacobian(J, fixed_scale)
        )
        return embed_pose_increment(tau, fixed_scale), cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
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

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
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

            tau_ij_sim3, new_cost = self.solve_pose_increment(sqrt_info2, r, J)
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
