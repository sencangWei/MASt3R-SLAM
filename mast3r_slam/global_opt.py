import lietorch
import numpy as np
import torch
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
from mast3r_slam.stereo_depth import (
    force_unit_sim3_scale,
    solve_metric_keyframe_pnp,
)
import mast3r_slam_backends


class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.window_size = self.cfg["window_size"]

        self.K = K

    def _metric_pnp_direction(
        self, source_frame, target_frame, target_to_source_index, valid_target
    ):
        """Estimate one directed camera motion from source stereo depth."""
        cfg = self.cfg
        report = {"accepted": False}
        if self.K is None:
            report["reason"] = "camera_intrinsics_unavailable"
            return None, report
        if (
            source_frame.metric_anchor_mask is None
            or target_frame.metric_anchor_mask is None
            or source_frame.metric_depth is None
        ):
            report["reason"] = "metric_depth_unavailable"
            return None, report

        pixel_count = int(target_to_source_index.numel())
        current_indices = torch.arange(pixel_count, device=self.device)
        source_indices = target_to_source_index.reshape(-1).long()
        valid = valid_target.reshape(-1).bool()
        valid &= source_indices >= 0
        valid &= source_indices < source_frame.metric_anchor_mask.numel()
        valid &= target_frame.metric_anchor_mask[:pixel_count]
        safe_source_indices = source_indices.clamp(
            min=0, max=source_frame.metric_anchor_mask.numel() - 1
        )
        valid &= source_frame.metric_anchor_mask[safe_source_indices]
        candidates = current_indices[valid]
        maximum_points = int(cfg.get("metric_loop_pnp_max_points", 5000))
        if candidates.numel() > maximum_points:
            sample = torch.linspace(
                0,
                candidates.numel() - 1,
                maximum_points,
                device=candidates.device,
            ).long()
            candidates = candidates[sample]
        if candidates.numel() == 0:
            report["reason"] = "no_metric_correspondences"
            return None, report

        source_indices = safe_source_indices[candidates].cpu()
        candidates = candidates.cpu()
        source_width = int(source_frame.img.shape[-1])
        target_width = int(target_frame.img.shape[-1])
        source_depth = source_frame.metric_depth.reshape(-1)[source_indices]
        finite_depth = torch.isfinite(source_depth) & (source_depth > 0.0)
        candidates = candidates[finite_depth]
        source_indices = source_indices[finite_depth]
        source_depth = source_depth[finite_depth]
        if candidates.numel() == 0:
            report["reason"] = "no_finite_metric_depth"
            return None, report
        source_u = (source_indices % source_width).to(source_depth.dtype)
        source_v = (source_indices // source_width).to(source_depth.dtype)
        camera_matrix = self.K.detach().cpu()
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        object_points = torch.stack(
            (
                (source_u - cx) * source_depth / fx,
                (source_v - cy) * source_depth / fy,
                source_depth,
            ),
            dim=-1,
        )
        image_points = torch.stack(
            (candidates % target_width, candidates // target_width), dim=-1
        ).float()
        transform, pnp_report = solve_metric_keyframe_pnp(
            object_points.detach().cpu().numpy(),
            image_points.detach().cpu().numpy(),
            camera_matrix.numpy(),
            minimum_points=int(cfg.get("metric_loop_pnp_min_points", 100)),
            minimum_inlier_ratio=float(
                cfg.get("metric_loop_pnp_min_inlier_ratio", 0.5)
            ),
            reprojection_error_px=float(
                cfg.get("metric_loop_pnp_reprojection_error_px", 2.0)
            ),
        )
        report.update(pnp_report)
        maximum_p95 = float(
            cfg.get("metric_loop_pnp_max_reprojection_p95_px", 4.0)
        )
        if not pnp_report.get("accepted"):
            return None, report
        if float(pnp_report["reprojection_p95_px"]) > maximum_p95:
            report["accepted"] = False
            report["reason"] = "metric_pnp_reprojection_p95_high"
            return None, report
        return transform, report

    def metric_loop_gate(
        self,
        frame_i,
        frame_j,
        index_j_to_i,
        valid_j,
        index_i_to_j,
        valid_i,
    ):
        """Validate a retrieval with bidirectional onboard-stereo geometry."""
        report = {
            "accepted": False,
            "first_frame_id": int(frame_i.frame_id),
            "second_frame_id": int(frame_j.frame_id),
            "geometry_source": "bidirectional_d405_stereo_depth",
        }
        forward, forward_report = self._metric_pnp_direction(
            frame_i, frame_j, index_j_to_i, valid_j
        )
        report["forward"] = forward_report
        if forward is None:
            report["reason"] = "forward_metric_pnp_failed"
            return False, report

        reverse, reverse_report = self._metric_pnp_direction(
            frame_j, frame_i, index_i_to_j, valid_i
        )
        report["reverse"] = reverse_report
        if reverse is None:
            report["reason"] = "reverse_metric_pnp_failed"
            return False, report

        cycle = reverse @ forward
        cycle_translation_m = float(np.linalg.norm(cycle[:3, 3]))
        rotation_cosine = np.clip((np.trace(cycle[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        cycle_rotation_deg = float(np.degrees(np.arccos(rotation_cosine)))
        report["cycle_translation_m"] = cycle_translation_m
        report["cycle_rotation_deg"] = cycle_rotation_deg
        if cycle_translation_m > float(
            self.cfg.get("metric_loop_pnp_max_cycle_translation_m", 0.01)
        ):
            report["reason"] = "metric_pnp_cycle_translation_high"
            return False, report
        if cycle_rotation_deg > float(
            self.cfg.get("metric_loop_pnp_max_cycle_rotation_deg", 2.0)
        ):
            report["reason"] = "metric_pnp_cycle_rotation_high"
            return False, report
        report["accepted"] = True
        return True, report

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

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
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if bool(self.cfg.get("metric_loop_gate", False)):
            for edge_index, consecutive in enumerate(consecutive_edges.tolist()):
                if consecutive or bool(invalid_edges[edge_index].item()):
                    continue
                accepted, report = self.metric_loop_gate(
                    kf_ii[edge_index],
                    kf_jj[edge_index],
                    idx_i2j[edge_index],
                    valid_j[edge_index],
                    idx_j2i[edge_index],
                    valid_i[edge_index],
                )
                print("Metric loop gate", report)
                if not accepted:
                    invalid_edges[edge_index] = True

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self):
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        if bool(config["tracking"].get("stereo_fix_pose_scale", False)):
            T_WCs = force_unit_sim3_scale(T_WCs)

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
