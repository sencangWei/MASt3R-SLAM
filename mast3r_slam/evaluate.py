import csv
from contextlib import ExitStack
import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_full_traj(logdir, logfile, timestamps, frames, tracked_poses, *, online_poses=None):
    """Save tracked camera poses after re-anchoring them to optimized keyframes.

    Each record stores the frame pose relative to the keyframe that tracked it.
    Applying the final optimized keyframe pose propagates offline backend and loop
    corrections without inventing poses from an external trajectory.
    """
    logfile = pathlib.Path(logdir) / logfile
    logfile.parent.mkdir(exist_ok=True, parents=True)
    if online_poses is not None:
        tracked_ids = [frame_id for frame_id, _, _ in tracked_poses]
        if (tracked_ids != [frame_id for frame_id, _ in online_poses]
                or len(set(tracked_ids)) != len(tracked_ids)):
            raise ValueError("diagnostic online/tracked frame IDs do not bind")
    with ExitStack() as stack:
        writer = None
        if online_poses is not None:
            statefile = logfile.with_name(f"{logfile.stem}_sim3_states.csv")
            state_stream = stack.enter_context(statefile.open("x", newline=""))
            writer = csv.writer(state_stream)
            components = ("tx", "ty", "tz", "qx", "qy", "qz", "qw", "s")
            writer.writerow(["frame_id", "t_sec", "anchor_idx", "anchor_frame_id"] +
                            [f"{prefix}_{name}" for prefix in
                             ("relative", "online", "final_anchor", "final")
                             for name in components])
        stream = stack.enter_context(logfile.open("w"))
        for index, (frame_id, anchor_idx, relative_pose_data) in enumerate(tracked_poses):
            keyframe = frames[anchor_idx]
            anchor = keyframe.T_WC
            if writer is not None:
                anchor = type(anchor)(anchor.data.detach().clone())
            relative = type(anchor)(relative_pose_data.to(anchor.data.device))
            final_pose = anchor * relative
            T_WC = as_SE3(final_pose)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            stream.write(
                f"{timestamps[frame_id]} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
            )
            if writer is not None:
                values = [data.detach().cpu().reshape(-1).tolist() for data in
                          (relative_pose_data, online_poses[index][1],
                           anchor.data, final_pose.data)]
                writer.writerow([frame_id, timestamps[frame_id], anchor_idx,
                                 keyframe.frame_id] + [value for row in values for value in row])


def save_online_traj(logdir, logfile, timestamps, frames, online_poses):
    """Save the poses emitted by the frontend before backend re-anchoring."""
    logfile = pathlib.Path(logdir) / logfile
    logfile.parent.mkdir(exist_ok=True, parents=True)
    pose_type = type(frames[0].T_WC)
    device = frames[0].T_WC.data.device
    scale_logfile = logfile.with_name(f"{logfile.stem}_sim3_scale.csv")
    with logfile.open("w") as stream, scale_logfile.open("w") as scale_stream:
        scale_stream.write("frame_id,t_sec,sim3_scale\n")
        for frame_id, pose_data in online_poses:
            T_WC = as_SE3(pose_type(pose_data.to(device)))
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            stream.write(
                f"{timestamps[frame_id]} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
            )
            scale_stream.write(
                f"{frame_id},{timestamps[frame_id]},{float(pose_data.reshape(-1)[7])}\n"
            )


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
