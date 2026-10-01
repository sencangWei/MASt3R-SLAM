"""Offline A/B of one saved calibrated keyframe graph (diagnostic only).

The crossed pose/geometry runs test sensitivity of the solver's basin. They
are not valid SLAM trajectories and must never be scored as production output.
"""

import argparse
import json

import torch
import mast3r_slam_backends


def checked_pair(control, spatial):
    if control["frame_ids"] != spatial["frame_ids"]:
        raise ValueError("snapshots have different keyframe identities")
    if len(control["args"]) != 19 or len(spatial["args"]) != 19:
        raise ValueError("expected calibrated-GN input arguments")
    for index in range(9, 19):
        if control["args"][index] != spatial["args"][index]:
            raise ValueError(f"solver setting {index} differs across snapshots")
    if not torch.equal(control["args"][3], spatial["args"][3]):
        raise ValueError("camera intrinsics differ across snapshots")


def swapped_keyframe_geometry(base, donor, frame_id, slots=(1, 2)):
    index = base["frame_ids"].index(frame_id)
    args = list(base["args"])
    for slot in slots:  # Ray-constrained 3D pointmap and confidence.
        replacement = args[slot].clone()
        replacement[index] = donor["args"][slot][index]
        args[slot] = replacement
    return {"frame_ids": base["frame_ids"], "args": tuple(args)}


def solve(pose_snapshot, geometry_snapshot):
    args = list(geometry_snapshot["args"])
    args[0] = pose_snapshot["args"][0]
    args = [value.clone().cuda() if isinstance(value, torch.Tensor) else value
            for value in args]
    before_poses = args[0].detach().cpu().clone()
    with torch.no_grad():
        mast3r_slam_backends.gauss_newton_calib(*args)
    torch.cuda.synchronize()
    after_poses = args[0].detach().cpu()
    scale_change = torch.abs(torch.log(after_poses[:, 7] / before_poses[:, 7]))
    position_change = torch.linalg.vector_norm(
        after_poses[:, :3] - before_poses[:, :3], dim=1
    )
    frame_ids = pose_snapshot["frame_ids"]
    return {
        "previous_frame_id": frame_ids[-2],
        "current_frame_id": frame_ids[-1],
        "previous_scale_before": float(before_poses[-2, 7]),
        "current_scale_before": float(before_poses[-1, 7]),
        "previous_scale_after": float(after_poses[-2, 7]),
        "current_scale_after": float(after_poses[-1, 7]),
        "max_abs_log_scale_update": float(scale_change.max()),
        "max_position_update_model_units": float(position_change.max()),
        "edge_directions": int(args[4].numel()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--spatial", required=True)
    parser.add_argument("--swap-frame-id", type=int)
    paths = parser.parse_args()
    control = torch.load(paths.control, map_location="cpu", weights_only=True)
    spatial = torch.load(paths.spatial, map_location="cpu", weights_only=True)
    checked_pair(control, spatial)
    output = {
        "diagnostic_only": True,
        "slam_supervision": False,
        "scale_unit": "dimensionless_sim3",
        "correspondence_tensors_identical": all(
            torch.equal(control["args"][index], spatial["args"][index])
            for index in range(4, 9)
        ),
        "pointmaps_identical": torch.equal(control["args"][1], spatial["args"][1]),
        "runs": {},
    }
    for pose_name, pose in (("control", control), ("spatial", spatial)):
        for geometry_name, geometry in (("control", control), ("spatial", spatial)):
            output["runs"][f"{pose_name}_pose__{geometry_name}_geometry"] = solve(
                pose, geometry
            )
    if paths.swap_frame_id is not None:
        frame_id = paths.swap_frame_id
        for label, slots in (("geometry", (1, 2)), ("pointmap", (1,)),
                             ("confidence", (2,))):
            output["runs"][f"spatial_with_control_{frame_id}_{label}"] = solve(
                spatial, swapped_keyframe_geometry(spatial, control, frame_id, slots)
            )
            output["runs"][f"control_with_spatial_{frame_id}_{label}"] = solve(
                control, swapped_keyframe_geometry(control, spatial, frame_id, slots)
            )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
