"""Camera-local VINS priors for an opt-in metric keyframe graph factor."""

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class MetricKeyframePrior:
    def __init__(self, path):
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or [int(row["input_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError("VINS camera priors need contiguous frame indices")
        self.poses = {}
        for row in rows:
            if row["valid"] != "1":
                continue
            position = np.array([float(row[key]) for key in ("x", "y", "z")])
            quaternion = np.array([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
            if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                raise ValueError("VINS camera prior has a nonfinite pose")
            self.poses[int(row["input_index"])] = (position, Rotation.from_quat(quaternion))
        self.world_from_vins = None

    def targets(self, frame_ids, visual_poses):
        """Return aligned metric centers and unit scale, without altering visual poses."""
        if self.world_from_vins is None:
            for frame_id, visual in zip(frame_ids, visual_poses):
                if frame_id not in self.poses:
                    continue
                position, rotation = self.poses[frame_id]
                world_rotation = Rotation.from_quat(visual[3:7]) * rotation.inv()
                world_offset = visual[:3] - world_rotation.apply(position)
                self.world_from_vins = (world_rotation, world_offset)
                break
        targets = np.zeros((len(frame_ids), 4), dtype=np.float32)
        targets[:, 3] = 1.0
        valid = np.zeros(len(frame_ids), dtype=bool)
        if self.world_from_vins is None:
            return targets, valid
        world_rotation, world_offset = self.world_from_vins
        for index, frame_id in enumerate(frame_ids):
            if frame_id not in self.poses:
                continue
            position, _ = self.poses[frame_id]
            targets[index, :3] = world_offset + world_rotation.apply(position)
            valid[index] = True
        return targets, valid
