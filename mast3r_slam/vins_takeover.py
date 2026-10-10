"""Onboard camera-relative VINS motion, independent of visual depth and GT."""

import numpy as np
from scipy.spatial.transform import Rotation


def propagate_vins_relative_pose(previous_world, previous_vins, current_vins):
    """Apply a metric SE3 increment without changing the visual world's gauge.

    World translations are metric in this path. Sim3 composition would multiply
    a local translation by the previous scale; composing in meter coordinates
    here avoids introducing the model's current pointmap scale into the motion.
    """
    world = np.asarray(previous_world, dtype=float).reshape(8)
    before = np.asarray(previous_vins, dtype=float).reshape(7)
    after = np.asarray(current_vins, dtype=float).reshape(7)
    for value in (world, before, after):
        if not np.all(np.isfinite(value)) or np.linalg.norm(value[3:7]) < 1e-8:
            raise ValueError("VINS takeover requires finite poses and nonzero quaternions")
    if world[7] <= 0:
        raise ValueError("VINS takeover requires positive visual-world scale")
    r_world = Rotation.from_quat(world[3:7])
    r_before = Rotation.from_quat(before[3:7])
    r_after = Rotation.from_quat(after[3:7])
    local_translation = r_before.inv().apply(after[:3] - before[:3])
    return np.r_[
        world[:3] + r_world.apply(local_translation),
        (r_world * r_before.inv() * r_after).as_quat(),
        world[7],
    ]
