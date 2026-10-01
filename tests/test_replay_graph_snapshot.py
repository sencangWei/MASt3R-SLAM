import pytest
import torch

from scripts.replay_graph_snapshot import checked_pair, swapped_keyframe_geometry


def snapshot():
    args = [0] * 19
    args[3] = torch.eye(3)
    return {"frame_ids": [1004, 1005], "args": tuple(args)}


def test_crossed_graph_replay_requires_same_keyframes_and_solver_settings():
    control = snapshot()
    spatial = snapshot()
    checked_pair(control, spatial)
    spatial["frame_ids"] = [1003, 1005]
    with pytest.raises(ValueError, match="keyframe identities"):
        checked_pair(control, spatial)
    spatial["frame_ids"] = control["frame_ids"]
    spatial["args"] = spatial["args"][:17] + (3, spatial["args"][18])
    with pytest.raises(ValueError, match="solver setting 17"):
        checked_pair(control, spatial)


def test_single_keyframe_swap_changes_only_pointmap_and_confidence():
    base = snapshot()
    donor = snapshot()
    base_args = list(base["args"])
    donor_args = list(donor["args"])
    base_args[1] = torch.zeros(2, 2, 3)
    base_args[2] = torch.zeros(2, 2, 1)
    donor_args[1] = torch.ones(2, 2, 3)
    donor_args[2] = torch.ones(2, 2, 1)
    base["args"] = tuple(base_args)
    donor["args"] = tuple(donor_args)
    swapped = swapped_keyframe_geometry(base, donor, 1004)
    assert swapped["args"][1][0].sum() == 6
    assert swapped["args"][1][1].sum() == 0
    assert swapped["args"][2][0].sum() == 2
    assert swapped["args"][2][1].sum() == 0
    assert base["args"][1].sum() == 0
