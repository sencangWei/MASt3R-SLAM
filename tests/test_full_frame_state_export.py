import csv
import math
from types import SimpleNamespace

import lietorch
import pytest
import torch

from mast3r_slam.evaluate import save_full_traj


def sim3(tx, scale):
    return lietorch.Sim3(torch.tensor([[tx, 0., 0., 0., 0., 0., 1., scale]]))


def fixture():
    anchors = [SimpleNamespace(frame_id=0, T_WC=sim3(2., 3.)),
               SimpleNamespace(frame_id=4, T_WC=sim3(8., 2.))]
    tracked = [(0, 0, sim3(0., 1.).data),
               (2, 0, sim3(1., 0.5).data),
               (4, 1, sim3(0., 1.).data)]
    online = [(frame, sim3(float(frame), 0.25).data) for frame, _, _ in tracked]
    return anchors, tracked, online


def test_diagnostics_preserve_trajectory_bytes_and_inputs(tmp_path):
    anchors, tracked, online = fixture()
    originals = [pose.clone() for _, _, pose in tracked]
    times = [0., 0.1, 0.2, 0.3, 0.4]
    save_full_traj(tmp_path, 'baseline.txt', times, anchors, tracked)
    save_full_traj(tmp_path, 'traced.txt', times, anchors, tracked, online_poses=online)
    assert (tmp_path / 'baseline.txt').read_bytes() == (tmp_path / 'traced.txt').read_bytes()
    assert not (tmp_path / 'baseline_sim3_states.csv').exists()
    with (tmp_path / 'traced_sim3_states.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['frame_id']) for row in rows] == [0, 2, 4]
    assert [int(row['anchor_idx']) for row in rows] == [0, 0, 1]
    assert [int(row['anchor_frame_id']) for row in rows] == [0, 0, 4]
    row = rows[1]
    assert float(row['relative_s']) == 0.5
    assert float(row['final_anchor_s']) == 3.
    assert float(row['online_s']) == 0.25
    assert float(row['final_tx']) == 5.  # anchor scale must multiply relative translation
    assert float(row['final_s']) == 1.5
    for original, (_, _, pose) in zip(originals, tracked):
        assert torch.equal(original, pose)


@pytest.mark.parametrize('bad_online', [[(0, sim3(0., 1.).data)],
                                      [(0, sim3(0., 1.).data)] * 3])
def test_rejects_unbound_online_states_before_writing(tmp_path, bad_online):
    anchors, tracked, _ = fixture()
    with pytest.raises(ValueError, match='frame'):
        save_full_traj(tmp_path, 'bad.txt', [0.] * 5, anchors, tracked,
                       online_poses=bad_online)
    assert not (tmp_path / 'bad.txt').exists()


def test_refuses_to_overwrite_diagnostic_sidecar(tmp_path):
    anchors, tracked, online = fixture()
    save_full_traj(tmp_path, 'traced.txt', [0.] * 5, anchors, tracked, online_poses=online)
    original = (tmp_path / 'traced_sim3_states.csv').read_bytes()
    with pytest.raises(FileExistsError):
        save_full_traj(tmp_path, 'traced.txt', [0.] * 5, anchors, tracked,
                       online_poses=online)
    assert (tmp_path / 'traced_sim3_states.csv').read_bytes() == original


def test_rotating_scaled_anchor_is_the_one_used_for_export(tmp_path):
    quat = [0., 0., math.sqrt(.5), math.sqrt(.5)]
    anchor = lietorch.Sim3(torch.tensor([[2., 0., 0., *quat, 3.]]))
    frames = [SimpleNamespace(frame_id=0, T_WC=anchor)]
    relative = sim3(1., .5).data
    online = sim3(4., .25).data
    save_full_traj(tmp_path, 'rot.txt', [0., .1], frames, [(1, 0, relative)],
                   online_poses=[(1, online)])
    with (tmp_path / 'rot_sim3_states.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row['final_tx']) == pytest.approx(2., abs=1e-6)
    assert float(row['final_ty']) == pytest.approx(3., abs=1e-6)
    assert float(row['final_s']) == pytest.approx(1.5)
