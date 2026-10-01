import pytest
import torch

from mast3r_slam.global_opt import save_graph_snapshot_if_requested


def test_snapshot_is_opt_in_and_preserves_solver_inputs(tmp_path, monkeypatch):
    path = tmp_path / "graph.pt"
    pose = torch.tensor([[1.0, 2.0]])
    monkeypatch.setenv("MAST3R_GRAPH_SNAPSHOT_PATH", str(path))
    assert not save_graph_snapshot_if_requested([1004, 1005], (pose, 7))
    assert not path.exists()

    monkeypatch.setenv("MAST3R_GRAPH_SNAPSHOT_FRAME", "1005")
    assert save_graph_snapshot_if_requested([1004, 1005], (pose, 7))
    pose[0, 0] = 9.0
    saved = torch.load(path, weights_only=True)
    assert saved["frame_ids"] == [1004, 1005]
    assert saved["args"][0].tolist() == [[1.0, 2.0]]
    assert saved["args"][1] == 7
    with pytest.raises(FileExistsError):
        save_graph_snapshot_if_requested([1004, 1005], (pose, 7))


def test_snapshot_target_requires_an_explicit_path(monkeypatch):
    monkeypatch.setenv("MAST3R_GRAPH_SNAPSHOT_FRAME", "1005")
    monkeypatch.delenv("MAST3R_GRAPH_SNAPSHOT_PATH", raising=False)
    with pytest.raises(ValueError, match="MAST3R_GRAPH_SNAPSHOT_PATH"):
        save_graph_snapshot_if_requested([1005], (torch.zeros(1),))
