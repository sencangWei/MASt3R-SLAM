from threading import RLock
from types import SimpleNamespace

import pytest

from main import raise_if_backend_exited, relocalization
from mast3r_slam.config import config
from mast3r_slam.frame import SharedStates
from mast3r_slam.global_opt import local_asymmetric_reloc_valid


def test_relocalization_waits_for_an_explicit_request():
    # No GPU/shared-memory allocation is needed to test the request handshake.
    states = object.__new__(SharedStates)
    states.lock = RLock()
    states.reloc_sem = SimpleNamespace(value=0)

    assert not states.has_pending_reloc()
    states.queue_reloc()
    assert states.has_pending_reloc()
    states.dequeue_reloc()
    assert not states.has_pending_reloc()


def test_backend_failure_is_reported_instead_of_waiting_forever():
    raise_if_backend_exited(SimpleNamespace(exitcode=None))
    with pytest.raises(RuntimeError, match="exit code 1"):
        raise_if_backend_exited(SimpleNamespace(exitcode=1))


@pytest.mark.parametrize("retry,expected,anchor", [(False, False, None), (True, True, 2)])
def test_relocalization_retries_valid_candidate_without_changing_default(
    monkeypatch, retry, expected, anchor
):
    class Keyframes:
        def __init__(self):
            self.lock = RLock()
            self.T_WC = [SimpleNamespace(clone=lambda i=i: i) for i in range(3)]

        def __len__(self):
            return len(self.T_WC)

        def append(self, frame):
            self.T_WC.append(None)

        def pop_last(self):
            self.T_WC.pop()

    class Graph:
        def __init__(self):
            self.calls = []
            self.solved = False

        def add_factors(self, ii, jj, min_match_frac, is_reloc):
            self.calls.append((list(ii), list(jj), min_match_frac, is_reloc))
            return list(jj) == [2]

        def solve_GN_rays(self):
            self.solved = True

    monkeypatch.setitem(config, "retrieval", {"k": 3, "min_thresh": 0.005})
    monkeypatch.setitem(config, "reloc", {"min_match_frac": 0.3, "strict": True})
    monkeypatch.setitem(config, "use_calib", False)
    if retry:
        monkeypatch.setenv("MAST3R_RELOC_RETRY_CANDIDATES", "1")
    else:
        monkeypatch.delenv("MAST3R_RELOC_RETRY_CANDIDATES", raising=False)
    keyframes = Keyframes()
    graph = Graph()
    retrieval = SimpleNamespace(update=lambda *args, **kwargs: [0, 1, 2])

    assert relocalization(object(), keyframes, graph, retrieval) is expected
    assert (keyframes.T_WC[-1] if expected else None) == anchor
    assert len(keyframes) == (4 if expected else 3)
    assert graph.solved is expected
    assert len(graph.calls) == (4 if retry else 1)


def test_local_asymmetric_relocalization_preserves_distant_loop_guard():
    assert local_asymmetric_reloc_valid(0.117, 0.396, 19, 0.1, 0.3)
    assert not local_asymmetric_reloc_valid(0.117, 0.396, 31, 0.1, 0.3)
    assert not local_asymmetric_reloc_valid(0.09, 0.396, 19, 0.1, 0.3)
    assert not local_asymmetric_reloc_valid(0.117, 0.29, 19, 0.1, 0.3)
