from threading import RLock
from types import SimpleNamespace

from mast3r_slam.frame import SharedStates


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
