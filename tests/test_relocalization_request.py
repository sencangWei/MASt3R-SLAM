from threading import RLock
from types import SimpleNamespace

import pytest

from main import raise_if_backend_exited
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


def test_backend_failure_is_reported_instead_of_waiting_forever():
    raise_if_backend_exited(SimpleNamespace(exitcode=None))
    with pytest.raises(RuntimeError, match="exit code 1"):
        raise_if_backend_exited(SimpleNamespace(exitcode=1))
