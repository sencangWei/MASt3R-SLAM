from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from mast3r_slam.temporal_descriptor_anchors import RecentDescriptorAnchors


def test_retry_older_accepted_anchor_with_composed_noncommuting_gyro():
    anchors = RecentDescriptorAnchors()
    anchors.accept(SimpleNamespace(frame_id=1))
    anchors.accept(SimpleNamespace(frame_id=2))
    priors = Rotation.from_euler("xyz", [[0, 0, 0], [0, 0, 0],
                                        [1, 0, 0], [0, 2, 0]], degrees=True).as_quat()
    seen = []

    def attempt(anchor, prior):
        seen.append((anchor.frame_id, prior))
        return dict(accepted=anchor.frame_id == 1)

    report = anchors.recover(3, priors, attempt)
    assert report["accepted"] and report["anchor_frame_id"] == 1
    assert [frame for frame, _ in seen] == [2, 1]
    expected = Rotation.from_quat(priors[2]) * Rotation.from_quat(priors[3])
    np.testing.assert_allclose(Rotation.from_quat(seen[-1][1]).as_matrix(), expected.as_matrix())


def test_failures_do_not_become_pose_anchors_and_long_tail_not_filled():
    anchors = RecentDescriptorAnchors()
    anchors.accept(SimpleNamespace(frame_id=1))
    priors = np.tile([0, 0, 0, 1], (20, 1))
    assert not anchors.recover(2, priors, lambda *_: dict(accepted=False))["accepted"]
    seen = []
    report = anchors.recover(10, priors, lambda anchor, _: seen.append(anchor))
    assert not report["accepted"] and not seen
    assert [frame.frame_id for frame in anchors.frames] == [1]


def test_bounded_memory_and_future_or_same_frame_not_attempted():
    anchors = RecentDescriptorAnchors()
    for index in range(5):
        anchors.accept(SimpleNamespace(frame_id=index))
    assert [frame.frame_id for frame in anchors.frames] == [2, 3, 4]
    priors = np.tile([0, 0, 0, 1], (10, 1))
    report = anchors.recover(2, priors, lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert not report["accepted"]
