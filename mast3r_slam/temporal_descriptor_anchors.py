"""Bounded accepted-frame anchors for opt-in stereo descriptor recovery."""
from collections import deque

from scipy.spatial.transform import Rotation


class RecentDescriptorAnchors:
    def __init__(self):
        self.frames = deque(maxlen=3)

    def accept(self, frame):
        if self.frames and frame.frame_id <= self.frames[-1].frame_id:
            raise ValueError("accepted descriptor anchors must be ordered")
        self.frames.append(frame)

    def recover(self, current_id, rotation_priors, attempt):
        """Retry only real accepted poses, with the full anchor-to-frame gyro.

        ``attempt`` retains the existing descriptor/PnP/depth/motion gates.
        This helper never supplies a pose or fills an unobserved frame.
        """
        attempts = []
        for anchor in reversed(self.frames):
            gap = current_id - anchor.frame_id
            if not 1 <= gap <= 8:
                continue
            rotation = Rotation.identity()
            for quaternion in rotation_priors[anchor.frame_id + 1:current_id + 1]:
                rotation = rotation * Rotation.from_quat(quaternion)
            report = attempt(anchor, rotation.as_quat())
            attempts.append(dict(report, anchor_frame_id=anchor.frame_id, frame_gap=gap))
            if report.get("accepted") is True:
                return dict(attempts[-1], temporal_anchor_attempts=attempts)
        return dict(accepted=False, reason="no_accepted_temporal_anchor",
                    temporal_anchor_attempts=attempts)
