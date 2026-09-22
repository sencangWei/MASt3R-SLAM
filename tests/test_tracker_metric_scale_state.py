from mast3r_slam.tracker import FrameTracker


def test_reset_metric_scale_state_drops_previous_keyframe_scale():
    tracker = FrameTracker.__new__(FrameTracker)
    tracker.metric_pointmap_scale = 0.42

    tracker.reset_metric_scale_state()

    assert tracker.metric_pointmap_scale is None
