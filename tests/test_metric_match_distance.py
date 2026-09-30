import pytest

from mast3r_slam.matching import match_distance_threshold


def test_metric_gate_uses_model_to_meter_scale():
    assert match_distance_threshold(0.1, 0.01, 0.067) == pytest.approx(0.1492537313)
    assert match_distance_threshold(0.1, 0.01, 0.106) == pytest.approx(0.0943396226)


def test_missing_metric_scale_keeps_original_gate():
    assert match_distance_threshold(0.1, 0.01, None) == 0.1
    assert match_distance_threshold(0.1, 0.0, 0.067) == 0.1


def test_nonfinite_metric_scale_keeps_original_gate():
    assert match_distance_threshold(0.1, 0.01, float("nan")) == 0.1
