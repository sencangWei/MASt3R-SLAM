"""Optional subpixel observations must not relax metric recovery gates."""
import numpy as np
import pytest

from mast3r_slam.stereo_descriptor_recovery import metric_descriptor_pose


def scene():
    yy, xx = np.indices((80, 120))
    z = (0.25 + 0.0005 * xx + 0.0003 * yy).astype(np.float32)
    a = np.column_stack((xx[4:-4:3, 4:-8:3].ravel(), yy[4:-4:3, 4:-8:3].ravel()))
    b = a + [2, 0]
    depth = np.full_like(z, np.nan)
    depth[b[:, 1], b[:, 0]] = z[a[:, 1], a[:, 0]]
    K = np.array([[200., 0, 60], [0, 200., 40], [0, 0, 1]])
    return a, b, z, depth, z / 0.4, np.full_like(z, 2.), K


def test_integer_opt_in_matches_legacy_path():
    args = scene()
    old = metric_descriptor_pose(*args)
    new = metric_descriptor_pose(*args, target_subpixel_xy=args[1].astype(float))
    np.testing.assert_array_equal(old[0], new[0])
    assert old[1:] == new[1:]


def test_fractional_observation_reaches_pose_solver(monkeypatch):
    import mast3r_slam.stereo_descriptor_recovery as module
    args = scene()
    refined = args[1].astype(float) + [0.35, 0.1]
    original = module.solve_metric_keyframe_pnp
    seen = []

    def capture(xyz, uv, K, **kwargs):
        seen.append(uv.copy())
        return original(xyz, uv, K, **kwargs)

    monkeypatch.setattr(module, 'solve_metric_keyframe_pnp', capture)
    pose, _, report = metric_descriptor_pose(*args, target_subpixel_xy=refined)
    assert pose is not None and report['accepted']
    np.testing.assert_array_equal(seen[0], refined)


def test_depth_is_checked_at_nearest_refined_cell():
    args = list(scene())
    refined = args[1].astype(float) + [0.8, 0]
    _, _, report = metric_descriptor_pose(*args, target_subpixel_xy=refined)
    assert not report['accepted'] and report['reason'] == 'too_few_depth_checks'
    cells = np.rint(refined).astype(int)
    args[3] = np.full_like(args[3], np.nan)
    args[3][cells[:, 1], cells[:, 0]] = args[2][args[0][:, 1], args[0][:, 0]]
    pose, _, report = metric_descriptor_pose(*args, target_subpixel_xy=refined)
    assert pose is not None and report['accepted']


@pytest.mark.parametrize('bad', ['shape', 'nan', 'outside'])
def test_invalid_subpixel_input_fails_explicitly(bad):
    args = scene()
    refined = args[1].astype(float)
    if bad == 'shape':
        refined = refined[:-1]
    elif bad == 'nan':
        refined[0, 0] = np.nan
    else:
        refined[0, 0] = 120
    with pytest.raises(ValueError):
        metric_descriptor_pose(*args, target_subpixel_xy=refined)


def test_native_support_and_confidence_gates_are_unchanged():
    args = list(scene())
    args[0], args[1] = args[0][:99], args[1][:99]
    _, _, report = metric_descriptor_pose(*args, target_subpixel_xy=args[1].astype(float))
    assert report['reason'] == 'too_few_stereo_matches'


def test_subpixel_geometry_rejects_inconsistent_target_depth():
    args = list(scene())
    args[3] += .02
    pose, scale, report = metric_descriptor_pose(
        *args, target_subpixel_xy=args[1].astype(float) + [0.35, 0.1],
    )
    assert pose is None and scale is None
    assert report['depth_consistent_points'] >= 100
    assert report['depth_median_m'] > .006
    assert report['reason'] == 'inconsistent_stereo_geometry'


def test_subpixel_confidence_gate_is_unchanged():
    args = list(scene())
    args[5][:] = 1.01
    _, _, report = metric_descriptor_pose(*args, target_subpixel_xy=args[1].astype(float))
    assert report['reason'] == 'too_few_stereo_matches'
