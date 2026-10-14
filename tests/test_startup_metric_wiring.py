import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mast3r_slam.dataloader import MonocularDataset
from mast3r_slam.frame import Mode


@pytest.fixture(autouse=True)
def initialized_dataset_config(monkeypatch):
    from mast3r_slam.config import config
    monkeypatch.setitem(config, "use_calib", True)


def test_dataset_uses_raw_intrinsics_and_incremental_inverse_imu(monkeypatch):
    import cv2
    dataset = MonocularDataset()
    dataset.rgb_files = ["0.png", "1.png"]
    increment = Rotation.from_euler("z", 5, degrees=True)
    dataset.rotation_priors = [Rotation.identity().as_quat(), increment.as_quat()]
    camera = np.array([[100., 0., 4.], [0., 100., 3.], [0., 0., 1.]])
    dataset.camera_intrinsics = SimpleNamespace(
        H=8, W=10, K_orig=camera, K_frame=camera*.4, distortion=np.zeros(4))
    dataset.stereo_depth_provider = SimpleNamespace(get_depth=lambda path, shape: np.full(shape, .3))
    image = np.zeros((8, 10), np.uint8)
    monkeypatch.setattr(cv2, "imread", lambda *args: image)
    a, b, depth, K, imu = dataset.get_startup_metric_observation(0, 1)
    assert a is image and b is image
    assert depth.shape == image.shape and np.allclose(K, camera)
    assert np.allclose(imu, increment.inv().as_matrix())


@pytest.mark.parametrize("failure", ["distortion", "no_imu", "no_stereo", "skip"])
def test_dataset_does_not_supply_unsupported_pixel_gauge(failure, monkeypatch):
    import cv2
    dataset = MonocularDataset()
    dataset.rgb_files = ["0.png", "1.png", "2.png"]
    dataset.rotation_priors = [Rotation.identity().as_quat()] * 3
    dataset.camera_intrinsics = SimpleNamespace(
        H=8, W=10, K_orig=np.eye(3), distortion=np.zeros(4))
    dataset.stereo_depth_provider = object()
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("must reject before reading"))
    current = 1
    if failure == "distortion":
        dataset.camera_intrinsics.distortion[0] = .1
    elif failure == "no_imu":
        dataset.rotation_priors = None
    elif failure == "no_stereo":
        dataset.stereo_depth_provider = None
    else:
        current = 2
    assert dataset.get_startup_metric_observation(0, current) is None


@pytest.mark.parametrize("tracked,mode,accepted,expected,calls", [
    (False, Mode.RELOC, True, True, 1),
    (False, Mode.RELOC, False, False, 1),
    (True, Mode.TRACKING, True, True, 0),
    (False, Mode.TRACKING, True, False, 0),
])
def test_main_output_recovery_never_promotes_relocalization_state(
        tracked, mode, accepted, expected, calls):
    # Execute the real loop boundary without loading a model/backend.
    tree = ast.parse((Path(__file__).parents[1]/"main.py").read_text())
    boundary = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "not tracked and states.get_mode() == Mode.RELOC")
    measured = []
    tracker = SimpleNamespace(recover_startup_metric_pose=lambda frame: measured.append(frame) or accepted)
    states = SimpleNamespace(get_mode=lambda: mode,
                             set_frame=lambda frame: pytest.fail("must not overwrite pending RELOC payload"))
    scope = dict(tracked=tracked, tracker=tracker, states=states, frame=object(), Mode=Mode)
    exec(compile(ast.Module(body=[boundary], type_ignores=[]), "real-startup-boundary", "exec"), scope)
    assert scope["tracked"] is expected and len(measured) == calls
    assert states.get_mode() == mode
