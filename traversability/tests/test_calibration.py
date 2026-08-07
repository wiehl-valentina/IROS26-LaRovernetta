import numpy as np

from rover_traversability.calibration import (
    horizontal_fov_deg,
    load_camera_K,
    load_camera_dist,
    load_T_base_camera,
)


def test_intrinsics_shape_and_values():
    K = load_camera_K()
    assert K.shape == (3, 3)
    assert K.dtype == np.float64
    assert K[0, 0] > 0 and K[1, 1] > 0          # focal lengths
    assert K[0, 2] == 512.0 and K[1, 2] == 288.0  # principal point of 1024x576


def test_distortion_shape():
    dist = load_camera_dist()
    assert dist.shape == (5,)
    assert np.allclose(dist, 0.0)  # Mini+ model treats distortion as negligible


def test_extrinsics_shape_and_plausibility():
    T = load_T_base_camera()
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])
    # Camera ~14 cm up, ~11 cm forward of base origin.
    assert 0.05 < T[2, 3] < 0.5
    assert 0.0 < T[0, 3] < 0.5


def test_hfov_matches_policy_default():
    from rover_traversability.policy import DEFAULT_HFOV_DEG

    assert abs(horizontal_fov_deg(1024) - DEFAULT_HFOV_DEG) < 0.5
