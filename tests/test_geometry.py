import numpy as np
import pytest

from futsal.geometry import CalibrationSpec, fit_calibration, project


def specification(**kwargs):
    values = dict(
        time_s=0,
        court_points=[[0, 0], [40, 0], [40, 20], [0, 20]],
        image_points=[[10, 20], [410, 20], [410, 220], [10, 220]],
        validation_court_points=[[20, 10]],
        validation_image_points=[[210, 120]],
        reviewed=True,
        scale_verified=True,
    )
    return CalibrationSpec(**(values | kwargs))


def test_metric_mapping():
    matrix, info = fit_calibration(specification())
    assert info["valid"]
    assert np.allclose(project(matrix, [[210, 120]]), [[20, 10]], atol=1e-5)


def test_without_independent_check_is_invalid():
    _, info = fit_calibration(specification(validation_court_points=[], validation_image_points=[]))
    assert not info["valid"]


def test_wrong_check_is_invalid():
    _, info = fit_calibration(specification(validation_image_points=[[500, 500]]))
    assert not info["valid"]


def test_reused_fit_point_is_not_independent_validation():
    _, info = fit_calibration(
        specification(validation_court_points=[[0, 0]], validation_image_points=[[10, 20]])
    )
    assert not info["valid"]


def test_collinear_rejected():
    with pytest.raises(ValueError, match="Collinear"):
        fit_calibration(specification(court_points=[[0, 0], [1, 0], [2, 0], [3, 0]]))


def test_camera_motion_composition_stationary_world():
    matrix, _ = fit_calibration(specification())
    motion = np.array([[1.2, 0, 70], [0, 1.2, -20], [0, 0, 1]])
    previous = [[210, 120]]
    current = project(motion, previous)
    propagated = matrix @ np.linalg.inv(motion)
    assert np.allclose(project(propagated, current), project(matrix, previous))


def test_projection_infinity_rejected():
    with pytest.raises(ValueError):
        project([[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[1, 2]])
