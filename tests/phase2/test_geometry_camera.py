import cv2
import numpy as np
import pytest

from futsal.phase2.calibration import Calibration, CameraCalibration
from futsal.phase2.geometry import Homography, project

COURT = np.array([[0, 0], [40, 0], [40, 20], [0, 20], [6, 10], [34, 10]], float)
CHECK = np.array([[20, 0], [20, 20]], float)
FORWARD = np.array([[10, .7, 25], [.2, 10, 30], [.001, .002, 1]], float)


def calibration():
    return Homography.fit(project(FORWARD, COURT), COURT, project(FORWARD, CHECK), CHECK)


def test_known_projective_transform_round_trip_in_meters():
    fitted = calibration()
    players = np.array([[12.7, 8.9], [31, 19], [0, 0]], float)
    assert np.allclose(fitted.to_pitch(project(FORWARD, players)), players, atol=1e-5)
    assert fitted.validation_error_px < .001
    assert project(fitted.image_to_pitch, []).shape == (0, 2)


def test_pixel_ransac_rejects_outlier():
    points = np.vstack([COURT, [20, 10], [10, 5]])
    pixels = project(FORWARD, points)
    pixels[-1] += [150, 90]
    model = Homography.fit(pixels, points, project(FORWARD, CHECK), CHECK, max_error_px=3)
    assert model.inliers == 7
    assert model.validation_error_px < .01


def test_independent_validation_rejects_bad_geometry():
    checks = project(FORWARD, CHECK) + [0, 20]
    with pytest.raises(ValueError, match="Held-out"):
        Homography.fit(project(FORWARD, COURT), COURT, checks, CHECK)
    with pytest.raises(ValueError, match="independent"):
        Homography.fit(project(FORWARD, COURT), COURT, project(FORWARD, COURT[:2]), COURT[:2])


def test_degenerate_and_horizon_points():
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    with pytest.raises(ValueError, match="collinear"):
        Homography.fit(line, line, CHECK, CHECK)
    with pytest.raises(ValueError, match="finite"):
        Homography.fit(np.full((4, 2), np.nan), COURT[:4], CHECK, CHECK)
    with pytest.raises(ValueError, match="horizon"):
        project(np.array([[1, 0, 0], [0, 1, 0], [1, 0, -2]]), [[2, 5]])


def texture():
    rng = np.random.default_rng(42)
    image = np.full((300, 480, 3), 90, np.uint8)
    for _ in range(400):
        x, y = rng.integers([30, 45], [420, 220])
        cv2.circle(image, (int(x), int(y)), 2, (230, 230, 230), -1)
    return image


def camera(max_age=2):
    return CameraCalibration(Calibration({"pitch_size_m": [40, 20]}, {0: calibration()}), 30, max_age)


def test_camera_motion_composition_keeps_stationary_player_in_same_pitch_location():
    instance = camera()
    image = texture()
    instance.update(image, 0)
    motion = np.array([[1, 0, 3], [0, 1, 2], [0, 0, 1]], float)
    following = cv2.warpPerspective(image, motion, (480, 300))
    matrix, status = instance.update(following, 1)
    assert status["calibration_status"] == "optical_flow"
    physical = [[20, 10]]
    moved_pixels = project(motion @ FORWARD, physical)
    assert np.allclose(project(matrix, moved_pixels), physical, atol=.05)


def test_cut_and_frame_gap_invalidate_without_reusing_matrix():
    instance = camera()
    image = texture()
    instance.update(image, 0)
    matrix, status = instance.update(image, 1, cut=True)
    assert matrix is None and status["calibration_status"] == "camera_cut"
    assert instance.update(image, 2)[0] is None
    instance = camera()
    instance.update(image, 0)
    assert instance.update(image, 2)[1]["calibration_status"] == "frame_gap"


def test_age_and_texture_failure_gates():
    instance = camera(max_age=.01)
    image = texture()
    instance.update(image, 0)
    assert instance.update(image, 1)[1]["calibration_status"] == "anchor_expired"
    instance = camera()
    blank = np.zeros_like(image)
    instance.update(blank, 0)
    matrix, state = instance.update(blank, 1)
    assert matrix is None and state["calibration_status"] == "insufficient_floor_texture"
