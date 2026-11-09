"""Fit court-to-image (pixel RANSAC threshold), then invert for meter projection."""

from dataclasses import dataclass

import cv2
import numpy as np


def points2d(value, minimum=0):
    points = np.asarray(value, np.float64)
    if points.size == 0 and minimum == 0:
        return np.empty((0, 2), np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum or not np.isfinite(points).all():
        raise ValueError(f"Require at least {minimum} finite 2D points")
    return points


def project(matrix, points):
    points = points2d(points)
    if not len(points):
        return points.copy()
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid homography")
    denominator = np.c_[points, np.ones(len(points))] @ matrix[2]
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("Projection reaches the homography horizon")
    result = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite projection")
    return result


@dataclass
class Homography:
    image_to_pitch: np.ndarray
    pitch_to_image: np.ndarray
    inliers: int
    fit_error_px: float
    validation_error_px: float

    @classmethod
    def fit(cls, image_points, pitch_points, check_image, check_pitch, max_error_px=8.0):
        image, pitch = points2d(image_points, 4), points2d(pitch_points, 4)
        checks, targets = points2d(check_image, 2), points2d(check_pitch, 2)
        if image.shape != pitch.shape or checks.shape != targets.shape:
            raise ValueError("Image and pitch correspondence counts differ")
        if not np.isfinite(max_error_px) or max_error_px <= 0:
            raise ValueError("max_error_px must be positive")
        for points in (image, pitch):
            if len(np.unique(points, axis=0)) != len(points) or np.linalg.matrix_rank(points - points.mean(0)) < 2:
                raise ValueError("Duplicate or collinear fitting points")
        if len(np.unique(checks, axis=0)) != len(checks) or len(np.unique(targets, axis=0)) != len(targets):
            raise ValueError("Validation points must be distinct")
        if (np.min(np.linalg.norm(checks[:, None] - image[None], axis=2)) < 2 or
                np.min(np.linalg.norm(targets[:, None] - pitch[None], axis=2)) < .05):
            raise ValueError("Validation landmarks must be independent of fitting landmarks")
        # Destination is image space, so the RANSAC threshold is in PIXELS, not meters.
        forward, mask = cv2.findHomography(pitch, image, cv2.RANSAC, max_error_px)
        if forward is None or mask is None or mask.sum() < 4 or mask.mean() < .6:
            raise ValueError("Homography lacks inlier support")
        try:
            inverse = np.linalg.inv(forward)
        except np.linalg.LinAlgError:
            raise ValueError("Singular homography") from None
        if not np.isfinite(inverse).all() or np.linalg.cond(forward) > 1e12:
            raise ValueError("Ill-conditioned homography")
        fit_error = np.linalg.norm(project(forward, pitch) - image, axis=1)
        validation_error = float(np.max(np.linalg.norm(project(forward, targets) - checks, axis=1)))
        if validation_error > max_error_px:
            raise ValueError(f"Held-out landmark error {validation_error:.1f}px exceeds {max_error_px:.1f}px")
        return cls(inverse, forward, int(mask.sum()),
                   float(np.median(fit_error[mask.ravel().astype(bool)])), validation_error)

    def to_pitch(self, points):
        return project(self.image_to_pitch, points)
