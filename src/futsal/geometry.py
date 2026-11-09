import cv2
import numpy as np
from pydantic import BaseModel, Field


class CalibrationSpec(BaseModel):
    time_s: float = Field(ge=0)
    image_points: list[list[float]]
    court_points: list[list[float]]
    validation_image_points: list[list[float]] = []
    validation_court_points: list[list[float]] = []
    length: float = Field(40, gt=0)
    width: float = Field(20, gt=0)
    scale_verified: bool = False
    reviewed: bool = False
    max_error_px: float = Field(8, gt=0)
    method: str = "manual"
    landmark_model_accepted: bool = False
    source_sha256: str | None = None


def project(matrix, points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if not len(points):
        return points
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("Projection at infinity")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def fit_calibration(spec: CalibrationSpec):
    image = np.asarray(spec.image_points, np.float64)
    court = np.asarray(spec.court_points, np.float64)
    if image.shape != court.shape or image.ndim != 2 or image.shape[1] != 2 or len(image) < 4:
        raise ValueError("At least four matching 2D correspondences are required")
    if not np.isfinite(image).all() or not np.isfinite(court).all():
        raise ValueError("Nonfinite calibration points")
    for points in (image, court):
        if np.linalg.matrix_rank(points - points.mean(0)) < 2:
            raise ValueError("Collinear calibration points")
        hull = cv2.convexHull(points.astype(np.float32))
        if cv2.contourArea(hull) < 1e-3:
            raise ValueError("Degenerate point distribution")
    # Fit court -> image, so RANSAC threshold has PIXEL units.
    forward, inliers = cv2.findHomography(court, image, cv2.RANSAC, spec.max_error_px)
    if forward is None or inliers.sum() < 4 or not np.isfinite(forward).all():
        raise ValueError("No supported homography")
    inverse = np.linalg.inv(forward)
    residual = np.linalg.norm(project(forward, court) - image, axis=1)
    error = None
    independent = False
    if spec.validation_image_points:
        vi = np.asarray(spec.validation_image_points, np.float64)
        vc = np.asarray(spec.validation_court_points, np.float64)
        if vi.shape != vc.shape or vi.ndim != 2 or vi.shape[1] != 2:
            raise ValueError("Validation correspondence shape mismatch")
        if not np.isfinite(vi).all() or not np.isfinite(vc).all():
            raise ValueError("Nonfinite validation points")
        error = float(np.max(np.linalg.norm(project(forward, vc) - vi, axis=1)))
        independent = bool(
            np.all(np.min(np.linalg.norm(vi[:, None, :] - image[None, :, :], axis=2), axis=1) > 2)
            and np.all(np.min(np.linalg.norm(vc[:, None, :] - court[None, :, :], axis=2), axis=1) > .05)
        )
    supported = spec.reviewed or (spec.method == "trained_landmarks" and spec.landmark_model_accepted)
    valid = bool(supported and independent and error is not None and error <= spec.max_error_px)
    return inverse, {
        "valid": valid,
        "error_px": error,
        "anchor_validation_error_px": error,
        "inliers": int(inliers.sum()),
        "fit_median_px": float(np.median(residual)),
        "scale_verified": spec.scale_verified,
        "reason": "accepted" if valid else "needs_review_and_independent_validation",
        "direction": "image_to_court",
        "units": "m" if spec.scale_verified else "assumed_m",
    }


class DynamicCalibration:
    def __init__(self, anchors, max_age=2):
        self.anchors = anchors
        self.max_age = max_age
        self.matrix = None
        self.info = {"valid": False, "reason": "no_calibration", "error_px": None, "scale_verified": False}
        self.previous = None
        self.anchor_time = None
        self.used = set()

    def update(self, frame, time, detections, cut=False, direct_anchor=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if cut:
            self.matrix = None
            self.anchor_time = None
            self.info = {**self.info, "valid": False, "reason": "camera_cut"}
        matching = [
            (i, a) for i, a in enumerate(self.anchors) if i not in self.used and abs(a.time_s - time) <= 0.15
        ]
        if matching:
            i, spec = min(matching, key=lambda a: abs(a[1].time_s - time))
            self.matrix, self.info = fit_calibration(spec)
            self.anchor_time = time
            self.used.add(i)
        elif direct_anchor is not None:
            self.matrix, self.info = fit_calibration(direct_anchor)
            self.anchor_time = time
        elif self.matrix is not None and self.info["valid"] and self.previous is not None:
            if time - self.anchor_time > self.max_age:
                self.info = {**self.info, "valid": False, "reason": "anchor_expired"}
            else:
                mask = np.ones_like(gray, np.uint8) * 255
                mask[: int(h * 0.12)] = 0
                for box in detections:
                    x1, y1, x2, y2 = np.asarray(box, int)
                    mask[max(0, y1 - 10) : min(h, y2 + 10), max(0, x1 - 10) : min(w, x2 + 10)] = 0
                points = cv2.goodFeaturesToTrack(self.previous, 400, 0.02, 12, mask=mask)
                good = False
                if points is not None and len(points) >= 12:
                    nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                    keep = status.ravel().astype(bool) if status is not None else np.zeros(len(points), bool)
                    if keep.sum() >= 12:
                        motion, inliers = cv2.findHomography(points[keep], nxt[keep], cv2.RANSAC, 2)
                        if motion is not None and inliers.mean() > 0.7:
                            try:
                                propagated = self.matrix @ np.linalg.inv(motion)
                                corners = project(motion, [[0, 0], [w, 0], [w, h], [0, h]])
                                good = bool(
                                    np.isfinite(propagated).all() and np.max(np.abs(corners)) < max(w, h) * 3
                                )
                                if good:
                                    self.matrix = propagated
                                    self.info = {**self.info, "error_px": None,
                                                 "reason": "optical_flow_propagated",
                                                 "anchor_age_s": time - self.anchor_time,
                                                 "motion_inlier_ratio": float(inliers.mean())}
                            except (ValueError, np.linalg.LinAlgError):
                                pass
                if not good:
                    self.info = {**self.info, "valid": False, "reason": "motion_not_supported"}
        self.previous = gray
        return self.matrix, self.info
