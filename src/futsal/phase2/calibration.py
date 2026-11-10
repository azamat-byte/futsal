"""Source-bound calibration anchors and bounded optical-flow propagation."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import Homography, points2d, project


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Calibration:
    raw: dict
    anchors: dict

    @classmethod
    def load(cls, path: Path, video: Path, info):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1 or raw.get("source_sha256") != fingerprint(video):
            raise ValueError("Calibration belongs to a different video or schema")
        if raw.get("image_size") != [info.width, info.height]:
            raise ValueError("Calibration image dimensions differ from video")
        length, width = raw.get("pitch_size_m", [40, 20])
        if not np.isfinite([length, width]).all() or min(length, width) <= 0:
            raise ValueError("Invalid pitch dimensions")
        anchors = {}
        for anchor in raw.get("anchors", []):
            index = anchor["frame_index"]
            if type(index) is not int or not 0 <= index < info.frame_count or index in anchors:
                raise ValueError("Duplicate or out-of-range anchor frame")
            if not anchor.get("reviewed"):
                raise ValueError("Review the calibration anchor in the point picker before rendering")
            for key, bounds in (("image_points", (info.width, info.height)),
                                ("check_image_points", (info.width, info.height)),
                                ("pitch_points", (length, width)), ("check_pitch_points", (length, width))):
                points = points2d(anchor[key])
                if (points < 0).any() or (points > bounds).any():
                    raise ValueError(f"Out-of-bounds {key}")
            model = Homography.fit(anchor["image_points"], anchor["pitch_points"],
                                   anchor["check_image_points"], anchor["check_pitch_points"],
                                   raw.get("max_error_px", 8))
            anchors[index] = model
        if not anchors:
            raise ValueError("Calibration needs at least one reviewed anchor")
        return cls(raw, anchors)


class CameraCalibration:
    """Track stationary floor texture; never reuse stale H after a cut/failure.

    Flow uses the previous floor mask AND current person exclusion. Forward/backward
    consistency, spatial spread, inlier ratio and age gates limit drift. This is a
    short bridge between anchors, not a substitute for ongoing court detection.
    """

    def __init__(self, calibration: Calibration, fps: float, max_age_s=2.0):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.calibration, self.fps, self.max_age_s = calibration, fps, max_age_s
        self.matrix = None
        self.anchor_frame = None
        self.previous_gray = self.previous_mask = None
        self.last_index = None
        self.reason = "no_anchor"
        self.flow_inlier_ratio = None

    def _floor_mask(self, gray, boxes):
        h, w = gray.shape
        mask = np.zeros_like(gray)
        if self.matrix is None:
            return mask
        length, width = self.calibration.raw.get("pitch_size_m", [40, 20])
        try:
            corners = project(np.linalg.inv(self.matrix), [[0, 0], [length, 0], [length, width], [0, width]])
            if np.abs(corners).max() > 1e6:
                return mask
            cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
        except (ValueError, np.linalg.LinAlgError):
            return mask
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
        mask[:int(h * .12)] = 0  # exclude typical scoreboard/advert overlays
        for x1, y1, x2, y2 in np.asarray(boxes, int).reshape(-1, 4):
            mask[max(0, y1-15):min(h, y2+15), max(0, x1-15):min(w, x2+15)] = 0
        return mask

    def _propagate(self, gray, current_boxes):
        points = cv2.goodFeaturesToTrack(self.previous_gray, 600, 0.002, 7, mask=self.previous_mask)
        if points is None or len(points) < 6:
            raise ValueError("insufficient_floor_texture")
        following, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None)
        if following is None:
            raise ValueError("optical_flow_failed")
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, following, None)
        if back is None or status is None or back_status is None:
            raise ValueError("optical_flow_failed")
        keep = status.ravel().astype(bool) & back_status.ravel().astype(bool)
        keep &= np.linalg.norm(back - points, axis=(1, 2)) < 2.5
        p, q = points.reshape(-1, 2), following.reshape(-1, 2)
        keep &= np.isfinite(q).all(axis=1) & (q[:, 0] >= 0) & (q[:, 0] < gray.shape[1])
        keep &= (q[:, 1] >= 0) & (q[:, 1] < gray.shape[0])
        for x1, y1, x2, y2 in np.asarray(current_boxes).reshape(-1, 4):
            keep &= ~((q[:, 0] >= x1 - 12) & (q[:, 0] <= x2 + 12) &
                      (q[:, 1] >= y1 - 12) & (q[:, 1] <= y2 + 12))
        if keep.sum() < 6:
            raise ValueError("insufficient_consistent_flow")
        motion, inliers = cv2.findHomography(p[keep], q[keep], cv2.RANSAC, 4.0)
        if motion is None or inliers is None or inliers.sum() < 6 or inliers.mean() < 0.50:
            raise ValueError("unsupported_camera_motion")
        selected = q[keep][inliers.ravel().astype(bool)]
        if cv2.contourArea(cv2.convexHull(selected)) < gray.size * 0.005:
            raise ValueError("flow_support_too_local")
        h, w = gray.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        moved = project(motion, corners)
        area = cv2.contourArea(moved.astype(np.float32), oriented=True) / (w * h)
        if not .7 < area < 1.4 or np.linalg.norm(moved - corners, axis=1).max() > .2 * np.hypot(w, h):
            raise ValueError("implausible_camera_step")
        self.matrix = self.matrix @ np.linalg.inv(motion)
        self.flow_inlier_ratio = float(inliers.mean())

    def update(self, frame, index, boxes=(), cut=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        discontinuity = self.last_index is not None and index != self.last_index + 1
        self.flow_inlier_ratio = None
        if cut or discontinuity:
            self.matrix = None
            self.reason = "camera_cut" if cut else "frame_gap"
        anchor = self.calibration.anchors.get(index)
        if anchor is not None:
            self.matrix = anchor.image_to_pitch.copy()
            self.anchor_frame = index
            self.reason = "anchor"
        elif self.matrix is not None:
            if (index - self.anchor_frame) / self.fps > self.max_age_s:
                self.matrix = None
                self.reason = "anchor_expired"
            else:
                try:
                    self._propagate(gray, boxes)
                    self.reason = "optical_flow"
                except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                    self.matrix = None
                    self.reason = str(error) if isinstance(error, ValueError) else "motion_estimation_failed"
        self.previous_mask = self._floor_mask(gray, boxes)
        self.previous_gray, self.last_index = gray, index
        return self.matrix, {"calibration_valid": self.matrix is not None, "calibration_status": self.reason,
                             "anchor_frame": self.anchor_frame, "flow_inlier_ratio": self.flow_inlier_ratio}
