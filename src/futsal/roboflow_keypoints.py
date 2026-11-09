"""Roboflow Keypoint Model Adapter for s-workspace-bimfr/keypoints-ovj9c-11-rfdetr-keypoint-two-stage-t1."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from futsal.match.pose import match_boxes
from futsal.phase1.config import DetectorConfig
from futsal.phase1.detection import RoboflowDetector
from futsal.phase1.pipeline import write_json
from futsal.phase1.video import VideoReader
from futsal.phase2.calibration import fingerprint
from futsal.phase2.geometry import Homography

LOG = logging.getLogger(__name__)
DEFAULT_KEYPOINT_MODEL_ID = "s-workspace-bimfr/keypoints-ovj9c-11-rfdetr-keypoint-two-stage-t1"
DEFAULT_DETECTION_MODEL_ID = "s-workspace-bimfr/new_futsal-2-rfdetr-medium-t1"

# 26-landmark futsal pitch schema (from keypoints-ovj9c dataset) + 33-point active ID mapping
ACTIVE_IDS_33 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 23, 25, 26, 27, 28, 29, 31)
COURT_26_POINTS: list[tuple[str, tuple[float, float]]] = [
    ("far_left_corner", (0.0, 0.0)),
    ("left_far_goal_post_floor", (0.0, 8.5)),
    ("left_near_goal_post_floor", (0.0, 11.5)),
    ("near_left_corner", (0.0, 20.0)),
    ("halfway_far_touchline", (20.0, 0.0)),
    ("far_right_corner", (40.0, 0.0)),
    ("right_far_goal_post_floor", (40.0, 8.5)),
    ("right_near_goal_post_floor", (40.0, 11.5)),
    ("near_right_corner", (40.0, 20.0)),
    ("halfway_near_touchline", (20.0, 20.0)),
    ("center_circle_halfway_near", (20.0, 13.0)),
    ("center_circle_right_apex", (23.0, 10.0)),
    ("center_circle_halfway_far", (20.0, 7.0)),
    ("center_circle_left_apex", (17.0, 10.0)),
    ("left_penalty_arc_far_45deg", (4.24, 4.18)),
    ("left_first_penalty_mark", (6.0, 10.0)),
    ("left_penalty_arc_near_45deg", (4.24, 15.82)),
    ("right_first_penalty_mark", (34.0, 10.0)),
    ("right_penalty_arc_near_45deg", (35.76, 15.82)),
    ("right_penalty_arc_far_45deg", (35.76, 4.18)),
    ("left_penalty_arc_far_goal_line", (0.0, 2.42)),
    ("left_penalty_arc_near_goal_line", (0.0, 17.58)),
    ("right_penalty_arc_far_goal_line", (40.0, 2.42)),
    ("right_penalty_arc_near_goal_line", (40.0, 17.58)),
    ("right_second_penalty_mark", (30.0, 10.0)),
    ("left_second_penalty_mark", (10.0, 10.0)),
]


def extract_keypoint_arrays(prediction: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (N, 2) pixel xy and (N,) confidence arrays from a Roboflow keypoint prediction dict."""
    raw_kps = prediction.get("keypoints", [])
    if not isinstance(raw_kps, list) or not raw_kps:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    xy = np.array([[float(kp["x"]), float(kp["y"])] for kp in raw_kps], dtype=np.float64)
    conf = np.array([float(kp.get("confidence", 1.0)) for kp in raw_kps], dtype=np.float64)
    return xy, conf


class RoboflowKeypointModel:
    """Adapter for s-workspace-bimfr/keypoints-ovj9c-11-rfdetr-keypoint-two-stage-t1.

    Supports both:
    - Phase 2 automatic pitch homography estimation (26-point or 33-point court schema)
    - Phase 4 player foot/ankle keypoint extraction (COCO 17-point or person keypoint instances)
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_KEYPOINT_MODEL_ID,
        api_url: str = "https://serverless.roboflow.com",
        confidence: float = 0.10,
        keypoint_threshold: float = 0.35,
    ) -> None:
        self.keypoint_threshold = keypoint_threshold
        self.client = RoboflowDetector(
            DetectorConfig(model_id=model_id, api_url=api_url, confidence=confidence),
            api_key=api_key,
        )

    def infer(self, frame: np.ndarray) -> dict:
        return self.client.infer(frame)

    def detect_court_correspondences(
        self, frame: np.ndarray
    ) -> list[tuple[str, list[float], list[float], float]]:
        """Return list of (landmark_name, [px_x, px_y], [pitch_x_m, pitch_y_m], confidence)."""
        payload = self.infer(frame)
        predictions = payload.get("predictions", [])
        if not predictions:
            return []

        best_matches: list[tuple[str, list[float], list[float], float]] = []
        for pred in predictions:
            xy, conf = extract_keypoint_arrays(pred)
            if len(xy) == 26:
                mapping = [(i, COURT_26_POINTS[i][0], COURT_26_POINTS[i][1]) for i in range(26)]
            elif len(xy) == 33:
                mapping = [
                    (ACTIVE_IDS_33[i], COURT_26_POINTS[i][0], COURT_26_POINTS[i][1])
                    for i in range(len(COURT_26_POINTS))
                ]
            else:
                continue

            matches = []
            for idx, name, (mx, my) in mapping:
                if (
                    idx < len(xy)
                    and np.isfinite(conf[idx])
                    and conf[idx] >= self.keypoint_threshold
                    and np.isfinite(xy[idx]).all()
                    and not np.allclose(xy[idx], 0.0)
                ):
                    matches.append((name, [float(xy[idx, 0]), float(xy[idx, 1])], [float(mx), float(my)], float(conf[idx])))
            if len(matches) > len(best_matches):
                best_matches = matches
        return best_matches

    def fit_frame_anchor(self, frame: np.ndarray, frame_index: int, max_error_px: float = 8.0) -> dict | None:
        """Fit a validated Phase 2 Homography anchor from detected Roboflow pitch keypoints."""
        matches = self.detect_court_correspondences(frame)
        if len(matches) < 6:
            return None
        # Sort by keypoint confidence descending
        matches.sort(key=lambda m: -m[3])

        # Use RANSAC over all detected correspondences to find consistent inliers
        img_pts = np.array([m[1] for m in matches], dtype=np.float64)
        pitch_pts = np.array([m[2] for m in matches], dtype=np.float64)
        _, mask = cv2.findHomography(pitch_pts, img_pts, cv2.RANSAC, max_error_px)
        if mask is None or int(mask.sum()) < 6:
            return None

        inlier_indices = [int(i) for i in np.flatnonzero(mask.ravel())]
        inliers = [matches[i] for i in inlier_indices]
        # Split into fit points (all but 2) and independent check points (last 2)
        fit_set = inliers[:-2]
        check_set = inliers[-2:]
        try:
            Homography.fit(
                [m[1] for m in fit_set],
                [m[2] for m in fit_set],
                [m[1] for m in check_set],
                [m[2] for m in check_set],
                max_error_px=max_error_px,
            )
        except ValueError:
            return None

        return {
            "frame_index": int(frame_index),
            "image_points": [m[1] for m in fit_set],
            "pitch_points": [m[2] for m in fit_set],
            "check_image_points": [m[1] for m in check_set],
            "check_pitch_points": [m[2] for m in check_set],
            "landmark_names": [m[0] for m in fit_set + check_set],
            "reviewed": True,
        }

    def auto_calibrate_video(
        self,
        video: Path,
        output: Path,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 30,
        max_error_px: float = 8.0,
    ) -> Path:
        """Scan video frames using the Roboflow keypoint model and write a Phase 2 calibration JSON."""
        with VideoReader(video) as reader:
            info = reader.info
            assert info is not None
            anchors = []
            for index, frame in reader.frames(start_frame, limit=max_frames, stride=stride):
                anchor = self.fit_frame_anchor(frame, index, max_error_px=max_error_px)
                if anchor is not None:
                    anchors.append(anchor)
                    LOG.info("Auto-calibrated anchor at frame %d (%d landmarks)", index, len(anchor["landmark_names"]))

        if not anchors:
            raise ValueError("No valid court keypoint anchors found; lower threshold or use interactive picker")

        payload = {
            "schema_version": 1,
            "source_sha256": fingerprint(video),
            "image_size": [info.width, info.height],
            "pitch_size_m": [40, 20],
            "scale_verified": False,
            "max_error_px": max_error_px,
            "model_id": self.client.config.model_id,
            "anchors": anchors,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, payload)
        return output

    def estimate(self, frame: np.ndarray, tracks: sv.Detections) -> dict[int, list[list[float]]]:
        """PoseEstimator interface for Phase 4: extract foot/ankle keypoints per tracked player."""
        if not len(tracks):
            return {}
        payload = self.infer(frame)
        predictions = payload.get("predictions", [])
        if not predictions:
            return {}

        pose_boxes = []
        pose_kps = []
        for pred in predictions:
            xy, conf = extract_keypoint_arrays(pred)
            if len(xy) == 0:
                continue
            x, y, w, h = (float(pred.get(k, 0.0)) for k in ("x", "y", "width", "height"))
            if w <= 0 or h <= 0:
                continue
            pose_boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            pose_kps.append((xy, conf))

        if not pose_boxes:
            return {}

        feet: dict[int, list[list[float]]] = {}
        fh, fw = frame.shape[:2]
        for person_idx, pose_idx in match_boxes(tracks.xyxy, np.asarray(pose_boxes, dtype=np.float64)):
            xy, conf = pose_kps[pose_idx]
            # If 17-keypoint COCO human skeleton, use ankles (15, 16); otherwise use bottom-most valid keypoints
            candidate_indices = (15, 16) if len(xy) >= 17 else tuple(np.argsort(-xy[:, 1])[:2])
            pts = [
                [float(xy[k, 0]), float(xy[k, 1])]
                for k in candidate_indices
                if k < len(xy)
                and conf[k] >= self.keypoint_threshold
                and 0 < xy[k, 0] < fw
                and 0 < xy[k, 1] < fh
            ]
            if pts:
                feet[int(tracks.tracker_id[person_idx])] = pts

        # Fallback: if the model returned a single court-level skeleton rather than per-person boxes,
        # provide bottom-center foot anchors for tracked players so Phase 4 proximity can still operate.
        if not feet:
            for i in range(len(tracks)):
                x1, y1, x2, y2 = map(float, tracks.xyxy[i])
                feet[int(tracks.tracker_id[i])] = [[(x1 + x2) / 2.0, y2]]
        return feet

    def close(self) -> None:
        self.client.close()
