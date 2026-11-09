"""Render a futsal tracking/radar video with existing Roboflow inference models.

Notebook usage (models/classifier must already be loaded and trained)::

    from futsal_live_overlay import OverlayConfig, run_overlay
    stats = run_overlay(
        PLAYER_DETECTION_MODEL, FIELD_DETECTION_MODEL, team_classifier,
        OverlayConfig(source_path='/content/1103.mp4',
                      output_path='/content/futsal_live_overlay.mp4'),
    )

CLI usage: python futsal_live_overlay.py --help
Dependencies: numpy, opencv-python, supervision (tested with 0.25.1), tqdm.
FFmpeg is optional; imageio-ffmpeg can also supply its executable.
Landmark IDs/coordinates below are inherited from the original script. Verify
them against YOUR trained skeleton before interpreting the radar as positions.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import importlib
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Mapping

import cv2
import numpy as np
import supervision as sv
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
SOURCE_VIDEO_PATH = "/content/1103.mp4"
OUTPUT_VIDEO_PATH = "/content/futsal_live_overlay.mp4"
BALL_ID, GOALKEEPER_ID, PLAYER_ID = 0, 1, 2
COURT_LENGTH, COURT_WIDTH = 40.0, 20.0
ACTIVE_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 23, 25, 26, 27, 28, 29, 31)
LANDMARKS = {
    0: (0, 0),
    3: (0, 20),
    4: (20, 0),
    5: (40, 0),
    8: (40, 20),
    9: (20, 20),
    1: (0, 8.5),
    2: (0, 11.5),
    6: (40, 8.5),
    7: (40, 11.5),
    25: (0, 2.5),
    26: (0, 17.5),
    27: (40, 2.5),
    28: (40, 17.5),
    10: (20, 13),
    12: (20, 7),
    11: (23, 10),
    13: (17, 10),
    15: (6, 10),
    20: (34, 10),
    29: (30, 10),
    31: (10, 10),
}
TEAM_BGR = {0: (255, 191, 0), 1: (147, 20, 255), 2: (150, 150, 150)}
BALL_BGR = (0, 215, 255)


@dataclass(frozen=True)
class OverlayConfig:
    source_path: str | Path = SOURCE_VIDEO_PATH
    output_path: str | Path = OUTPUT_VIDEO_PATH
    detection_threshold: float = 0.3
    keypoint_threshold: float = 0.35
    keypoint_interval: int = 5
    max_homography_age: int = 10  # Frames since the last SUCCESSFUL calibration.
    ransac_threshold: float = 0.75  # Court meters, because image -> court is fitted.
    max_median_error: float = 0.5
    min_inlier_ratio: float = 0.5
    radar_width_ratio: float = 0.47
    radar_opacity: float = 0.62
    radar_position: str = "bottom-center"
    show_radar_ids: bool = True
    team_interval: int = 5
    vote_window: int = 15
    track_ttl: int = 150
    fallback_fps: float = 30.0
    max_frames: int | None = None
    overwrite: bool = False
    progress: bool = True
    active_ids: tuple[int, ...] = ACTIVE_IDS
    full_keypoint_count: int = 33
    landmarks: Mapping[int, tuple[float, float]] = field(default_factory=lambda: dict(LANDMARKS))

    def __post_init__(self):
        for name in ("detection_threshold", "keypoint_threshold", "min_inlier_ratio", "radar_opacity"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 < self.radar_width_ratio <= 1:
            raise ValueError("radar_width_ratio must be in (0, 1]")
        for name in ("keypoint_interval", "team_interval", "vote_window", "track_ttl", "full_keypoint_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.max_homography_age, int) or self.max_homography_age < 0:
            raise ValueError("max_homography_age must be a nonnegative integer")
        for name in ("ransac_threshold", "max_median_error", "fallback_fps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_frames is not None and (
            not isinstance(self.max_frames, int) or isinstance(self.max_frames, bool) or self.max_frames < 1
        ):
            raise ValueError("max_frames must be a positive integer or None")
        if self.radar_position not in ("bottom-center", "bottom-left", "bottom-right"):
            raise ValueError("Unsupported radar_position")
        if len(set(self.active_ids)) != len(self.active_ids) or any(
            not isinstance(pid, int) or pid < 0 or pid >= self.full_keypoint_count for pid in self.active_ids
        ):
            raise ValueError("active_ids must be unique IDs within full_keypoint_count")
        coordinates = np.asarray(list(self.landmarks.values()), dtype=float)
        if len(self.landmarks) < 4 or not set(self.landmarks).issubset(self.active_ids):
            raise ValueError("At least four landmarks are required, all present in active_ids")
        if coordinates.shape != (len(self.landmarks), 2) or not np.isfinite(coordinates).all():
            raise ValueError("Landmarks must be finite (x, y) pairs")
        if np.any(coordinates < 0) or np.any(coordinates > (COURT_LENGTH, COURT_WIDTH)):
            raise ValueError("Landmarks must be inside the 40 x 20 meter court")
        if np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0)) < 2:
            raise ValueError("Landmarks must not be collinear")


@dataclass(frozen=True)
class OverlayStats:
    output_path: Path
    frames_processed: int
    frames_calibrated: int
    elapsed_seconds: float
    h264_encoded: bool


@dataclass
class HomographyState:
    matrix: np.ndarray | None = None
    frame_index: int = -1
    inliers: int = 0

    def update(self, matrix, frame_index, inliers):
        if matrix is not None:
            self.matrix, self.frame_index, self.inliers = matrix, frame_index, inliers

    def current(self, frame_index, max_age):
        if self.matrix is None or not 0 <= frame_index - self.frame_index <= max_age:
            return None
        return self.matrix


def make_radar_base(width=900, padding=28):
    if width <= 2 * padding or padding < 0:
        raise ValueError("Radar width must exceed twice its nonnegative padding")
    ppm = (width - 2 * padding) / COURT_LENGTH
    height = round(COURT_WIDTH * ppm) + padding * 2
    canvas = np.full((height, width, 3), (45, 115, 45), dtype=np.uint8)

    def p(x, y):
        return round(padding + x * ppm), round(padding + y * ppm)

    white = (235, 245, 235)
    cv2.rectangle(canvas, p(0, 0), p(40, 20), white, 2, cv2.LINE_AA)
    cv2.line(canvas, p(20, 0), p(20, 20), white, 2, cv2.LINE_AA)
    cv2.circle(canvas, p(20, 10), round(3 * ppm), white, 2, cv2.LINE_AA)
    cv2.circle(canvas, p(20, 10), 3, white, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, p(-1, 8.5), p(0, 11.5), white, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, p(40, 8.5), p(41, 11.5), white, 2, cv2.LINE_AA)
    for gx, direction in ((0, 1), (40, -1)):
        a, b = np.linspace(-np.pi / 2, 0, 80), np.linspace(0, np.pi / 2, 80)
        xx = np.r_[
            gx + direction * 6 * np.cos(a), np.full(25, gx + direction * 6), gx + direction * 6 * np.cos(b)
        ]
        yy = np.r_[8.5 + 6 * np.sin(a), np.linspace(8.5, 11.5, 25), 11.5 + 6 * np.sin(b)]
        points = np.array([p(x, y) for x, y in zip(xx, yy)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, white, 2, cv2.LINE_AA)
    for x in (6, 10, 30, 34):
        cv2.circle(canvas, p(x, 10), 3, white, -1, cv2.LINE_AA)
    for x, y, start, end in ((0, 0, 0, 90), (40, 0, 90, 180), (40, 20, 180, 270), (0, 20, 270, 360)):
        cv2.ellipse(canvas, p(x, y), (max(1, round(0.25 * ppm)),) * 2, 0, start, end, white, 2, cv2.LINE_AA)
    return canvas, ppm, padding


def field_homography(frame, field_model, config):
    """Choose the strongest valid court instance; reject uncertain/degenerate fits."""
    results = field_model.infer(frame, confidence=config.detection_threshold)
    if not results:
        return None, 0
    try:
        kps = sv.KeyPoints.from_inference(results[0])
        confidence = getattr(kps, "keypoint_confidence", None)
        if confidence is None:
            confidence = getattr(kps, "confidence", None)
        if confidence is None or not len(kps.xy):
            return None, 0
        best_h, best_score = None, 0
        for xy, conf in zip(kps.xy, confidence):
            xy, conf = np.asarray(xy, np.float32), np.asarray(conf, np.float32).reshape(-1)
            if xy.shape == (len(config.active_ids), 2):
                index = {pid: i for i, pid in enumerate(config.active_ids)}
            elif xy.shape == (config.full_keypoint_count, 2):
                index = {pid: pid for pid in config.active_ids}
            else:
                LOGGER.debug("Unexpected keypoint shape: %s", xy.shape)
                continue
            if len(conf) != len(xy):
                continue
            pairs = [
                (xy[index[pid]], point)
                for pid, point in config.landmarks.items()
                if np.isfinite(conf[index[pid]])
                and conf[index[pid]] >= config.keypoint_threshold
                and np.isfinite(xy[index[pid]]).all()
                and not np.allclose(xy[index[pid]], 0)
            ]
            if len(pairs) < 4:
                continue
            src, dst = (np.asarray(points, np.float32) for points in zip(*pairs))
            if any(np.linalg.matrix_rank(points - points.mean(axis=0)) < 2 for points in (src, dst)):
                continue
            matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, config.ransac_threshold)
            if matrix is None or inliers is None or not np.isfinite(matrix).all():
                continue
            good = inliers.ravel().astype(bool)
            score = int(good.sum())
            if score < 4 or score / len(src) < config.min_inlier_ratio or np.linalg.matrix_rank(matrix) < 3:
                continue
            errors = np.linalg.norm(court_points(src, matrix) - dst, axis=1)
            if not np.isfinite(errors[good]).all() or np.median(errors[good]) > config.max_median_error:
                continue
            if score > best_score:
                best_h, best_score = matrix, score
        return best_h, best_score
    except (ValueError, IndexError, KeyError, TypeError, cv2.error) as exc:
        LOGGER.debug("Field calibration rejected: %s", exc)
        return None, 0


def court_points(points, matrix):
    """Preserve row alignment; points on the projective horizon become NaN."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if matrix is None or len(points) == 0:
        return np.empty((0, 2), np.float32)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-8)
    projected[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
    return projected


class TeamHistory:
    """Classify valid crops periodically and forget tracks that disappeared."""

    def __init__(self, config):
        self.config = config
        self.votes = {}
        self.last_seen = {}

    def assign(self, frame, tracked, classifier, frame_index):
        for tid in list(self.last_seen):
            if frame_index - self.last_seen[tid] > self.config.track_ttl:
                self.last_seen.pop(tid)
                self.votes.pop(tid, None)
        teams = np.full(len(tracked), 2, dtype=int)
        crops, indices = [], []
        height, width = frame.shape[:2]
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            self.last_seen[tid] = frame_index
            if tracked.class_id[i] != PLAYER_ID:
                continue
            history = self.votes.setdefault(tid, deque(maxlen=self.config.vote_window))
            if frame_index % self.config.team_interval and history:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i]
            x1, x2 = np.clip([math.floor(x1), math.ceil(x2)], 0, width)
            y1, y2 = np.clip([math.floor(y1), math.ceil(y2)], 0, height)
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2].copy())
                indices.append(i)
        if crops:
            predictions = np.asarray(classifier.predict(crops)).reshape(-1)
            if len(predictions) != len(indices):
                raise ValueError("Team predictions do not match the number of valid player crops")
            for i, team in zip(indices, predictions):
                if team in (0, 1):
                    self.votes[int(tracked.tracker_id[i])].append(int(team))
        for i in range(len(tracked)):
            if tracked.class_id[i] != PLAYER_ID:
                continue  # Goalkeepers stay neutral; their team is not inferred from proximity.
            history = self.votes.get(int(tracked.tracker_id[i]))
            if history:
                counts = (history.count(0), history.count(1))
                teams[i] = history[-1] if counts[0] == counts[1] else int(counts[1] > counts[0])
        return teams


def draw_camera(frame, tracked, teams, ball_box):
    height, width = frame.shape[:2]
    for i, (box, tid, role) in enumerate(zip(tracked.xyxy, tracked.tracker_id, tracked.class_id)):
        x1, y1, x2, y2 = map(int, box)
        color, cx = TEAM_BGR[int(teams[i])], (x1 + x2) // 2
        cv2.ellipse(
            frame,
            (cx, y2),
            (max(10, (x2 - x1) // 2), max(4, (x2 - x1) // 5)),
            0,
            0,
            360,
            color,
            2,
            cv2.LINE_AA,
        )
        label = ("GK " if role == GOALKEEPER_ID else "") + f"#{int(tid)}"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        left = max(0, min(cx - text_width // 2 - 5, width - text_width - 10))
        top = max(0, min(y1 - 24, height - 22))
        cv2.rectangle(frame, (left, top), (left + text_width + 10, top + 21), color, -1)
        cv2.putText(
            frame, label, (left + 5, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    if ball_box is not None:
        x1, y1, x2, y2 = ball_box
        bx, by = round((x1 + x2) / 2), round((y1 + y2) / 2)
        triangle = np.array([(bx, by - 17), (bx - 10, by - 32), (bx + 10, by - 32)], np.int32)
        cv2.fillPoly(frame, [triangle], BALL_BGR)


class Radar:
    def __init__(self, frame_shape, config):
        self.config = config
        base, self.ppm, self.padding = make_radar_base()
        height, width = frame_shape[:2]
        margin = min(max(8, round(0.025 * height)), min(width, height) // 8)
        scale = min(
            width * config.radar_width_ratio / base.shape[1],
            (width - 2 * margin) / base.shape[1],
            (height - 2 * margin) / base.shape[0],
        )
        radar_w, radar_h = max(1, round(base.shape[1] * scale)), max(1, round(base.shape[0] * scale))
        self.base = cv2.resize(base, (radar_w, radar_h), interpolation=cv2.INTER_AREA)
        self.scale_x, self.scale_y = radar_w / base.shape[1], radar_h / base.shape[0]
        self.x = {
            "bottom-center": (width - radar_w) // 2,
            "bottom-left": margin,
            "bottom-right": width - margin - radar_w,
        }[config.radar_position]
        self.y = height - margin - radar_h

    def draw(self, frame, tracked, teams, ball_box, matrix, state, frame_index):
        height, width = self.base.shape[:2]
        roi = frame[self.y : self.y + height, self.x : self.x + width]
        cv2.addWeighted(self.base, self.config.radar_opacity, roi, 1 - self.config.radar_opacity, 0, roi)
        if matrix is None:
            status = "RADAR: calibration unavailable"
            status_color = (100, 210, 255)
        else:
            age = frame_index - state.frame_index
            status = f"RADAR: {state.inliers} landmarks" + (f" | held {age}f" if age else "")
            status_color = (220, 255, 220)
            positions = court_points(tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER), matrix)
            for i, position in enumerate(positions):
                label = str(int(tracked.tracker_id[i])) if self.config.show_radar_ids else None
                self.dot(frame, position, TEAM_BGR[int(teams[i])], 6, label)
            if ball_box is not None:
                x1, y1, x2, y2 = ball_box
                # Ball projection assumes the ball lies on the court plane.
                position = court_points([[(x1 + x2) / 2, (y1 + y2) / 2]], matrix)[0]
                self.dot(frame, position, BALL_BGR, 5)
        if width >= 200:
            font_scale = min(0.45, width / 850)
            y = self.y - 6 if self.y >= 18 else self.y + 14
            cv2.putText(
                frame,
                status,
                (self.x + 3, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (15, 15, 15),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                status,
                (self.x + 3, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                status_color,
                1,
                cv2.LINE_AA,
            )

    def dot(self, frame, position, color, radius, label=None):
        x, y = position
        if not np.isfinite(position).all() or not (0 <= x <= COURT_LENGTH and 0 <= y <= COURT_WIDTH):
            return
        cx = self.x + round((self.padding + self.ppm * x) * self.scale_x)
        cy = self.y + round((self.padding + self.ppm * y) * self.scale_y)
        radius = max(2, round(radius * min(1.0, self.base.shape[1] / 500)))
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, (20, 20, 20), 1, cv2.LINE_AA)
        if label:
            cv2.putText(
                frame,
                label,
                (cx + radius + 2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (cx + radius + 2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def finalize_video(raw_path, source_path, staging_path, frames_processed, fps):
    """Encode into a staging file so a failed render never replaces the destination."""
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:a",
            "aac",
            "-t",
            f"{frames_processed / fps:.9f}",
            "-movflags",
            "+faststart",
            str(staging_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
            if result.returncode == 0 and staging_path.is_file() and staging_path.stat().st_size:
                return True
            LOGGER.warning("FFmpeg conversion failed: %s", result.stderr[-1000:])
        except OSError as exc:
            LOGGER.warning("Could not run FFmpeg: %s", exc)
    else:
        LOGGER.warning("FFmpeg is unavailable")
    LOGGER.warning("Saving MPEG-4 video without source audio; H.264 browser compatibility is unavailable")
    shutil.copy2(raw_path, staging_path)
    return False


def _safe_frame_count(value):
    return int(value) if math.isfinite(value) and value > 0 else None


def run_overlay(player_model, field_model, team_classifier, config=None):
    """Render using model.infer(frame, confidence=...) and classifier.predict(crops).

    Models must return Roboflow-compatible results; classifier labels are 0/1.
    Goalkeepers remain neutral. The output is published only after success.
    """
    config = config or OverlayConfig()
    source = Path(config.source_path).expanduser().resolve()
    output = Path(config.output_path).expanduser().resolve()
    if source == output or (source.exists() and output.exists() and os.path.samefile(source, output)):
        raise ValueError("Source and output must be different files")
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_path must use the .mp4 extension")
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"Output already exists: {output}; set overwrite=True to replace it")
    for model, name, method in (
        (player_model, "player_model", "infer"),
        (field_model, "field_model", "infer"),
        (team_classifier, "team_classifier", "predict"),
    ):
        if not callable(getattr(model, method, None)):
            raise TypeError(f"{name} must provide {method}()")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = calibrated = 0
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open the source video: {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            LOGGER.warning("Invalid source FPS; using %.2f", config.fallback_fps)
            fps = config.fallback_fps
        total = _safe_frame_count(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if config.max_frames is not None and total is not None:
            total = min(total, config.max_frames)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Source video contains no decodable frames")
        height, width = frame.shape[:2]
        # OpenCV's MPEG-4 writer otherwise silently truncates odd dimensions.
        encoded_size = (width + width % 2, height + height % 2)
        tracker = sv.ByteTrack(frame_rate=max(1, round(fps)))
        teams_history, state, radar = TeamHistory(config), HomographyState(), Radar(frame.shape, config)
        with tempfile.TemporaryDirectory(prefix=".futsal-overlay-", dir=output.parent) as directory:
            raw_path, staging = Path(directory) / "raw.mp4", Path(directory) / "final.mp4"
            writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, encoded_size)
            try:
                if not writer.isOpened():
                    raise RuntimeError("OpenCV VideoWriter failed to initialize (MPEG-4 codec)")
                with tqdm(
                    total=total, desc="Rendering futsal overlay", unit="frame", disable=not config.progress
                ) as progress:
                    while ok and (config.max_frames is None or processed < config.max_frames):
                        if frame is None or frame.shape[:2] != (height, width):
                            raise RuntimeError(f"Invalid frame dimensions at frame {processed}")
                        results = player_model.infer(frame, confidence=config.detection_threshold)
                        detections = (
                            sv.Detections.from_inference(results[0]) if results else sv.Detections.empty()
                        )
                        if len(detections) and (detections.class_id is None or detections.confidence is None):
                            raise ValueError("Player model must return class IDs and detection confidence")
                        # Filter malformed boxes before cropping, NMS or tracking.
                        detections.xyxy = np.clip(detections.xyxy, 0, [width, height, width, height])
                        valid = np.isfinite(detections.xyxy).all(axis=1)
                        valid &= np.all(detections.xyxy[:, 2:] > detections.xyxy[:, :2], axis=1)
                        if detections.confidence is not None:
                            valid &= np.isfinite(detections.confidence)
                        detections = detections[valid]
                        balls = detections[detections.class_id == BALL_ID]
                        ball_box = balls.xyxy[int(np.argmax(balls.confidence))] if len(balls) else None
                        people = detections[np.isin(detections.class_id, [GOALKEEPER_ID, PLAYER_ID])]
                        tracked = tracker.update_with_detections(
                            people.with_nms(threshold=0.5, class_agnostic=True)
                        )
                        teams = teams_history.assign(frame, tracked, team_classifier, processed)
                        if processed % config.keypoint_interval == 0:
                            matrix, score = field_homography(frame, field_model, config)
                            state.update(matrix, processed, score)
                        matrix = state.current(processed, config.max_homography_age)
                        calibrated += int(matrix is not None)
                        # All inference sees the unannotated frame. Draw only afterwards.
                        draw_camera(frame, tracked, teams, ball_box)
                        radar.draw(frame, tracked, teams, ball_box, matrix, state, processed)
                        if encoded_size != (width, height):
                            frame = cv2.copyMakeBorder(
                                frame, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE
                            )
                        writer.write(frame)
                        processed += 1
                        progress.update(1)
                        if config.max_frames is not None and processed >= config.max_frames:
                            break
                        ok, frame = cap.read()
            finally:
                writer.release()
                writer = None
            # Close the input before FFmpeg reopens it, including on Windows.
            cap.release()
            if not processed or not raw_path.is_file() or raw_path.stat().st_size == 0:
                raise RuntimeError("No video frames were written")
            h264 = finalize_video(raw_path, source, staging, processed, fps)
            if config.overwrite:
                os.replace(staging, output)
            else:
                # Same filesystem: atomically publish without replacing a concurrent output.
                os.link(staging, output)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    stats = OverlayStats(output, processed, calibrated, time.perf_counter() - started, h264)
    LOGGER.info(
        "Saved %s | %d frames | radar calibrated %.1f%% | %.1f seconds",
        output,
        processed,
        100 * calibrated / processed,
        stats.elapsed_seconds,
    )
    if calibrated == 0:
        LOGGER.warning(
            "No usable court calibration: verify landmark order, coordinates and keypoint confidence"
        )
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default=SOURCE_VIDEO_PATH)
    parser.add_argument("--output", default=OUTPUT_VIDEO_PATH)
    parser.add_argument(
        "--model-factory",
        required=True,
        metavar="MODULE:FUNCTION",
        help="Zero-argument function returning (player_model, field_model, team_classifier)",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--keypoint-interval", type=int, default=5)
    parser.add_argument("--max-homography-age", type=int, default=10)
    parser.add_argument("--keypoint-threshold", type=float, default=0.35)
    parser.add_argument("--radar-width-ratio", type=float, default=0.47)
    parser.add_argument("--radar-opacity", type=float, default=0.62)
    parser.add_argument(
        "--radar-position", choices=["bottom-center", "bottom-left", "bottom-right"], default="bottom-center"
    )
    parser.add_argument("--team-interval", type=int, default=5)
    parser.add_argument("--hide-radar-ids", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        config = OverlayConfig(
            source_path=args.source,
            output_path=args.output,
            max_frames=args.max_frames,
            keypoint_interval=args.keypoint_interval,
            max_homography_age=args.max_homography_age,
            keypoint_threshold=args.keypoint_threshold,
            radar_width_ratio=args.radar_width_ratio,
            radar_opacity=args.radar_opacity,
            radar_position=args.radar_position,
            team_interval=args.team_interval,
            show_radar_ids=not args.hide_radar_ids,
            overwrite=args.overwrite,
            progress=not args.no_progress,
        )
        module, separator, function = args.model_factory.partition(":")
        if not separator or not module or not function:
            raise ValueError("--model-factory must have the form MODULE:FUNCTION")
        factory = getattr(importlib.import_module(module), function)
        stats = run_overlay(*factory(), config=config)
    except (ValueError, TypeError, OSError, RuntimeError, ImportError, AttributeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(
        f"Saved: {stats.output_path}\nFrames: {stats.frames_processed}; "
        f"calibrated: {stats.frames_calibrated}; H.264: {stats.h264_encoded}"
    )


if __name__ == "__main__":
    # Pasting into Colab or using %run -i retains the original loaded-model workflow.
    if all(
        name in globals() for name in ("PLAYER_DETECTION_MODEL", "FIELD_DETECTION_MODEL", "team_classifier")
    ):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        run_overlay(
            globals()["PLAYER_DETECTION_MODEL"],
            globals()["FIELD_DETECTION_MODEL"],
            globals()["team_classifier"],
        )
    else:
        main()
