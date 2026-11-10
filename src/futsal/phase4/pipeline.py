"""Phase 4 Pipeline: Ball tracking, Pose-to-Ankle proximity possession, and Pass/Shot detection."""

from __future__ import annotations

from collections import deque
import csv
import json
import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from futsal.match.events import BallSelector, EventEngine, PossessionEngine
from futsal.phase1.annotations import COLORS, GROUPS
from futsal.phase1.detection import Detector
from futsal.phase1.pipeline import write_json
from futsal.phase1.video import VideoInfo, VideoReader, VideoWriter
from futsal.phase2.geometry import project
from futsal.phase3.pipeline import PHASE3_COLUMNS, SpeedTrackingAnnotator, position_groups
from futsal.phase3.radar import SpatialRadar

LOG = logging.getLogger(__name__)
PHASE4_COLUMNS = PHASE3_COLUMNS + [
    "ball_x_px",
    "ball_y_px",
    "ball_pitch_x_m",
    "ball_pitch_y_m",
    "possession_state",
    "possession_owner_id",
    "possession_team",
    "in_possession",
]
EVENT_COLUMNS = [
    "event_id",
    "type",
    "time_s",
    "segment_id",
    "source_id",
    "receiver_id",
    "team",
    "evidence",
    "status",
    "trajectory_basis",
]


class PoseEstimator(Protocol):
    def estimate(self, frame: np.ndarray, tracks: sv.Detections) -> dict[int, list[list[float]]]:
        """Return {track_id: [[ankle_x, ankle_y], ...]} in source pixel coordinates."""
        ...


def load_homography_lookup(phase2_dir: Path) -> dict[int, np.ndarray | None]:
    """Load per-frame 3x3 image_to_pitch matrices from Phase 2 calibration_frames.jsonl."""
    lookup: dict[int, np.ndarray | None] = {}
    audit_path = phase2_dir / "calibration_frames.jsonl"
    if not audit_path.is_file():
        return lookup
    with audit_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            mat = rec.get("image_to_pitch")
            lookup[int(rec["frame_index"])] = np.asarray(mat, dtype=np.float64) if mat is not None else None
    return lookup


class PossessionAnnotator(SpeedTrackingAnnotator):
    """Draws player boxes, ankle keypoints, active possession rings, ball trail, and recent events."""

    def __init__(self, trail_length: int = 20) -> None:
        super().__init__()
        self.ball_trail: deque[tuple[int, int]] = deque(maxlen=trail_length)
        self.recent_event: tuple[int, str] | None = None

    def reset_trail(self) -> None:
        self.ball_trail.clear()

    def annotate_phase4(
        self,
        frame: np.ndarray,
        tracks: sv.Detections,
        rows: list[dict],
        feet: dict[int, list[list[float]]],
        ball: dict | None,
        possession: dict,
        emitted_events: list[dict],
        frame_index: int,
        segment_id: int,
    ) -> np.ndarray:
        owner_id = possession.get("owner_id")
        base_frame = frame.copy()
        if owner_id is not None:
            for row in rows:
                if int(row["track_id"]) == int(owner_id):
                    cx = round((row["x1"] + row["x2"]) / 2)
                    cy = round(row["y2"])
                    rx = max(24, round((row["x2"] - row["x1"]) * 0.65))
                    ry = max(9, round(rx * 0.35))
                    # Draw ground ring BEFORE annotate_with_metrics so LabelAnnotator draws the text pill on top
                    cv2.ellipse(base_frame, (cx, cy), (rx, ry), 0, -45, 235, (0, 255, 180), 3, cv2.LINE_AA)

        scene = self.annotate_with_metrics(base_frame, tracks, rows, frame_index, segment_id)

        # 1. Draw detected ankle keypoints and overhead EA FC pointer triangle for the player in possession
        for row in rows:
            tid = int(row["track_id"])
            for ax, ay in feet.get(tid, []):
                pt = (round(ax), round(ay))
                cv2.circle(scene, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(scene, pt, 5, (20, 20, 20), 1, cv2.LINE_AA)
            if owner_id is not None and tid == int(owner_id):
                cx = round((row["x1"] + row["x2"]) / 2)
                ty = max(16, round(row["y1"]) - 8)
                tri = np.array([[cx, ty], [cx - 8, ty - 12], [cx + 8, ty - 12]], dtype=np.int32)
                cv2.fillPoly(scene, [tri], (0, 255, 180), cv2.LINE_AA)
                cv2.polylines(scene, [tri], True, (20, 20, 20), 1, cv2.LINE_AA)

        # 2. Draw ball trajectory trail and current ball marker
        if ball is not None:
            if not ball.get("continuous", False):
                self.ball_trail.clear()
            bpt = (round(ball["x"]), round(ball["y"]))
            self.ball_trail.append(bpt)
            pts = list(self.ball_trail)
            for i in range(1, len(pts)):
                thickness = max(1, round(3 * (i / len(pts))))
                cv2.line(scene, pts[i - 1], pts[i], (0, 200, 255), thickness, cv2.LINE_AA)
            cv2.circle(scene, bpt, 6, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(scene, bpt, 7, (15, 15, 15), 1, cv2.LINE_AA)
        else:
            self.ball_trail.clear()

        # 3. Possession & Event banner
        if emitted_events:
            ev = emitted_events[-1]
            label = f"EVENT: {ev['type'].upper()} (#{ev['source_id']}"
            if ev.get("receiver_id") is not None:
                label += f" -> #{ev['receiver_id']}"
            label += f", {ev['team']})"
            self.recent_event = (frame_index, label)

        status_text = f"Possession: {possession['state']}"
        if owner_id is not None:
            status_text += f" | #{owner_id} ({possession.get('team')})"
        cv2.rectangle(scene, (0, 30), (min(scene.shape[1] - 1, 520), 58), (32, 38, 42), -1)
        cv2.putText(scene, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1, cv2.LINE_AA)

        if self.recent_event is not None and 0 <= frame_index - self.recent_event[0] <= 45:
            cv2.rectangle(scene, (0, 58), (min(scene.shape[1] - 1, 520), 86), (20, 75, 115), -1)
            cv2.putText(
                scene,
                self.recent_event[1],
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 240, 150),
                1,
                cv2.LINE_AA,
            )
        return scene


class Phase4Pipeline:
    """Combines spatial tracks, ball detections, and ankle pose keypoints to compute possession & events."""

    def __init__(
        self,
        detector: Detector,
        pose_estimator: PoseEstimator,
        confirm_s: float = 0.12,
    ) -> None:
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.confirm_s = confirm_s

    def run(
        self,
        phase2: Path,
        output: Path,
        phase3: Path | None = None,
        video_override: Path | None = None,
    ) -> Path:
        manifest = json.loads((phase2 / "run.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError("Phase 4 requires a completed Phase 2 run")
        video = video_override or Path(manifest["source"])
        start = int(manifest["start_frame"])
        count = int(manifest["frames_processed"])
        length, width = manifest.get("pitch_size_m", [40.0, 20.0])
        units = manifest.get("coordinate_units", "assumed_m")

        with VideoReader(video) as reader:
            info = reader.info
            assert info is not None

        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)

        homographies = load_homography_lookup(phase2)
        csv_source = (phase3 / "metrics.csv") if (phase3 and (phase3 / "metrics.csv").is_file()) else (phase2 / "positions.csv")
        groups = position_groups(csv_source)
        current_index, current_rows = next(groups, (None, []))
        while current_index is not None and current_index < start:
            current_index, current_rows = next(groups, (None, []))

        ball_selector = BallSelector()
        possession_engine = PossessionEngine(confirm_s=self.confirm_s)
        event_engine = EventEngine(length=length, width=width)
        radar = SpatialRadar(length=length, width=width)
        annotator = PossessionAnnotator()

        run_meta = {
            "phase": 4,
            "status": "running",
            "phase2_run": str(phase2.resolve()),
            "phase3_run": str(phase3.resolve()) if phase3 else None,
            "source": str(video.resolve()),
            "start_frame": start,
            "frames_requested": count,
            "fps": info.fps,
            "confirm_s": self.confirm_s,
        }
        write_json(output / "run.json", run_meta)

        frames_done = 0
        team_possession_counts = {"team_a": 0, "team_b": 0}
        last_segment = None

        try:
            with (
                VideoReader(video) as reader,
                VideoWriter(output / "tactical_events.mp4", info) as writer,
                (output / "possession_tracks.csv").open("w", newline="", encoding="utf-8") as csv_file,
                (output / "possession_frames.jsonl").open("w", encoding="utf-8") as frame_log,
            ):
                csv_writer = csv.DictWriter(csv_file, fieldnames=PHASE4_COLUMNS, extrasaction="ignore")
                csv_writer.writeheader()

                for index, frame in reader.frames(start, count):
                    rows = current_rows if current_index == index else []
                    if rows:
                        current_index, current_rows = next(groups, (None, []))

                    time_s = index / info.fps
                    segment = rows[0]["segment_id"] if rows else (last_segment or 0)
                    if last_segment is not None and segment != last_segment:
                        ball_selector.reset()
                        possession_engine.reset()
                        event_engine.reset()
                        annotator.reset_trail()
                    last_segment = segment

                    boxes = np.array([[r[k] for k in ("x1", "y1", "x2", "y2")] for r in rows]).reshape(-1, 4)
                    tracks = sv.Detections(
                        xyxy=boxes.astype(np.float32) if len(rows) else np.empty((0, 4), np.float32),
                        confidence=np.array([r["detection_confidence"] for r in rows], np.float32),
                        class_id=np.zeros(len(rows), int),
                        tracker_id=np.array([r["track_id"] for r in rows], int),
                        data={"role": np.array([r["role"] for r in rows])},
                    )

                    # 1. Estimate ankle keypoints for tracked players and detect ball (masking player shoes/socks)
                    feet = self.pose_estimator.estimate(frame, tracks)
                    try:
                        raw_detections = self.detector.detect(frame, rows, feet)
                    except TypeError:
                        raw_detections = self.detector.detect(frame)
                    ball = ball_selector.select(raw_detections, time_s, frame.shape)

                    # 2. Project ball center to 2D pitch if homography is valid
                    matrix = homographies.get(index)
                    ball_pitch = None
                    if ball is not None and matrix is not None:
                        try:
                            bx, by = project(matrix, [[ball["x"], ball["y"]]])[0]
                            if 0.0 <= bx <= length and 0.0 <= by <= width:
                                ball_pitch = [float(bx), float(by)]
                        except ValueError:
                            ball_pitch = None

                    # 3. Update possession state machine & event detector
                    possession = possession_engine.update(time_s, ball, rows, feet)
                    if possession["state"] == "controlled" and possession["team"] in team_possession_counts:
                        team_possession_counts[possession["team"]] += 1

                    anchor = rows[0].get("anchor_frame") if rows else None
                    emitted = event_engine.update(
                        time_s=time_s,
                        segment_id=segment,
                        possession=possession,
                        ball_pitch=ball_pitch,
                        continuous=bool(ball and ball.get("continuous", False)),
                        anchor=anchor,
                    )

                    # 4. Write enriched rows and frame log
                    for row in rows:
                        tid = int(row["track_id"])
                        item = {
                            **row,
                            "speed_kmh": float(row["speed_kmh"]) if row.get("speed_kmh") not in ("", None) else None,
                            "distance_step_m": (
                                float(row["distance_step_m"]) if row.get("distance_step_m") not in ("", None) else None
                            ),
                            "observed_distance_m": (
                                float(row["observed_distance_m"])
                                if row.get("observed_distance_m") not in ("", None)
                                else None
                            ),
                            "ball_x_px": ball["x"] if ball else None,
                            "ball_y_px": ball["y"] if ball else None,
                            "ball_pitch_x_m": ball_pitch[0] if ball_pitch else None,
                            "ball_pitch_y_m": ball_pitch[1] if ball_pitch else None,
                            "possession_state": possession["state"],
                            "possession_owner_id": possession["owner_id"],
                            "possession_team": possession["team"],
                            "in_possession": possession["state"] == "controlled" and possession["owner_id"] == tid,
                        }
                        csv_writer.writerow(item)

                    frame_log.write(
                        json.dumps(
                            {
                                "frame_index": index,
                                "timestamp_s": time_s,
                                "ball": ball,
                                "ball_pitch_m": ball_pitch,
                                "possession": possession,
                                "ankles_px": {str(k): v for k, v in feet.items()},
                                "events": emitted,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )

                    # 5. Render video frame with radar + ball dot
                    cal_status = rows[0]["calibration_status"] if rows else ("anchor" if matrix is not None else "no_anchor")
                    radar_img = radar.render(rows, status=cal_status, units=units)
                    if ball_pitch is not None:
                        bp = radar.point(ball_pitch[0], ball_pitch[1])
                        cv2.circle(radar_img, bp, 5, (0, 215, 255), -1, cv2.LINE_AA)
                        cv2.circle(radar_img, bp, 6, (15, 15, 15), 1, cv2.LINE_AA)

                    annotated = annotator.annotate_phase4(
                        frame, tracks, rows, feet, ball, possession, emitted, index, segment
                    )
                    writer.write(radar.overlay(annotated, radar_img))
                    frames_done += 1

            pd.DataFrame(event_engine.events, columns=EVENT_COLUMNS).to_csv(output / "events.csv", index=False)
            total_controlled = sum(team_possession_counts.values())
            possession_summary = {
                "controlled_frames": total_controlled,
                "team_a_controlled_frames": team_possession_counts["team_a"],
                "team_b_controlled_frames": team_possession_counts["team_b"],
                "team_a_possession_pct": (
                    100.0 * team_possession_counts["team_a"] / total_controlled if total_controlled else None
                ),
                "team_b_possession_pct": (
                    100.0 * team_possession_counts["team_b"] / total_controlled if total_controlled else None
                ),
                "events_detected": len(event_engine.events),
                "passes_detected": sum(1 for e in event_engine.events if e["type"] == "pass_candidate"),
                "shots_detected": sum(1 for e in event_engine.events if e["type"] == "shot_candidate"),
            }
            write_json(output / "possession_summary.json", possession_summary)
            run_meta.update(status="completed", frames_processed=frames_done, **possession_summary)
        except BaseException as error:
            run_meta.update(status="failed", error_type=type(error).__name__)
            write_json(output / "run.json", run_meta)
            raise
        finally:
            groups.close()

        write_json(output / "run.json", run_meta)
        return output
