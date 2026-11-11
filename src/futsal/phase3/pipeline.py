"""Phase 3 Pipeline: consumes Phase 2 positions.csv, computes speed/distance & Voronoi/Hulls."""

from __future__ import annotations

import csv
import json
import logging
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from futsal.match.spatial import MovementEngine, SpatialEngine
from futsal.phase1.annotations import GROUPS, TrackingAnnotator
from futsal.phase1.pipeline import TRACK_COLUMNS, write_json
from futsal.phase1.video import VideoInfo, VideoReader, VideoWriter
from futsal.phase2.pipeline import EXTRA_COLUMNS
from .radar import SpatialRadar

LOG = logging.getLogger(__name__)
PHASE3_COLUMNS = TRACK_COLUMNS + EXTRA_COLUMNS + [
    "speed_kmh",
    "distance_step_m",
    "observed_distance_m",
]


def position_groups(path: Path):
    """Stream Phase 2 positions.csv grouped by frame_index."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = set(TRACK_COLUMNS + EXTRA_COLUMNS)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Phase 2 positions.csv is missing required columns")
        last_index = -1
        for index, rows in groupby(reader, key=lambda r: int(r["frame_index"])):
            if index <= last_index:
                raise ValueError("Phase 2 positions must be sorted by frame_index")
            last_index = index
            group = []
            for row in rows:
                item = dict(row)
                for k in ("frame_index", "segment_id", "track_id"):
                    item[k] = int(row[k])
                for k in ("timestamp_s", "detection_confidence", "team_vote_fraction",
                          "x1", "y1", "x2", "y2", "foot_x_px", "foot_y_px"):
                    item[k] = float(row[k])
                item["position_valid"] = str(row["position_valid"]).lower() == "true"
                item["calibration_valid"] = str(row["calibration_valid"]).lower() == "true"
                item["pitch_x_m"] = float(row["pitch_x_m"]) if row["pitch_x_m"] not in ("", None) else None
                item["pitch_y_m"] = float(row["pitch_y_m"]) if row["pitch_y_m"] not in ("", None) else None
                item["anchor_frame"] = int(row["anchor_frame"]) if row["anchor_frame"] not in ("", None) else None
                item["flow_inlier_ratio"] = (
                    float(row["flow_inlier_ratio"]) if row["flow_inlier_ratio"] not in ("", None) else None
                )
                for opt_k in ("speed_kmh", "distance_step_m", "observed_distance_m"):
                    if opt_k in row:
                        item[opt_k] = float(row[opt_k]) if row[opt_k] not in ("", None) else None
                group.append(item)
            yield index, group


class SpeedTrackingAnnotator(TrackingAnnotator):
    """Extends Phase 1 TrackingAnnotator to display instantaneous speed (km/h) and distance (m)."""

    def annotate_with_metrics(
        self,
        frame: np.ndarray,
        tracks: sv.Detections,
        rows: list[dict],
        frame_index: int,
        segment_id: int,
        team_a_control_pct: float | None = None,
    ) -> np.ndarray:
        scene = frame.copy()
        if len(tracks):
            roles = tracks.data["role"]
            teams = [r["team"] for r in rows]
            groups = [
                team if (role == "goalkeeper" and team in {"team_a", "team_b"})
                else (role if role in {"goalkeeper", "referee"} else team)
                for role, team in zip(roles, teams)
            ]
            color_indices = np.array([GROUPS.index(group) for group in groups], dtype=int)
            names = {
                "team_a": "Team A", "team_b": "Team B", "goalkeeper": "GK",
                "referee": "Ref", "other": "Other", "unknown": "Unknown",
            }
            labels = []
            for row, role, group in zip(rows, roles, groups):
                resolved_name = row.get("player_name")
                if resolved_name not in ("", None):
                    base = str(resolved_name)
                else:
                    prefix = "GK " if role == "goalkeeper" else ""
                    base = f"#{int(row['track_id'])} {prefix}{names[group]}"
                raw_speed = row.get("speed_kmh")
                speed = float(raw_speed) if raw_speed not in ("", None) else None
                if speed is not None and np.isfinite(speed):
                    base += f" | {speed:.1f} km/h"
                labels.append(base)
            scene = self.boxes.annotate(scene, tracks, custom_color_lookup=color_indices)
            label_tracks = tracks[np.arange(len(tracks))]
            for i, label in enumerate(labels):
                (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_tracks.xyxy[i, 0] = np.clip(label_tracks.xyxy[i, 0], 0, max(0, scene.shape[1] - width - 8))
                label_tracks.xyxy[i, 1] = min(scene.shape[0] - 1, max(label_tracks.xyxy[i, 1], 30 + height + 8))
            scene = self.labels.annotate(scene, label_tracks, labels=labels, custom_color_lookup=color_indices)

        header = f"Phase 3 | frame {frame_index} | segment {segment_id}"
        if team_a_control_pct is not None:
            header += f" | Pitch Control A: {team_a_control_pct:.1f}% B: {100.0 - team_a_control_pct:.1f}%"
        cv2.rectangle(scene, (0, 0), (min(scene.shape[1] - 1, 720), 30), (24, 24, 24), -1)
        cv2.putText(scene, header, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        return scene


class Phase3Pipeline:
    """Computes physical (speed/distance) and tactical (Voronoi/Convex Hull) analytics from a Phase 2 run."""

    def __init__(self, smoothing_window: int = 5, max_gap_s: float = 0.25, max_speed_mps: float = 12.0) -> None:
        if smoothing_window < 1 or max_gap_s <= 0 or max_speed_mps <= 0:
            raise ValueError("Invalid MovementEngine parameters")
        self.smoothing_window = smoothing_window
        self.max_gap_s = max_gap_s
        self.max_speed_mps = max_speed_mps

    def run(self, phase2: Path, output: Path, video_override: Path | None = None) -> Path:
        manifest = json.loads((phase2 / "run.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError("Phase 3 requires a completed Phase 2 run")
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
        movement = MovementEngine(
            window=self.smoothing_window,
            max_gap_s=self.max_gap_s,
            max_speed_mps=self.max_speed_mps,
        )
        spatial = SpatialEngine(length=length, width=width)
        radar = SpatialRadar(length=length, width=width)
        radar_info = VideoInfo(*radar.size, info.fps, count)
        annotator = SpeedTrackingAnnotator()

        run_meta = {
            "phase": 3,
            "status": "running",
            "phase2_run": str(phase2.resolve()),
            "source": str(video.resolve()),
            "start_frame": start,
            "frames_requested": count,
            "fps": info.fps,
            "pitch_size_m": [length, width],
            "coordinate_units": units,
            "smoothing_window": self.smoothing_window,
            "max_gap_s": self.max_gap_s,
            "max_speed_mps": self.max_speed_mps,
        }
        write_json(output / "run.json", run_meta)

        groups = position_groups(phase2 / "positions.csv")
        current_index, current_rows = next(groups, (None, []))
        while current_index is not None and current_index < start:
            current_index, current_rows = next(groups, (None, []))

        frames_done = 0
        control_frames = 0
        last_segment = None

        try:
            with (
                VideoReader(video) as reader,
                VideoWriter(output / "tactical_spatial.mp4", info) as writer,
                VideoWriter(output / "radar_spatial.mp4", radar_info) as radar_writer,
                (output / "metrics.csv").open("w", newline="", encoding="utf-8") as csv_file,
                (output / "tactical_frames.jsonl").open("w", encoding="utf-8") as tactical_log,
            ):
                csv_writer = csv.DictWriter(csv_file, fieldnames=PHASE3_COLUMNS, extrasaction="ignore")
                csv_writer.writeheader()

                for index, frame in reader.frames(start, count):
                    rows = current_rows if current_index == index else []
                    if rows:
                        current_index, current_rows = next(groups, (None, []))

                    time_s = index / info.fps
                    segment = rows[0]["segment_id"] if rows else (last_segment or 0)
                    if last_segment is not None and segment != last_segment:
                        movement.reset()
                    last_segment = segment

                    enriched = []
                    for row in rows:
                        move = movement.update(row, time_s, row.get("anchor_frame"))
                        item = {**row, **move}
                        enriched.append(item)
                        csv_writer.writerow(item)

                    tactical = spatial.update(enriched, time_s)
                    if tactical["control_valid"]:
                        control_frames += 1

                    tactical_log.write(
                        json.dumps({"frame_index": index, "timestamp_s": time_s, **tactical}, allow_nan=False) + "\n"
                    )

                    cal_status = enriched[0]["calibration_status"] if enriched else "no_observations"
                    radar_frame = radar.render_spatial(enriched, tactical, status=cal_status, units=units)

                    boxes = np.array([[r[k] for k in ("x1", "y1", "x2", "y2")] for r in enriched]).reshape(-1, 4)
                    detections = sv.Detections(
                        xyxy=boxes.astype(np.float32) if len(enriched) else np.empty((0, 4), np.float32),
                        confidence=np.array([r["detection_confidence"] for r in enriched], np.float32),
                        class_id=np.zeros(len(enriched), int),
                        tracker_id=np.array([r["track_id"] for r in enriched], int),
                        data={"role": np.array([r["role"] for r in enriched])},
                    )
                    annotated = annotator.annotate_with_metrics(
                        frame,
                        detections,
                        enriched,
                        index,
                        segment,
                        tactical.get("team_a_control_pct"),
                    )
                    writer.write(radar.overlay(annotated, radar_frame))
                    radar_writer.write(radar_frame)
                    frames_done += 1

            summary_rows = movement.summary()
            pd.DataFrame(
                summary_rows,
                columns=[
                    "segment_id", "track_id", "team",
                    "observed_distance_m", "valid_duration_s",
                    "mean_speed_kmh", "max_speed_kmh",
                ],
            ).to_csv(output / "player_summary.csv", index=False)

            run_meta.update(
                status="completed",
                frames_processed=frames_done,
                pitch_control_frames=control_frames,
                tracks_summarized=len(summary_rows),
            )
        except BaseException as error:
            run_meta.update(status="failed", error_type=type(error).__name__)
            write_json(output / "run.json", run_meta)
            raise
        finally:
            groups.close()

        write_json(output / "run.json", run_meta)
        return output
