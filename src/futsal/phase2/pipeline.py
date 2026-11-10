"""Reuse Phase 1 observations; no player inference or API credentials are needed."""

import csv
import json
import logging
from itertools import groupby
from pathlib import Path

import numpy as np
import supervision as sv

from futsal.phase1.annotations import TrackingAnnotator
from futsal.phase1.pipeline import TRACK_COLUMNS, write_json
from futsal.phase1.tracking import SceneCutDetector
from futsal.phase1.video import VideoInfo, VideoReader, VideoWriter
from .calibration import Calibration, CameraCalibration
from .geometry import project
from .radar import PitchRadar

LOG = logging.getLogger(__name__)
EXTRA_COLUMNS = ["pitch_x_m", "pitch_y_m", "position_valid", "position_status", "coordinate_units",
                 "calibration_valid", "calibration_status", "anchor_frame", "flow_inlier_ratio"]


def stabilize_teams_and_goalkeepers(rows_by_frame: list[tuple[int, list[dict]]]) -> list[tuple[int, list[dict]]]:
    """Lock each player's majority team and cluster goalkeepers to their defending side's team."""
    from collections import Counter

    votes: dict[tuple[int, int], Counter] = {}
    gk_xs: dict[tuple[int, int], list[float]] = {}
    team_xs: dict[tuple[int, str], list[float]] = {}

    for _, group in rows_by_frame:
        for r in group:
            seg = int(r["segment_id"])
            tid = int(r["track_id"])
            role = r["role"]
            team = r["team"]
            if role == "player" and team in {"team_a", "team_b"}:
                votes.setdefault((seg, tid), Counter())[team] += 1
                team_xs.setdefault((seg, team), []).append(float(r["foot_x_px"]))
            elif role == "goalkeeper":
                gk_xs.setdefault((seg, tid), []).append(float(r["foot_x_px"]))

    locked_team: dict[tuple[int, int], str] = {}
    for key, counter in votes.items():
        total_votes = sum(counter.values())
        a_votes = counter.get("team_a", 0)
        b_votes = counter.get("team_b", 0)
        # On a blue futsal court, yellow jerseys (team_a) only trigger on true Team A players,
        # whereas hunched-over players can pick up blue court background (team_b).
        if a_votes >= 15 and a_votes / max(1, total_votes) >= 0.20:
            locked_team[key] = "team_a"
        elif b_votes >= 3 and a_votes == 0:
            locked_team[key] = "team_b"
        else:
            winner, count = counter.most_common(1)[0]
            if count >= 3:
                locked_team[key] = winner

    # Assign goalkeepers to whichever team defends their half of the court (left vs right)
    for (seg, tid), xs in gk_xs.items():
        if (seg, tid) in locked_team:
            continue
        mean_gk_x = float(np.mean(xs))
        mean_a = float(np.mean(team_xs.get((seg, "team_a"), [960.0])))
        mean_b = float(np.mean(team_xs.get((seg, "team_b"), [960.0])))
        left_team = "team_a" if mean_a <= mean_b else "team_b"
        right_team = "team_b" if left_team == "team_a" else "team_a"
        locked_team[(seg, tid)] = left_team if mean_gk_x < 960.0 else right_team

    for _, group in rows_by_frame:
        for r in group:
            key = (int(r["segment_id"]), int(r["track_id"]))
            if key in locked_team:
                r["team"] = locked_team[key]
                r["team_vote_fraction"] = max(float(r.get("team_vote_fraction", 0.0)), 0.95)
    return rows_by_frame


def track_groups(path):
    """Read and stabilize Phase 1 observations, locking player teams and clustering goalkeepers."""
    raw_groups = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(TRACK_COLUMNS) <= set(reader.fieldnames or []):
            raise ValueError("Phase 1 tracks.csv is missing required columns")
        last_index = -1
        for index, rows in groupby(reader, key=lambda row: int(row["frame_index"])):
            if index <= last_index:
                raise ValueError("Phase 1 observations must be sorted by source frame")
            last_index = index
            group = []
            for row in rows:
                converted = dict(row)
                for name in ("frame_index", "segment_id", "track_id"):
                    converted[name] = int(row[name])
                for name in ("timestamp_s", "detection_confidence", "team_vote_fraction", "x1", "y1", "x2", "y2",
                             "foot_x_px", "foot_y_px"):
                    converted[name] = float(row[name])
                    if not np.isfinite(converted[name]):
                        raise ValueError("Nonfinite Phase 1 observation")
                group.append(converted)
            if len({r["track_id"] for r in group}) != len(group):
                raise ValueError("Duplicate track identity in a source frame")
            raw_groups.append((index, group))
    return iter(stabilize_teams_and_goalkeepers(raw_groups))


class Phase2Pipeline:
    def run(self, phase1: Path, calibration_path: Path, output: Path, max_age_s=2.0,
            video_override: Path | None = None, max_frames: int | None = None, start_frame: int | None = None):
        manifest = json.loads((phase1 / "run.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError("Phase 2 requires a completed Phase 1 run")
        video = video_override or Path(manifest["source"])
        start, count = manifest["start_frame"], manifest["frames_processed"]
        if start_frame is not None:
            if not start <= start_frame < start+count:
                raise ValueError("start_frame must lie inside the completed Phase 1 range")
            count -= start_frame-start
            start = start_frame
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            count = min(count, max_frames)
        with VideoReader(video) as reader:
            info = reader.info
        if (info.width != manifest["video"]["width"] or info.height != manifest["video"]["height"]
                or abs(info.fps - manifest["video"]["fps"]) > .001
                or info.frame_count != manifest["video"]["frame_count"]):
            raise ValueError("Source video no longer matches Phase 1 metadata")
        # Phase 1 predates source hashes. A relocated source must still have the
        # same content as the source bound to the reviewed calibration file.
        calibration = Calibration.load(calibration_path, video, info)
        if not any(start <= i < start+count for i in calibration.anchors):
            raise ValueError("No calibration anchor falls inside the Phase 1 frame range")
        camera = CameraCalibration(calibration, info.fps, max_age_s)
        length, width = calibration.raw.get("pitch_size_m", [40, 20])
        radar = PitchRadar(length, width)
        radar_info = VideoInfo(*radar.size, info.fps, count)
        units = "m" if calibration.raw.get("scale_verified", False) else "assumed_m"
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        result = {"phase": 2, "status": "running", "phase1_run": str(phase1.resolve()),
                  "source": str(video.resolve()), "source_sha256": calibration.raw["source_sha256"],
                  "start_frame": start, "frames_requested": count, "fps": info.fps,
                  "pitch_size_m": [length, width], "coordinate_units": units,
                  "max_anchor_age_s": max_age_s, "audio": False}
        write_json(output / "run.json", result)
        write_json(output / "calibration.json", calibration.raw)
        try:
            frames, valid, observations, projected_count = self._render(
                phase1, video, output, start, count, info, radar_info, camera, radar, units, manifest)
            result.update(status="completed", frames_processed=frames, calibrated_frames=valid,
                          observations=observations, projected_observations=projected_count,
                          calibration_coverage=valid/frames if frames else 0)
        except BaseException as error:
            result.update(status="failed", error_type=type(error).__name__)
            write_json(output / "run.json", result)
            raise
        write_json(output / "run.json", result)
        return output

    def _render(self, phase1, video, output, start, count, info, radar_info, camera, radar, units, manifest):
        groups = track_groups(phase1 / "tracks.csv")
        current_index, current_rows = next(groups, (None, []))
        while current_index is not None and current_index < start:
            current_index, current_rows = next(groups, (None, []))
        annotator = TrackingAnnotator()
        cut = SceneCutDetector(manifest["config"]["tracking"].get("cut_threshold", .65))
        manual_cuts = set(manifest.get("manual_cut_frames", []))
        last_segment = None
        frames = valid = observations = projected_count = 0
        try:
            with (VideoReader(video) as reader,
                  VideoWriter(output / "tactical.mp4", info) as writer,
                  VideoWriter(output / "radar.mp4", radar_info) as radar_writer,
                  (output / "positions.csv").open("w", newline="", encoding="utf-8") as csv_file,
                  (output / "calibration_frames.jsonl").open("w", encoding="utf-8") as audit):
                csv_writer = csv.DictWriter(csv_file, fieldnames=TRACK_COLUMNS + EXTRA_COLUMNS, extrasaction="ignore")
                csv_writer.writeheader()
                for index, frame in reader.frames(start, count):
                    if current_index is not None and current_index < index:
                        raise ValueError("Phase 1 observations lie outside the manifest's frame range")
                    rows = current_rows if current_index == index else []
                    if rows:
                        current_index, current_rows = next(groups, (None, []))
                    segments = {r["segment_id"] for r in rows}
                    if len(segments) > 1:
                        raise ValueError("Multiple Phase 1 segments in the same frame")
                    segment = next(iter(segments), last_segment)
                    changed = cut.update(frame) or index in manual_cuts
                    if last_segment is not None and segment != last_segment:
                        changed = True
                    if segment is not None:
                        last_segment = segment
                    boxes = np.array([[r[k] for k in ("x1", "y1", "x2", "y2")] for r in rows]).reshape(-1, 4)
                    matrix, state = camera.update(frame, index, boxes, cut=changed)
                    enriched = []
                    for row in rows:
                        if abs(row["timestamp_s"] - index/info.fps) > .001:
                            raise ValueError("Phase 1 timestamp differs from source FPS")
                        item = {**row, **state, "pitch_x_m": None, "pitch_y_m": None,
                                "position_valid": False, "position_status": "no_calibration", "coordinate_units": units}
                        if matrix is not None:
                            try:
                                x, y = project(matrix, [[row["foot_x_px"], row["foot_y_px"]]])[0]
                                if 0 <= x <= radar.length and 0 <= y <= radar.width:
                                    item.update(pitch_x_m=float(x), pitch_y_m=float(y),
                                                position_valid=True, position_status="projected")
                                else:
                                    item["position_status"] = "outside_pitch"
                            except ValueError:
                                item["position_status"] = "projection_at_horizon"
                        enriched.append(item)
                        csv_writer.writerow(item)
                    observations += len(enriched)
                    projected_count += sum(r["position_valid"] for r in enriched)
                    audit.write(json.dumps({"frame_index": index, **state,
                                            "image_to_pitch": matrix.tolist() if matrix is not None else None},
                                           allow_nan=False) + "\n")
                    view = radar.render(enriched, state["calibration_status"], units)
                    detections = sv.Detections(
                        xyxy=boxes.astype(np.float32), confidence=np.array([r["detection_confidence"] for r in rows]),
                        class_id=np.zeros(len(rows), int), tracker_id=np.array([r["track_id"] for r in rows], int),
                        data={"role": np.array([r["role"] for r in rows])},
                    )
                    annotated = annotator.annotate(frame, detections, [r["team"] for r in rows], index, segment or 0)
                    writer.write(radar.overlay(annotated, view))
                    radar_writer.write(view)
                    valid += int(matrix is not None)
                    frames += 1
                    if frames % 100 == 0:
                        LOG.info("Rendered %d frames; calibrated %d", frames, valid)
            if frames != count:
                raise OSError("Source ended before the Phase 1 frame range")
            for name in ("tactical.mp4", "radar.mp4"):
                with VideoReader(output / name) as check:
                    if check.info.frame_count != count or not next(check.frames(limit=1), None):
                        raise OSError(f"Video verification failed: {name}")
        finally:
            if hasattr(groups, "close"):
                groups.close()
        return frames, valid, observations, projected_count
