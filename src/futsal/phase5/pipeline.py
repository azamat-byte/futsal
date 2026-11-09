"""Phase 5 Pipeline: Broadcast tactical video export, player heatmaps, and dashboard report generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from futsal.match.spatial import MovementEngine, SpatialEngine
from futsal.phase1.pipeline import write_json
from futsal.phase1.video import VideoReader, VideoWriter
from futsal.phase3.pipeline import position_groups
from futsal.phase3.radar import SpatialRadar
from .compositor import BroadcastCompositor, mux_h264_with_audio

LOG = logging.getLogger(__name__)


def compute_heatmaps(
    tracks_df: pd.DataFrame,
    length: float = 40.0,
    width: float = 20.0,
    bins_x: int = 40,
    bins_y: int = 20,
) -> dict[str, np.ndarray]:
    """Compute 2D pitch occupancy histograms for Team A, Team B, and individual track IDs."""
    heatmaps: dict[str, np.ndarray] = {
        "team_a": np.zeros((bins_y, bins_x), dtype=np.float32),
        "team_b": np.zeros((bins_y, bins_x), dtype=np.float32),
    }
    if tracks_df.empty or "position_valid" not in tracks_df.columns:
        return heatmaps

    valid = tracks_df[
        tracks_df["position_valid"].astype(str).str.lower().isin(["true", "1"])
        & tracks_df["pitch_x_m"].notna()
        & tracks_df["pitch_y_m"].notna()
    ]
    for team in ("team_a", "team_b"):
        sub = valid[valid["team"] == team]
        if len(sub):
            h, _, _ = np.histogram2d(
                sub["pitch_y_m"].to_numpy(float),
                sub["pitch_x_m"].to_numpy(float),
                bins=[bins_y, bins_x],
                range=[[0.0, width], [0.0, length]],
            )
            heatmaps[team] = h.astype(np.float32)

    for tid, group in valid.groupby("track_id"):
        h, _, _ = np.histogram2d(
            group["pitch_y_m"].to_numpy(float),
            group["pitch_x_m"].to_numpy(float),
            bins=[bins_y, bins_x],
            range=[[0.0, width], [0.0, length]],
        )
        heatmaps[f"player_{int(tid)}"] = h.astype(np.float32)
    return heatmaps


def build_match_report(
    tracks_df: pd.DataFrame,
    player_summary_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    tactical_records: list[dict] | None = None,
) -> dict:
    """Aggregate match statistics (Possession %, Pitch Control %, Distance Covered, Events)."""
    tactical_records = tactical_records or []
    ctrl_vals = [r["team_a_control_pct"] for r in tactical_records if r.get("team_a_control_pct") is not None]
    mean_ctrl_a = float(np.mean(ctrl_vals)) if ctrl_vals else None

    poss_a_pct = poss_b_pct = None
    if "possession_state" in tracks_df.columns and "possession_team" in tracks_df.columns:
        frame_poss = tracks_df[["frame_index", "possession_state", "possession_team"]].drop_duplicates("frame_index")
        controlled = frame_poss[
            (frame_poss["possession_state"] == "controlled")
            & (frame_poss["possession_team"].isin(["team_a", "team_b"]))
        ]
        if len(controlled):
            count_a = int((controlled["possession_team"] == "team_a").sum())
            poss_a_pct = 100.0 * count_a / len(controlled)
            poss_b_pct = 100.0 - poss_a_pct

    passes = shots = 0
    if events_df is not None and not events_df.empty and "type" in events_df.columns:
        passes = int((events_df["type"] == "pass_candidate").sum())
        shots = int((events_df["type"] == "shot_candidate").sum())

    total_dist_a = total_dist_b = 0.0
    if not player_summary_df.empty and "observed_distance_m" in player_summary_df.columns:
        valid_dist = player_summary_df.dropna(subset=["observed_distance_m"])
        total_dist_a = float(valid_dist[valid_dist["team"] == "team_a"]["observed_distance_m"].sum())
        total_dist_b = float(valid_dist[valid_dist["team"] == "team_b"]["observed_distance_m"].sum())

    return {
        "team_a_possession_pct": poss_a_pct,
        "team_b_possession_pct": poss_b_pct,
        "team_a_pitch_control_pct": mean_ctrl_a,
        "team_b_pitch_control_pct": (100.0 - mean_ctrl_a) if mean_ctrl_a is not None else None,
        "team_a_distance_m": total_dist_a,
        "team_b_distance_m": total_dist_b,
        "passes_detected": passes,
        "shots_detected": shots,
        "players_tracked": int(len(player_summary_df)),
    }


def _load_phase4_frame_log(phase4_dir: Path | None) -> dict[int, dict]:
    if phase4_dir is None or not (phase4_dir / "possession_frames.jsonl").is_file():
        return {}
    lookup: dict[int, dict] = {}
    with (phase4_dir / "possession_frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                lookup[int(rec["frame_index"])] = rec
    return lookup


class Phase5Pipeline:
    """Produces final H.264 broadcast tactical video, player heatmaps, and dashboard analytics bundle."""

    def __init__(self, radar_opacity: float = 0.85, smoothing_window: int = 5) -> None:
        self.compositor = BroadcastCompositor(radar_opacity=radar_opacity)
        self.smoothing_window = smoothing_window

    def run(
        self,
        phase2: Path,
        output: Path,
        phase3: Path | None = None,
        phase4: Path | None = None,
        video_override: Path | None = None,
    ) -> Path:
        manifest = json.loads((phase2 / "run.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError("Phase 5 requires a completed Phase 2 run")
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

        csv_path = (
            phase4 / "possession_tracks.csv"
            if (phase4 and (phase4 / "possession_tracks.csv").is_file())
            else (
                phase3 / "metrics.csv"
                if (phase3 and (phase3 / "metrics.csv").is_file())
                else (phase2 / "positions.csv")
            )
        )
        groups = position_groups(csv_path)
        current_index, current_rows = next(groups, (None, []))
        while current_index is not None and current_index < start:
            current_index, current_rows = next(groups, (None, []))

        phase4_frames = _load_phase4_frame_log(phase4)
        movement = MovementEngine(window=self.smoothing_window)
        spatial = SpatialEngine(length=length, width=width)
        radar = SpatialRadar(length=length, width=width)

        raw_video_path = output / "raw_broadcast.mp4"
        final_video_path = output / "broadcast_tactical.mp4"
        tactical_records: list[dict] = []
        enriched_all: list[dict] = []
        controlled_counts = {"team_a": 0, "team_b": 0}
        frames_done = 0
        last_segment = None

        run_meta = {
            "phase": 5,
            "status": "running",
            "phase2_run": str(phase2.resolve()),
            "phase3_run": str(phase3.resolve()) if phase3 else None,
            "phase4_run": str(phase4.resolve()) if phase4 else None,
            "source": str(video.resolve()),
            "start_frame": start,
            "frames_requested": count,
            "fps": info.fps,
        }
        write_json(output / "run.json", run_meta)

        try:
            with VideoReader(video) as reader, VideoWriter(raw_video_path, info) as writer:
                for index, frame in reader.frames(start, count):
                    rows = current_rows if current_index == index else []
                    if rows:
                        current_index, current_rows = next(groups, (None, []))

                    time_s = index / info.fps
                    segment = rows[0]["segment_id"] if rows else (last_segment or 0)
                    if last_segment is not None and segment != last_segment:
                        movement.reset()
                        self.compositor.reset()
                    last_segment = segment

                    p4 = phase4_frames.get(index, {})
                    ball = p4.get("ball")
                    ball_pitch = p4.get("ball_pitch_m")
                    possession = p4.get("possession") or {"state": "unknown", "owner_id": None, "team": None}
                    feet = {int(k): v for k, v in (p4.get("ankles_px") or {}).items()}
                    emitted = p4.get("events") or []

                    if possession.get("state") == "controlled" and possession.get("team") in controlled_counts:
                        controlled_counts[possession["team"]] += 1
                    tot_poss = sum(controlled_counts.values())
                    poss_a_pct = (100.0 * controlled_counts["team_a"] / tot_poss) if tot_poss else None

                    enriched = []
                    for row in rows:
                        move = movement.update(row, time_s, row.get("anchor_frame"))
                        item = {
                            **row,
                            **move,
                            "possession_state": possession.get("state"),
                            "possession_owner_id": possession.get("owner_id"),
                            "possession_team": possession.get("team"),
                        }
                        enriched.append(item)
                        enriched_all.append(item)

                    tactical = spatial.update(enriched, time_s)
                    tactical_records.append({"frame_index": index, "timestamp_s": time_s, **tactical})

                    cal_status = enriched[0]["calibration_status"] if enriched else "no_observations"
                    radar_img = radar.render_spatial(enriched, tactical, status=cal_status, units=units)
                    if ball_pitch is not None:
                        bp = radar.point(ball_pitch[0], ball_pitch[1])
                        cv2.circle(radar_img, bp, 5, (0, 215, 255), -1, cv2.LINE_AA)
                        cv2.circle(radar_img, bp, 6, (15, 15, 15), 1, cv2.LINE_AA)

                    boxes = np.array([[r[k] for k in ("x1", "y1", "x2", "y2")] for r in enriched]).reshape(-1, 4)
                    detections = sv.Detections(
                        xyxy=boxes.astype(np.float32) if len(enriched) else np.empty((0, 4), np.float32),
                        confidence=np.array([r["detection_confidence"] for r in enriched], np.float32),
                        class_id=np.zeros(len(enriched), int),
                        tracker_id=np.array([r["track_id"] for r in enriched], int),
                        data={"role": np.array([r["role"] for r in enriched])},
                    )

                    scene = self.compositor.render_frame(
                        frame=frame,
                        tracks=detections,
                        rows=enriched,
                        feet=feet,
                        ball=ball,
                        possession=possession,
                        emitted_events=emitted,
                        radar_img=radar_img,
                        frame_index=index,
                        segment_id=segment,
                        team_a_control_pct=tactical.get("team_a_control_pct"),
                        team_a_possession_pct=poss_a_pct,
                    )
                    writer.write(scene)
                    frames_done += 1

            h264_ok = mux_h264_with_audio(raw_video_path, video, final_video_path, start, frames_done, info.fps)
            if raw_video_path.exists() and final_video_path.exists():
                raw_video_path.unlink(missing_ok=True)

            tracks_df = pd.DataFrame(enriched_all)
            tracks_df.to_csv(output / "dashboard_tracks.csv", index=False)

            player_summary_df = pd.DataFrame(
                movement.summary(),
                columns=[
                    "segment_id", "track_id", "team",
                    "observed_distance_m", "valid_duration_s",
                    "mean_speed_kmh", "max_speed_kmh",
                ],
            )
            player_summary_df.to_csv(output / "player_summary.csv", index=False)

            events_df = None
            if phase4 and (phase4 / "events.csv").is_file():
                events_df = pd.read_csv(phase4 / "events.csv")
                events_df.to_csv(output / "events.csv", index=False)
            else:
                events_df = pd.DataFrame(
                    columns=["event_id", "type", "time_s", "segment_id", "source_id", "receiver_id", "team", "evidence"]
                )
                events_df.to_csv(output / "events.csv", index=False)

            heatmaps = compute_heatmaps(tracks_df, length=length, width=width)
            np.savez_compressed(output / "heatmaps.npz", **heatmaps)

            report = build_match_report(tracks_df, player_summary_df, events_df, tactical_records)
            report["h264_encoded"] = h264_ok
            write_json(output / "match_report.json", report)

            run_meta.update(status="completed", frames_processed=frames_done, **report)
        except BaseException as error:
            run_meta.update(status="failed", error_type=type(error).__name__)
            write_json(output / "run.json", run_meta)
            raise
        finally:
            groups.close()

        write_json(output / "run.json", run_meta)
        return output
