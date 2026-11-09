from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import pytest
import supervision as sv

from futsal.phase1.pipeline import TRACK_COLUMNS
from futsal.phase2.pipeline import EXTRA_COLUMNS
from futsal.phase4 import BallSelector, EventEngine, Phase4Pipeline, PossessionEngine, match_boxes


def test_hungarian_box_matching_for_pose():
    people = np.array([[10, 10, 50, 100], [100, 20, 140, 110]], float)
    poses = np.array([[102, 22, 138, 108], [12, 12, 48, 98]], float)
    matches = match_boxes(people, poses, min_iou=0.5)
    assert set(matches) == {(0, 1), (1, 0)}


def test_possession_engine_proximity_and_temporal_smoothing():
    engine = PossessionEngine(confirm_s=0.12)
    rows = [
        {"track_id": 1, "role": "player", "team": "team_a", "y1": 100.0, "y2": 200.0},
        {"track_id": 2, "role": "player", "team": "team_b", "y1": 100.0, "y2": 200.0},
    ]
    feet = {
        1: [[150.0, 195.0], [160.0, 196.0]],  # Player 1 ankles near ball (152, 196)
        2: [[300.0, 195.0]],
    }
    ball = {"x": 152.0, "y": 196.0}

    # Frame 0 (t=0.00s): within threshold (dist ~2px / 100px = 0.02 <= 0.35), but dwell time < 0.12s -> pending
    r0 = engine.update(0.00, ball, rows, feet)
    assert r0["state"] == "pending"
    assert r0["owner_id"] is None

    # Frame 1 (t=0.05s): still pending
    r1 = engine.update(0.05, ball, rows, feet)
    assert r1["state"] == "pending"

    # Frame 3 (t=0.15s >= 0.12s): confirmed controlled by player #1 (team_a)
    r2 = engine.update(0.15, ball, rows, feet)
    assert r2["state"] == "controlled"
    assert r2["owner_id"] == 1
    assert r2["team"] == "team_a"

    # Contested check: if player 2's ankle is also right next to the ball
    feet_contested = {1: [[150.0, 195.0]], 2: [[154.0, 195.0]]}
    r3 = engine.update(0.20, ball, rows, feet_contested)
    assert r3["state"] == "contested"
    assert r3["owner_id"] is None


def test_event_engine_pass_and_shot_detection():
    events = EventEngine(length=40.0, width=20.0)
    # Player 1 controls ball slowly near (15, 10), then kicks it fast toward Player 2 at (22, 10)
    ctrl_p1 = {"state": "controlled", "owner_id": 1, "team": "team_a"}
    free = {"state": "free_ball", "owner_id": None, "team": None}
    ctrl_p2 = {"state": "controlled", "owner_id": 2, "team": "team_a"}

    assert events.update(0.0, 0, ctrl_p1, [15.0, 10.0], True, 0) == []
    assert events.update(0.1, 0, ctrl_p1, [15.1, 10.0], True, 0) == []  # v = 1 m/s
    # Kick at t=0.2s: ball jumps 0.8m in 0.1s (v = 8 m/s, dv = 7 m/s)
    assert events.update(0.2, 0, free, [15.9, 10.0], True, 0) == []
    # Receiver #2 (same team) controls ball at (18.0, 10.0) -> pass_candidate!
    out_pass = events.update(0.4, 0, ctrl_p2, [18.0, 10.0], True, 0)
    assert len(out_pass) == 1
    assert out_pass[0]["type"] == "pass_candidate"
    assert out_pass[0]["source_id"] == 1
    assert out_pass[0]["receiver_id"] == 2

    # Now Player 2 shoots from (30, 10) toward right goal mouth (40, 10) at 20 m/s
    events.reset()
    events.update(1.0, 0, ctrl_p2, [30.0, 10.0], True, 0)
    events.update(1.1, 0, ctrl_p2, [30.1, 10.0], True, 0)
    shot = events.update(1.2, 0, free, [32.1, 10.0], True, 0)
    assert len(shot) == 1
    assert shot[0]["type"] == "shot_candidate"
    assert shot[0]["source_id"] == 2


class _FakeBallDetector:
    def __init__(self, ball_xy):
        self.ball_xy = ball_xy

    def detect(self, frame):
        bx, by = next(self.ball_xy)
        return sv.Detections(
            xyxy=np.array([[bx - 4, by - 4, bx + 4, by + 4]], np.float32),
            confidence=np.array([0.92], np.float32),
            class_id=np.array([0], int),
            data={"role": np.array(["ball"]), "class_name": np.array(["ball"])},
        )


class _FakePoseEstimator:
    def __init__(self, feet_per_frame):
        self.feet_per_frame = feet_per_frame

    def estimate(self, frame, tracks):
        return next(self.feet_per_frame)


def test_phase4_pipeline_end_to_end(tmp_path: Path):
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    for _ in range(6):
        writer.write(np.full((180, 320, 3), 40, dtype=np.uint8))
    writer.release()

    phase2_dir = tmp_path / "phase2_run"
    phase2_dir.mkdir()
    (phase2_dir / "run.json").write_text(
        json.dumps({
            "phase": 2,
            "status": "completed",
            "source": str(video_path),
            "start_frame": 0,
            "frames_processed": 6,
            "fps": 10.0,
            "pitch_size_m": [40.0, 20.0],
            "coordinate_units": "assumed_m",
        }),
        encoding="utf-8",
    )
    # Write identity scale homography: 10 px = 1 m
    H = np.diag([0.1, 0.1, 1.0]).tolist()
    with (phase2_dir / "calibration_frames.jsonl").open("w", encoding="utf-8") as f:
        for idx in range(6):
            f.write(json.dumps({"frame_index": idx, "image_to_pitch": H}) + "\n")

    rows = []
    for idx in range(6):
        for tid, x_px in ((1, 100.0), (2, 180.0)):
            rows.append({
                "frame_index": idx,
                "timestamp_s": idx * 0.1,
                "segment_id": 0,
                "track_id": tid,
                "role": "player",
                "team": "team_a",
                "team_vote_fraction": 1.0,
                "detection_confidence": 0.9,
                "x1": x_px - 10, "y1": 60.0, "x2": x_px + 10, "y2": 100.0,
                "foot_x_px": x_px, "foot_y_px": 100.0,
                "pitch_x_m": x_px * 0.1, "pitch_y_m": 10.0,
                "position_valid": True,
                "position_status": "projected",
                "coordinate_units": "assumed_m",
                "calibration_valid": True,
                "calibration_status": "anchor",
                "anchor_frame": 0,
                "flow_inlier_ratio": None,
            })
    pd.DataFrame(rows, columns=TRACK_COLUMNS + EXTRA_COLUMNS).to_csv(phase2_dir / "positions.csv", index=False)

    # Ball starts at player 1's feet (100, 100) for 3 frames (0, 1, 2), then accelerates to player 2 (180, 100)
    ball_coords = iter([(100.0, 100.0), (101.0, 100.0), (102.0, 100.0), (140.0, 100.0), (180.0, 100.0), (180.0, 100.0)])
    feet_seq = iter([
        {1: [[100.0, 100.0]], 2: [[180.0, 100.0]]},
        {1: [[101.0, 100.0]], 2: [[180.0, 100.0]]},
        {1: [[102.0, 100.0]], 2: [[180.0, 100.0]]},
        {1: [[102.0, 100.0]], 2: [[180.0, 100.0]]},
        {1: [[102.0, 100.0]], 2: [[180.0, 100.0]]},
        {1: [[102.0, 100.0]], 2: [[180.0, 100.0]]},
    ])

    out_dir = tmp_path / "phase4_out"
    Phase4Pipeline(
        _FakeBallDetector(ball_coords),
        _FakePoseEstimator(feet_seq),
        confirm_s=0.05,
    ).run(phase2_dir, out_dir)

    assert (out_dir / "tactical_events.mp4").is_file()
    assert (out_dir / "possession_tracks.csv").is_file()
    assert (out_dir / "events.csv").is_file()
    events_df = pd.read_csv(out_dir / "events.csv")
    assert len(events_df) == 1
    assert events_df.iloc[0]["type"] == "pass_candidate"
    assert events_df.iloc[0]["source_id"] == 1
    assert events_df.iloc[0]["receiver_id"] == 2
