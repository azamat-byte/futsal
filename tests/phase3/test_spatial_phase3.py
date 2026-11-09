from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import pytest

from futsal.phase1.pipeline import TRACK_COLUMNS
from futsal.phase2.pipeline import EXTRA_COLUMNS
from futsal.phase3 import MovementEngine, Phase3Pipeline, SpatialEngine, SpatialRadar, polygon_area


def test_movement_engine_rolling_smoothing_and_distance():
    engine = MovementEngine(window=3, max_gap_s=0.25, max_speed_mps=15.0)
    # Player 1 moves along X at 4 m/s (14.4 km/h) across 5 frames at 10 FPS (dt=0.1s)
    speeds = []
    for idx in range(5):
        t = idx * 0.1
        row = {
            "segment_id": 0,
            "track_id": 7,
            "team": "team_a",
            "position_valid": True,
            "pitch_x_m": 10.0 + idx * 0.4,
            "pitch_y_m": 10.0,
        }
        out = engine.update(row, t, anchor=0)
        if out["speed_kmh"] is not None:
            speeds.append(out["speed_kmh"])

    assert len(speeds) == 4
    assert np.allclose(speeds, 14.4, atol=1e-2)
    summary = engine.summary()
    assert len(summary) == 1
    assert summary[0]["track_id"] == 7
    assert summary[0]["observed_distance_m"] == pytest.approx(1.2, rel=1e-2)
    assert summary[0]["mean_speed_kmh"] == pytest.approx(14.4, rel=1e-2)


def test_movement_engine_rejects_gaps_and_teleport_spikes():
    engine = MovementEngine(window=3, max_gap_s=0.25, max_speed_mps=12.0)
    base = {"segment_id": 0, "track_id": 1, "team": "team_a", "position_valid": True}
    engine.update({**base, "pitch_x_m": 5.0, "pitch_y_m": 5.0}, 0.0, anchor=0)
    # Impossible jump: 20m in 0.1s (200 m/s) -> rejected and resets window
    spike = engine.update({**base, "pitch_x_m": 25.0, "pitch_y_m": 5.0}, 0.1, anchor=0)
    assert spike["speed_kmh"] is None
    assert spike["observed_distance_m"] == 0.0


def test_spatial_engine_voronoi_partition_and_convex_hulls():
    spatial = SpatialEngine(length=40.0, width=20.0)
    # Symmetric layout: Team A on left half (x=10), Team B on right half (x=30)
    rows = [
        {"segment_id": 0, "track_id": 1, "role": "player", "team": "team_a", "position_valid": True, "pitch_x_m": 10.0, "pitch_y_m": 4.0},
        {"segment_id": 0, "track_id": 2, "role": "player", "team": "team_a", "position_valid": True, "pitch_x_m": 10.0, "pitch_y_m": 16.0},
        {"segment_id": 0, "track_id": 3, "role": "player", "team": "team_a", "position_valid": True, "pitch_x_m": 16.0, "pitch_y_m": 10.0},
        {"segment_id": 0, "track_id": 4, "role": "player", "team": "team_b", "position_valid": True, "pitch_x_m": 30.0, "pitch_y_m": 4.0},
        {"segment_id": 0, "track_id": 5, "role": "player", "team": "team_b", "position_valid": True, "pitch_x_m": 30.0, "pitch_y_m": 16.0},
        {"segment_id": 0, "track_id": 6, "role": "player", "team": "team_b", "position_valid": True, "pitch_x_m": 24.0, "pitch_y_m": 10.0},
    ]
    res = spatial.update(rows, 0.0)
    assert res["control_valid"] is True
    # Total area of all Voronoi cells must equal the 40x20m pitch area (800 m^2)
    total_area = sum(c["area_m2"] for c in res["cells"])
    assert total_area == pytest.approx(800.0, rel=1e-4)
    assert res["team_a_control_pct"] == pytest.approx(50.0, abs=1e-2)
    # Each team triangle has base 12m (y=4..16) and height 6m (x=10..16) -> area = 36 m^2
    hulls = {h["team"]: h for h in res["hulls"]}
    assert hulls["team_a"]["area_m2"] == pytest.approx(36.0, rel=1e-4)
    assert hulls["team_b"]["area_m2"] == pytest.approx(36.0, rel=1e-4)


def test_phase3_pipeline_end_to_end(tmp_path: Path):
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    for _ in range(4):
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
            "frames_processed": 4,
            "fps": 10.0,
            "pitch_size_m": [40.0, 20.0],
            "coordinate_units": "assumed_m",
        }),
        encoding="utf-8",
    )

    rows = []
    for f in range(4):
        for tid, team, x0, y0 in ((1, "team_a", 10.0 + f * 0.3, 8.0), (2, "team_b", 30.0 - f * 0.3, 12.0)):
            rows.append({
                "frame_index": f,
                "timestamp_s": f / 10.0,
                "segment_id": 0,
                "track_id": tid,
                "role": "player",
                "team": team,
                "team_vote_fraction": 1.0,
                "detection_confidence": 0.9,
                "x1": 50.0, "y1": 50.0, "x2": 80.0, "y2": 110.0,
                "foot_x_px": 65.0, "foot_y_px": 110.0,
                "pitch_x_m": x0, "pitch_y_m": y0,
                "position_valid": True,
                "position_status": "projected",
                "coordinate_units": "assumed_m",
                "calibration_valid": True,
                "calibration_status": "anchor",
                "anchor_frame": 0,
                "flow_inlier_ratio": None,
            })
    pd.DataFrame(rows, columns=TRACK_COLUMNS + EXTRA_COLUMNS).to_csv(phase2_dir / "positions.csv", index=False)

    out_dir = tmp_path / "phase3_out"
    Phase3Pipeline(smoothing_window=2).run(phase2_dir, out_dir)
    assert (out_dir / "metrics.csv").is_file()
    assert (out_dir / "player_summary.csv").is_file()
    assert (out_dir / "tactical_spatial.mp4").is_file()
    assert (out_dir / "radar_spatial.mp4").is_file()
    summary = pd.read_csv(out_dir / "player_summary.csv")
    assert len(summary) == 2
    assert (summary["observed_distance_m"] > 0).all()
