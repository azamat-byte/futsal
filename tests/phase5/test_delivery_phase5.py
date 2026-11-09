from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import pytest

from futsal.phase1.pipeline import TRACK_COLUMNS
from futsal.phase2.pipeline import EXTRA_COLUMNS
from futsal.phase5 import BroadcastCompositor, Phase5Pipeline, build_match_report, compute_heatmaps


def test_compute_heatmaps_and_match_report():
    tracks_df = pd.DataFrame([
        {
            "frame_index": 0, "track_id": 1, "team": "team_a",
            "position_valid": True, "pitch_x_m": 10.0, "pitch_y_m": 5.0,
            "possession_state": "controlled", "possession_team": "team_a",
        },
        {
            "frame_index": 1, "track_id": 2, "team": "team_b",
            "position_valid": True, "pitch_x_m": 30.0, "pitch_y_m": 15.0,
            "possession_state": "controlled", "possession_team": "team_b",
        },
    ])
    summary_df = pd.DataFrame([
        {"segment_id": 0, "track_id": 1, "team": "team_a", "observed_distance_m": 12.5},
        {"segment_id": 0, "track_id": 2, "team": "team_b", "observed_distance_m": 8.0},
    ])
    events_df = pd.DataFrame([
        {"event_id": 1, "type": "pass_candidate", "team": "team_a"},
        {"event_id": 2, "type": "shot_candidate", "team": "team_b"},
    ])
    heatmaps = compute_heatmaps(tracks_df)
    assert heatmaps["team_a"].sum() == 1.0
    assert heatmaps["team_b"].sum() == 1.0
    assert heatmaps["player_1"].sum() == 1.0

    report = build_match_report(
        tracks_df,
        summary_df,
        events_df,
        tactical_records=[{"team_a_control_pct": 55.0}],
    )
    assert report["team_a_possession_pct"] == pytest.approx(50.0)
    assert report["team_a_pitch_control_pct"] == pytest.approx(55.0)
    assert report["team_a_distance_m"] == pytest.approx(12.5)
    assert report["passes_detected"] == 1
    assert report["shots_detected"] == 1


def test_phase5_pipeline_end_to_end(tmp_path: Path):
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (480, 270))
    for _ in range(4):
        writer.write(np.full((270, 480, 3), 35, dtype=np.uint8))
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
        for tid, team, x_m, y_m in ((1, "team_a", 12.0 + f * 0.4, 8.0), (2, "team_b", 28.0 - f * 0.4, 12.0)):
            rows.append({
                "frame_index": f,
                "timestamp_s": f / 10.0,
                "segment_id": 0,
                "track_id": tid,
                "role": "player",
                "team": team,
                "team_vote_fraction": 1.0,
                "detection_confidence": 0.95,
                "x1": 60.0, "y1": 60.0, "x2": 90.0, "y2": 130.0,
                "foot_x_px": 75.0, "foot_y_px": 130.0,
                "pitch_x_m": x_m, "pitch_y_m": y_m,
                "position_valid": True,
                "position_status": "projected",
                "coordinate_units": "assumed_m",
                "calibration_valid": True,
                "calibration_status": "anchor",
                "anchor_frame": 0,
                "flow_inlier_ratio": None,
            })
    pd.DataFrame(rows, columns=TRACK_COLUMNS + EXTRA_COLUMNS).to_csv(phase2_dir / "positions.csv", index=False)

    out_dir = tmp_path / "phase5_out"
    Phase5Pipeline(radar_opacity=0.8, smoothing_window=2).run(phase2_dir, out_dir)

    assert (out_dir / "broadcast_tactical.mp4").is_file()
    assert (out_dir / "dashboard_tracks.csv").is_file()
    assert (out_dir / "player_summary.csv").is_file()
    assert (out_dir / "heatmaps.npz").is_file()
    assert (out_dir / "match_report.json").is_file()
    report = json.loads((out_dir / "match_report.json").read_text(encoding="utf-8"))
    assert report["players_tracked"] == 2
    assert report["team_a_distance_m"] > 0
