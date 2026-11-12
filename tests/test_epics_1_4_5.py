"""Unit tests for Epic 1 (`BroadcastSceneManager`), Epic 4 (`TelemetrySQLiteStreamer`), and Epic 5 (`ShotEventDetector`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from futsal.broadcast_scene import BroadcastSceneManager
from futsal.telemetry_sqlite import ShotEventDetector, TelemetrySQLiteStreamer


class _DummyResetEngine:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_broadcast_scene_manager_pauses_on_closeup_and_resets_on_tactical_return() -> None:
    """Epic 1: Pauses on close-up/replay and triggers engine reset() when wide tactical angle returns."""
    tracker_eng = _DummyResetEngine()
    homography_eng = _DummyResetEngine()
    scene_mgr = BroadcastSceneManager(
        min_court_ratio=0.30,
        max_box_height_ratio=0.25,
        confirm_tactical_frames=2,
        engines_to_reset=[tracker_eng, homography_eng],
    )

    # Wide-angle blue futsal court frame (BGR ~ (210, 130, 30) -> HSV H~106, S~219, V~210)
    wide_frame = np.full((360, 640, 3), (210, 130, 30), dtype=np.uint8)
    wide_boxes = [[100.0, 150.0, 130.0, 200.0], [300.0, 160.0, 330.0, 210.0]]  # ~14% height

    d0 = scene_mgr.update(wide_frame, frame_index=0, boxes_xyxy=wide_boxes)
    assert d0.is_tactical is True
    assert d0.scene_label == "Tactical Wide Angle"
    assert d0.segment_id == 0

    # Close-up replay frame: dark crowd background + massive player box (>50% frame height)
    closeup_frame = np.full((360, 640, 3), (20, 20, 20), dtype=np.uint8)
    closeup_boxes = [[200.0, 40.0, 420.0, 320.0]]
    d1 = scene_mgr.update(closeup_frame, frame_index=1, boxes_xyxy=closeup_boxes)
    assert d1.is_tactical is False
    assert d1.scene_label == "Replay/Close-up"

    # Tactical wide angle returns for 2 confirmation frames -> resets engines and increments segment_id
    d2 = scene_mgr.update(wide_frame, frame_index=2, boxes_xyxy=wide_boxes)
    assert d2.is_tactical is False
    d3 = scene_mgr.update(wide_frame, frame_index=3, boxes_xyxy=wide_boxes)
    assert d3.is_tactical is True
    assert d3.reinitialized is True
    assert d3.segment_id == 1
    assert tracker_eng.reset_calls == 1
    assert homography_eng.reset_calls == 1


def test_sqlite_telemetry_streamer_groups_by_persistent_jersey_number(tmp_path: Path) -> None:
    """Epic 4: Groups across broken ByteTrack IDs by persistent jersey_number and excludes inactive bench rows."""
    db_file = tmp_path / "telemetry.db"
    with TelemetrySQLiteStreamer(db_file) as streamer:
        # Frame 1: #10 Pito on track_id=3 (in play, +4.5m, 28.0 km/h)
        streamer.stream_frame(
            frame_id=1,
            timestamp_s=0.04,
            rows=[
                {
                    "track_id": 3,
                    "team": "team_a",
                    "jersey_number": 10,
                    "player_name": "Pito",
                    "is_active": True,
                    "pitch_x_m": 18.0,
                    "pitch_y_m": 10.0,
                    "speed_kmh": 28.0,
                    "distance_step_m": 4.5,
                    "possession_state": "controlled",
                    "in_possession": True,
                }
            ],
        )
        # Frame 2: #10 Pito after substitution on track_id=11 (in play, +5.5m, 34.2 km/h)
        streamer.stream_frame(
            frame_id=310,
            timestamp_s=12.40,
            rows=[
                {
                    "track_id": 11,
                    "team": "team_a",
                    "jersey_number": 10,
                    "player_name": "Pito",
                    "is_active": True,
                    "pitch_x_m": 24.0,
                    "pitch_y_m": 11.0,
                    "speed_kmh": 34.2,
                    "distance_step_m": 5.5,
                    "possession_state": "controlled",
                    "in_possession": False,
                },
                # Inactive bench row for #10 Pito (should be excluded from distance/speed)
                {
                    "track_id": 99,
                    "team": "team_a",
                    "jersey_number": 10,
                    "player_name": "Pito",
                    "is_active": False,
                    "pitch_x_m": 12.0,
                    "pitch_y_m": -1.5,
                    "speed_kmh": 50.0,
                    "distance_step_m": 100.0,
                },
            ],
        )

        summary = streamer.query_player_match_totals()
        assert len(summary) == 1
        row = summary[0]
        assert row["team"] == "Team A"
        assert row["jersey_number"] == 10
        assert row["player_name"] == "Pito"
        assert row["tracker_ids"] == "10"
        assert row["total_distance_m"] == 10.0
        assert row["top_sprint_speed_kmh"] == 34.2
        assert row["possession_frames"] == 1


def test_shot_event_detector_logs_shot_made_to_sqlite(tmp_path: Path) -> None:
    """Epic 5: High-acceleration strike near penalty box followed by goal-net intersection logs 'Shot Made'."""
    db_file = tmp_path / "shots.db"
    with TelemetrySQLiteStreamer(db_file) as streamer:
        detector = ShotEventDetector(db_streamer=streamer)

        # Frame 10: #10 Pito strikes ball near right penalty box (x=32.5m, y=10.0m) at 24.5 km/h
        ev1 = detector.update(
            frame_id=10,
            timestamp_s=1.00,
            rows=[
                {
                    "track_id": 3,
                    "team": "team_a",
                    "jersey_number": 10,
                    "player_name": "Pito",
                    "is_active": True,
                    "position_valid": True,
                    "pitch_x_m": 32.5,
                    "pitch_y_m": 10.0,
                    "speed_kmh": 24.5,
                }
            ],
            ball_pitch_m=(33.0, 10.0),
            possession={"state": "controlled", "owner_id": 3, "team": "team_a"},
        )
        assert ev1 is None

        # Frame 15 (t=1.20s): Ball rockets into the right goal net (x=39.6m, y=10.1m)
        ev2 = detector.update(
            frame_id=15,
            timestamp_s=1.20,
            rows=[],
            ball_pitch_m=(39.6, 10.1),
            possession={"state": "free", "owner_id": None, "team": None},
        )
        assert ev2 is not None
        assert ev2["event_type"] == "Shot Made"
        assert ev2["team"] == "Team A"
        assert ev2["jersey_number"] == 10
        assert ev2["player_name"] == "Pito"

        logged = streamer.query_events("Shot Made")
        assert len(logged) == 1
        assert logged[0]["jersey_number"] == 10
        assert logged[0]["goal_x_m"] == 39.6
