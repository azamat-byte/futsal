"""Unit tests for Epic 2 (`PlayerIdentityManager`) and Epic 3 (`InPlayBoundaryFilter`)."""

from __future__ import annotations

import numpy as np

from futsal.match.spatial import MovementEngine, SpatialEngine
from futsal.reid_boundary import (
    InPlayBoundaryFilter,
    PlayerIdentityManager,
    SigLIPEmbedder,
    SubstitutionZone,
)


def test_player_identity_manager_stitches_broken_track_via_ocr() -> None:
    """When #10 Pito subs out (track 3) and later subs back in (track 11), both stitch to Team A:#10."""
    manager = PlayerIdentityManager()
    dummy_crop = np.full((128, 64, 3), (40, 220, 240), dtype=np.uint8)

    # Track 3 enters at frame 0 and gets 2 votes for #10
    manager.register_known_vote(3, "team_a", 10, frame_index=0, crop_bgr=dummy_crop)
    p1 = manager.register_known_vote(3, "team_a", 10, frame_index=3, crop_bgr=dummy_crop)
    assert p1 is not None
    assert p1.profile_id == "Team A:#10"
    assert p1.player_name == "Pito"
    assert p1.historical_tracker_ids == [3]

    # Track 3 leaves; Track 11 subs in at frame 308 and gets 2 votes for #10
    manager.register_known_vote(11, "team_a", 10, frame_index=308, crop_bgr=dummy_crop)
    p2 = manager.register_known_vote(11, "team_a", 10, frame_index=311, crop_bgr=dummy_crop)
    assert p2 is p1
    assert p2.active_tracker_id == 11
    assert p2.historical_tracker_ids == [3, 11]


def test_player_identity_manager_siglip_memory_bank_reid() -> None:
    """When a subbed-out player returns with an un-OCR'd track, SigLIP cosine similarity restores their profile."""
    # Custom deterministic embedding function based on mean BGR color
    embedder = SigLIPEmbedder(embed_fn=lambda crop: crop.mean(axis=(0, 1)).astype(np.float32))
    manager = PlayerIdentityManager(embedder=embedder, reid_similarity_threshold=0.95)

    pito_crop = np.full((80, 40, 3), (20, 200, 250), dtype=np.uint8)
    manager.register_known_vote(3, "team_a", 10, frame_index=0, crop_bgr=pito_crop)
    manager.register_known_vote(3, "team_a", 10, frame_index=3, crop_bgr=pito_crop)

    # Frame 310: Track 3 is gone, Track 99 appears with matching visual appearance
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:100, 30:70] = (20, 200, 250)
    rows = [{"track_id": 99, "team": "team_a", "role": "player", "x1": 30, "y1": 20, "x2": 70, "y2": 100}]
    enriched = manager.update_frame(frame, rows, frame_index=310)
    assert enriched[0]["jersey_number"] == 10
    assert enriched[0]["player_name"] == "Pito"
    assert enriched[0]["profile_id"] == "Team A:#10"
    assert manager.profiles[("Team A", 10)].historical_tracker_ids == [3, 99]


def test_in_play_boundary_filter_excludes_bench_and_substitution_zone() -> None:
    """Bench and substitution-zone coordinates set is_active=False and are excluded from Voronoi/Hulls/Distance."""
    bfilter = InPlayBoundaryFilter(exit_confirm_frames=1, enter_confirm_frames=1)
    homography = np.eye(3, dtype=np.float64)

    rows = [
        # Active Team A player in center court (20m, 10m)
        {
            "segment_id": 0, "track_id": 1, "team": "team_a", "role": "player",
            "x1": 19.0, "y1": 8.0, "x2": 21.0, "y2": 10.0,
            "pitch_x_m": 20.0, "pitch_y_m": 10.0, "position_valid": True,
        },
        # Active Team B player (25m, 12m)
        {
            "segment_id": 0, "track_id": 2, "team": "team_b", "role": "player",
            "x1": 24.0, "y1": 10.0, "x2": 26.0, "y2": 12.0,
            "pitch_x_m": 25.0, "pitch_y_m": 12.0, "position_valid": True,
        },
        # Bench player outside court (12m, -1.8m)
        {
            "segment_id": 0, "track_id": 99, "team": "team_a", "role": "player",
            "x1": 11.0, "y1": -3.0, "x2": 13.0, "y2": -1.8,
            "pitch_x_m": 12.0, "pitch_y_m": -1.8, "position_valid": True,
        },
        # Player in substitution corridor (12.5m, 0.05m)
        {
            "segment_id": 0, "track_id": 98, "team": "team_a", "role": "player",
            "x1": 12.0, "y1": -1.0, "x2": 13.0, "y2": 0.05,
            "pitch_x_m": 12.5, "pitch_y_m": 0.05, "position_valid": True,
        },
    ]

    filtered = bfilter.filter_frame_rows(rows, homography_img_to_pitch=homography)
    assert filtered[0]["is_active"] is True
    assert filtered[0]["boundary_status"] == "in_play"
    assert filtered[1]["is_active"] is True
    assert filtered[2]["is_active"] is False
    assert filtered[2]["boundary_status"] == "out_of_bounds_bench"
    assert filtered[3]["is_active"] is False
    assert filtered[3]["boundary_status"] == "substitution_zone"

    # Verify SpatialEngine excludes inactive bench/sub players from Voronoi cells
    spatial = SpatialEngine(length=40.0, width=20.0)
    tactical = spatial.update(filtered, time_s=1.0)
    cell_ids = [c["track_id"] for c in tactical["cells"]]
    assert cell_ids == [1, 2]

    # Verify MovementEngine does not accumulate distance for inactive bench players
    movement = MovementEngine(window=1)
    movement.update(filtered[2], time_s=1.0, anchor=0)
    moved_bench = {**filtered[2], "pitch_x_m": 14.0}
    res = movement.update(moved_bench, time_s=1.1, anchor=0)
    assert res["speed_kmh"] is None
    assert res["observed_distance_m"] == 0.0
