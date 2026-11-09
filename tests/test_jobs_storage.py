import pandas as pd
import pytest

from futsal.config import RunConfig
from futsal.jobs import create_job, claim_job, update_job, save_correction, corrections
from futsal.storage import TrackWriter, load_tracks


def test_atomic_claim_and_new_attempt(tmp_path):
    job = create_job("video.mp4", RunConfig(), tmp_path, launch=False)
    first = claim_job(tmp_path, job)
    with pytest.raises(ValueError):
        claim_job(tmp_path, job)
    update_job(tmp_path, job, state="queued")
    second = claim_job(tmp_path, job)
    assert first["run_path"] != second["run_path"]
    assert second["attempt"] == 2


def test_corrections_audited(tmp_path):
    save_correction(tmp_path, "segments", 0, "replay")
    assert corrections(tmp_path)["segments"]["0"] == "replay"
    assert (tmp_path / "review_history.jsonl").exists()


def test_parquet_nulls_survive(tmp_path):
    writer = TrackWriter(tmp_path / "tracks")
    writer.add([{"frame_index": 1, "time_s": 0.1, "court_x": None, "tracklet_id": "s0p1"}])
    writer.flush()
    result = load_tracks(tmp_path)
    assert pd.isna(result.court_x.iloc[0])


@pytest.mark.parametrize("max_rows", [None, 2])
def test_windowed_tracks_filter_before_projection_and_limit(tmp_path, max_rows):
    writer = TrackWriter(tmp_path / "tracks")
    writer.add([{"time_s": float(i), "tracklet_id": f"p{i}"} for i in range(4)])
    writer.flush()
    writer.add([{"time_s": float(i), "tracklet_id": f"p{i}"} for i in range(4, 8)])
    writer.flush()
    result = load_tracks(tmp_path, columns=["tracklet_id"], start=3, end=5, max_rows=max_rows)
    assert result.columns.tolist() == ["tracklet_id"]
    assert result.tracklet_id.tolist() == (["p3", "p4"] if max_rows else ["p3", "p4", "p5"])


def test_track_window_without_observations_and_zero_limit(tmp_path):
    writer = TrackWriter(tmp_path / "tracks")
    writer.add([{"time_s": 1.0}])
    writer.flush()
    assert load_tracks(tmp_path, start=10, end=20, max_rows=10).empty
    assert load_tracks(tmp_path, max_rows=0).empty
