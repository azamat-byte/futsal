import pandas as pd
import json

from futsal.exports import export_run
from futsal.storage import TrackWriter, write_json


def test_streaming_export_preserves_nulls_across_different_batches(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "manifest.json", {"schema_version": 1})
    writer = TrackWriter(run / "tracks")
    writer.add(
        [
            dict(
                frame_index=1,
                time_s=0.1,
                shot_id=0,
                class_name="ball",
                role="unknown",
                team="unknown",
                observed=True,
                scale_verified=False,
                calibration_valid=False,
                score=0.8,
                x1=1.0,
                y1=2.0,
                x2=3.0,
                y2=4.0,
            )
        ]
    )
    writer.flush()
    writer.add(
        [
            dict(
                frame_index=2,
                time_s=0.2,
                shot_id=0,
                class_name="person",
                role="outfield_player",
                team="team_a",
                tracklet_id="s0p1",
                observed=True,
                scale_verified=False,
                calibration_valid=False,
                score=0.9,
                x1=1.0,
                y1=2.0,
                x2=3.0,
                y2=4.0,
            )
        ]
    )
    writer.flush()
    archive = export_run(run, tmp_path / "export")
    assert archive.exists()
    data = pd.read_parquet(tmp_path / "export/tracks.parquet")
    assert len(data) == 2
    assert pd.isna(data.entity_id.iloc[0])
    assert data.entity_id.iloc[1] == "s0p1"
    assert data.court_x.isna().all()
    records = json.loads((tmp_path / "export/tracks.json").read_text("utf-8"))
    assert len(records) == 2
    assert records[0]["court_x"] is None
