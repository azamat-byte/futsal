from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from futsal.quality import empty_observation_message
from futsal.storage import write_json
from futsal.video import VideoWriter


def test_empty_message_distinguishes_finished_running_and_window():
    manifest = {"state": "completed", "frames": 70, "detections": 1}
    message = empty_observation_message(manifest)
    assert "70 frames" in message
    assert "1 detector candidate" in message
    assert "none passed" in message
    assert "pending result" in message
    assert "time window" in empty_observation_message(manifest, has_stored_tracks=True)
    assert "still running" in empty_observation_message({"state": "running"})
    assert "no supported objects" in empty_observation_message({**manifest, "detections": 0})
    assert "failed" in empty_observation_message({"state": "failed"})


def test_finished_empty_run_still_shows_result_video(tmp_path, monkeypatch):
    run = tmp_path / "empty_run"
    run.mkdir()
    video = run / "annotated.mp4"
    writer = VideoWriter(video, 5, (64, 48))
    writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.close()
    write_json(run / "manifest.json", {"state": "completed", "frames": 70, "detections": 1,
                                       "config": {"start": 0}})
    job = {"id": "empty_test", "source": str(video), "state": "completed", "frames": 70,
           "progress": 1, "attempt": 1, "run_path": str(run), "error": None}
    monkeypatch.setattr("futsal.jobs.list_jobs", lambda root: [job])
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard/app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Match viewer").run()
    assert not app.exception
    assert any("none passed" in warning.value for warning in app.warning)
    assert len(app.get("video")) == 1
