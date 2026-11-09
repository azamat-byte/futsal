"""Cross-module checks for the custom-detector tactical path and honest calibration gates."""

import json
from pathlib import Path

import numpy as np
import pytest

from futsal.config import RunConfig
from futsal.geometry import CalibrationSpec, fit_calibration
from futsal.models import class_kind, suppress_duplicate_people
from futsal.notebook_ui import named_landmarks
from futsal.pipeline import run_pipeline
from futsal.quality import summarize_run
from futsal.radar import RadarRenderer, court_paths, palette
from futsal.storage import load_tracks
from futsal.video import VideoWriter, probe


def test_primary_model_and_mixed_class_policy():
    assert RunConfig().model == "best.pt"
    assert RunConfig().tracker == "bytetrack"
    assert class_kind("player- ball")[0] is None
    assert class_kind("goalkeeper") == ("person", "goalkeeper")
    assert class_kind("referee") == ("person", "referee")


def test_role_duplicates_do_not_create_extra_people_or_remove_ball():
    player = {"box": [10, 10, 40, 60], "score": .9, "class_name": "person", "role": "outfield_player"}
    duplicate = {**player, "role": "goalkeeper", "score": .7}
    ball = {"box": [10, 10, 40, 60], "score": .8, "class_name": "ball"}
    output = suppress_duplicate_people([player, duplicate, ball])
    assert output == [player, ball]


def test_futsal_marking_geometry():
    paths, marks = court_paths()
    # Quarter circles meet a 3.16 m line at x=6 m, based on outside post edges.
    left_join = next(p for p in paths if p.shape == (2, 2) and np.all(p[:, 0] == 6))
    assert left_join[1, 1] - left_join[0, 1] == pytest.approx(3.16)
    assert any(np.allclose(point, [10, 10]) for point in marks)
    assert named_landmarks()["arc_left_far_baseline"] == pytest.approx([0, 2.42])
    with pytest.raises(ValueError):
        court_paths(10, 10)


def test_invalid_calibration_cannot_render_player_positions():
    radar = RadarRenderer()
    rows = [{"court_x": 20, "court_y": 10, "calibration_valid": True, "class_name": "person",
             "role": "outfield_player", "team": "team_a", "tracklet_id": "s0:p1"}]
    off = radar.render(rows, 0, 0, {"valid": False, "reason": "camera_cut"})
    color = palette()["team_a"]
    assert not np.any(np.all(off == color, axis=2))
    on = radar.render(rows, .1, 0, {"valid": True, "scale_verified": False})
    assert np.any(np.all(on == color, axis=2))


def test_nonfinite_check_points_rejected():
    spec = CalibrationSpec(time_s=0, image_points=[[0, 0], [400, 0], [400, 200], [0, 200]],
                           court_points=[[0, 0], [40, 0], [40, 20], [0, 20]],
                           validation_image_points=[[float('nan'), 5]],
                           validation_court_points=[[20, 10]], reviewed=True)
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_calibration(spec)


def test_camera_cut_namespaces_tracks_and_synchronizes_radar(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.mp4"
    video = VideoWriter(source, 10, (160, 90))
    for i in range(10):
        video.write(np.full((90, 160, 3), 20 if i < 5 else 230, np.uint8))
    video.close()

    class ScriptedDetector:
        manifest = {"names": {0: "player"}, "checkpoint": "test_fixture"}

        def __init__(self, *args):
            self.count = 0

        def predict(self, image):
            self.count += 1
            return [{"box": [30 + self.count, 25, 50 + self.count, 65], "score": .95,
                     "class_name": "person", "role": "outfield_player", "model_class": "player"}]

    monkeypatch.setattr("futsal.pipeline.Detector", ScriptedDetector)
    run = tmp_path / "analysis"
    result = run_pipeline(source, run, RunConfig(duration=1, sample_fps=10, team_mode="off"))
    data = load_tracks(run)
    assert result["frames"] == 10
    assert data.shot_id.nunique() == 2
    ids = [set(part.tracklet_id) for _, part in data.groupby("shot_id")]
    assert len(ids[0]) == len(ids[1]) == 1
    assert ids[0].isdisjoint(ids[1])
    assert data.court_x.isna().all()
    for name in ["annotated.mp4", "radar.mp4", "tactical.mp4"]:
        assert probe(run / name)["duration_s"] == pytest.approx(1, abs=.05)
    quality = summarize_run(run)
    assert quality["accuracy"]["IDF1"]["value"] is None
    assert quality["calibration_frame_coverage"] == 0


def test_colab_code_cells_and_embedded_source():
    notebook = Path(__file__).parents[1] / "notebooks/futsal_ai_colab.ipynb"
    if not notebook.exists():
        pytest.skip("Generate notebook with scripts/build_colab.py")
    nb = json.loads(notebook.read_text("utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{i}", "exec")
    assert "best.pt" in notebook.read_text("utf-8")
