import csv
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from futsal.phase1.pipeline import TRACK_COLUMNS, write_json
from futsal.phase1.video import VideoReader
from futsal.phase2.calibration import Calibration, fingerprint
from futsal.phase2.pipeline import Phase2Pipeline, track_groups


@pytest.fixture
def fixture_run(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 300))
    assert writer.isOpened()
    rng = np.random.default_rng(123)
    image = np.full((300, 480, 3), 80, np.uint8)
    for x, y in rng.integers([25, 40], [420, 220], (350, 2)):
        cv2.circle(image, (int(x), int(y)), 2, (220, 220, 220), -1)
    for _ in range(20):
        writer.write(image)
    writer.release()
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    write_json(phase1 / "run.json", {"status": "completed", "source": str(source),
                                    "video": {"width": 480, "height": 300, "fps": 10, "frame_count": 20},
                                    "start_frame": 5, "frames_processed": 10,
                                    "manual_cut_frames": [], "config": {"tracking": {}}})
    with (phase1 / "tracks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, TRACK_COLUMNS)
        writer.writeheader()
        for frame in range(5, 15):
            for identity, foot in [(1, (220, 130)), (2, (470, 270))]:
                writer.writerow(dict(frame_index=frame, timestamp_s=frame/10, segment_id=0, track_id=identity,
                                     role="player", team="team_a", team_vote_fraction=1,
                                     detection_confidence=.9, x1=foot[0]-10, x2=foot[0]+10,
                                     y1=foot[1]-40, y2=foot[1], foot_x_px=foot[0], foot_y_px=foot[1]))
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"schema_version": 1, "source_sha256": fingerprint(source),
                            "image_size": [480, 300], "pitch_size_m": [40, 20], "scale_verified": False,
                            "anchors": [{"frame_index": 5, "reviewed": True,
                                         "pitch_points": [[0, 0], [40, 0], [40, 20], [0, 20]],
                                         "image_points": [[20, 30], [420, 30], [420, 230], [20, 230]],
                                         "check_pitch_points": [[20, 0], [20, 20]],
                                         "check_image_points": [[220, 30], [220, 230]]}]})
    return source, phase1, calibration


def test_end_to_end_offsets_projection_outside_and_synced_videos(fixture_run, tmp_path):
    _, phase1, calibration = fixture_run
    output = Phase2Pipeline().run(phase1, calibration, tmp_path / "phase2")
    data = pd.read_csv(output / "positions.csv")
    assert len(data) == 20 and data.frame_index.min() == 5
    valid = data[data.track_id == 1]
    assert valid.position_valid.all()
    assert np.allclose(valid.pitch_x_m, 20, atol=.05) and np.allclose(valid.pitch_y_m, 10, atol=.05)
    outside = data[data.track_id == 2]
    assert set(outside.position_status) == {"outside_pitch"}
    assert outside.pitch_x_m.isna().all()
    assert set(data.coordinate_units) == {"assumed_m"}
    for name in ("tactical.mp4", "radar.mp4"):
        with VideoReader(output / name) as reader:
            assert reader.info.frame_count == 10 and reader.info.fps == 10
            assert len(list(reader.frames())) == 10
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "completed" and report["calibration_coverage"] == 1


def test_expiry_creates_unknown_coordinates_and_audited_gaps(fixture_run, tmp_path):
    _, phase1, calibration = fixture_run
    output = Phase2Pipeline().run(phase1, calibration, tmp_path / "phase2", max_age_s=.01)
    data = pd.read_csv(output / "positions.csv")
    assert data[data.frame_index > 5].pitch_x_m.isna().all()
    states = [json.loads(line) for line in (output / "calibration_frames.jsonl").read_text().splitlines()]
    assert len(states) == 10 and states[1]["calibration_status"] == "anchor_expired"


def test_wrong_source_and_unreviewed_calibration_rejected(fixture_run):
    source, _, calibration = fixture_run
    raw = json.loads(calibration.read_text())
    raw["anchors"][0]["reviewed"] = False
    write_json(calibration, raw)
    with VideoReader(source) as reader:
        with pytest.raises(ValueError, match="Review"):
            Calibration.load(calibration, source, reader.info)
        raw["source_sha256"] = "wrong"
        write_json(calibration, raw)
        with pytest.raises(ValueError, match="different video"):
            Calibration.load(calibration, source, reader.info)


def test_existing_output_is_not_overwritten(fixture_run, tmp_path):
    _, phase1, calibration = fixture_run
    target = tmp_path / "keep"
    target.mkdir()
    with pytest.raises(FileExistsError):
        Phase2Pipeline().run(phase1, calibration, target)


def test_unsorted_tracks_fail_with_failed_manifest(fixture_run, tmp_path):
    _, phase1, calibration = fixture_run
    path = phase1 / "tracks.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], *lines[3:5], *lines[1:3], *lines[5:]]) + "\n")
    with pytest.raises(ValueError, match="sorted"):
        list(track_groups(path))
    target = tmp_path / "failed"
    with pytest.raises(ValueError):
        Phase2Pipeline().run(phase1, calibration, target)
    assert json.loads((target / "run.json").read_text())["status"] == "failed"


def test_point_picker_converts_clicks_and_saves_reviewed_anchor(fixture_run, tmp_path, monkeypatch):
    from futsal.phase2.picker import pick

    source, _, _ = fixture_run
    callbacks = {}
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, callback: callbacks.update(click=callback))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)
    # Preset center-view landmarks, represented in source pixels plus the UI header.
    pixels = [[220, 30], [170, 230], [270, 230], [220, 100], [220, 230], [220, 160]]
    actions = iter([*pixels, 13, 13, ord('q')])

    def wait(_):
        action = next(actions)
        if isinstance(action, list):
            callbacks["click"](cv2.EVENT_LBUTTONDOWN, action[0], action[1]+70, 0, None)
            return 0
        return action

    monkeypatch.setattr(cv2, "waitKey", wait)
    target = tmp_path / "picked.json"
    assert pick(source, target, 5) == target
    raw = json.loads(target.read_text())
    assert raw["anchors"][0]["image_points"] == pixels[:4]
    assert raw["anchors"][0]["reviewed"] is True
    with VideoReader(source) as reader:
        assert 5 in Calibration.load(target, source, reader.info).anchors


def test_render_subrange_reuses_absolute_tracks(fixture_run, tmp_path):
    _, phase1, calibration = fixture_run
    raw = json.loads(calibration.read_text())
    raw["anchors"][0]["frame_index"] = 8
    write_json(calibration, raw)
    output = Phase2Pipeline().run(phase1, calibration, tmp_path / "subset", start_frame=8, max_frames=3)
    data = pd.read_csv(output / "positions.csv")
    assert set(data.frame_index) == {8, 9, 10}
    assert np.allclose(data.timestamp_s, data.frame_index/10)
    with pytest.raises(ValueError, match="inside"):
        Phase2Pipeline().run(phase1, calibration, tmp_path / "bad", start_frame=20)
