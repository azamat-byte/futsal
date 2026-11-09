import json

import cv2
import numpy as np
import pandas as pd
import pytest

from futsal.phase1.config import Phase1Config, TeamConfig
from futsal.phase1.detection import parse_predictions
from futsal.phase1.pipeline import Phase1Pipeline
from futsal.phase1.video import VideoReader


class ColorDetector:
    """Synthetic fixture detector: never makes network requests or claims model quality."""

    def detect(self, frame):
        rows = []
        # Red/blue players and a green goalkeeper with deliberately arbitrary class IDs.
        for channel, role, class_id in [(2, "player", 12), (0, "player", 12), (1, "goalkeeper", 63)]:
            mask = (frame[:, :, channel] > 160) & (np.delete(frame, channel, axis=2).max(axis=2) < 100)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 20:
                    rows.append({"x": x + w / 2, "y": y + h / 2, "width": w, "height": h,
                                 "confidence": .95, "class": role, "class_id": class_id})
        return parse_predictions({"predictions": rows}, frame.shape, Phase1Config().detector)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 180))
    if not writer.isOpened():
        pytest.skip("MP4 encoder unavailable")
    for i in range(40):
        frame = np.full((180, 320, 3), 40, np.uint8)
        cv2.rectangle(frame, (20 + i, 50), (44 + i, 130), (0, 0, 230), -1)
        cv2.rectangle(frame, (160 - i, 50), (184 - i, 130), (230, 0, 0), -1)
        cv2.rectangle(frame, (270, 60), (295, 140), (0, 230, 0), -1)
        writer.write(frame)
    writer.release()
    return path


def config():
    return Phase1Config(teams=TeamConfig(calibration_frames=8, calibration_stride=3, min_samples=12))


def test_end_to_end_video_csv_and_cut_boundaries(video, tmp_path):
    output = tmp_path / "result"
    Phase1Pipeline(ColorDetector(), config()).run(video, output, max_frames=40, cut_frames=(20,))
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "completed" and manifest["frames_processed"] == 40
    assert manifest["segments"] == 2
    tracks = pd.read_csv(output / "tracks.csv")
    assert set(tracks.role) == {"player", "goalkeeper"}
    assert set(tracks[tracks.role == "goalkeeper"].team) == {"unknown"}
    assert {"team_a", "team_b"} <= set(tracks.team)
    assert np.allclose(tracks.timestamp_s, tracks.frame_index / 20)
    assert np.allclose(tracks.foot_x_px, (tracks.x1 + tracks.x2) / 2)
    before = set(tracks[tracks.frame_index < 20].track_id)
    after = set(tracks[tracks.frame_index >= 20].track_id)
    assert before.isdisjoint(after)
    assert tracks.groupby("segment_id").track_id.nunique().to_list() == [3, 3]
    with VideoReader(output / "annotated.mp4") as reader:
        assert len(list(reader.frames())) == 40
        assert reader.info.fps == 20
    assert (output / "calibration.jpg").stat().st_size > 1000
    assert len(json.loads((output / "teams.json").read_text())["clusters"]) == 2


def test_existing_output_is_preserved(video, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep me")
    with pytest.raises(FileExistsError):
        Phase1Pipeline(ColorDetector(), config()).run(video, output)
    assert sentinel.read_text() == "keep me"


def test_calibration_failure_marks_partial_run(video, tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Not enough"):
        Phase1Pipeline(ColorDetector(), config()).run(video, output, max_frames=1)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_inference_failure_closes_video_and_marks_failed(video, tmp_path):
    class FailingDetector(ColorDetector):
        calls = 0

        def detect(self, frame):
            self.calls += 1
            if self.calls > 12:
                raise RuntimeError("simulated outage")
            return super().detect(frame)

    output = tmp_path / "failed_inference"
    with pytest.raises(RuntimeError, match="simulated outage"):
        Phase1Pipeline(FailingDetector(), config()).run(video, output)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    # Windows rejects this rename when a writer keeps the file open.
    (output / "annotated.mp4").rename(output / "partial.mp4")


def test_start_frame_is_absolute_and_output_contains_only_requested_frames(video, tmp_path):
    output = tmp_path / "offset"
    Phase1Pipeline(ColorDetector(), config()).run(video, output, start_frame=10, max_frames=20)
    tracks = pd.read_csv(output / "tracks.csv")
    assert tracks.frame_index.min() >= 10 and tracks.frame_index.max() == 29
    assert np.allclose(tracks.timestamp_s, tracks.frame_index / 20)
    with VideoReader(output / "annotated.mp4") as reader:
        assert reader.info.frame_count == 20
