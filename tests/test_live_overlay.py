"""Regression tests for the standalone overlay; no model weights or network needed."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

sv = pytest.importorskip("supervision")
module_path = Path(__file__).resolve().parents[1] / "futsal_live_overlay.py"
spec = importlib.util.spec_from_file_location("futsal_live_overlay", module_path)
overlay = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = overlay
spec.loader.exec_module(overlay)


class Model:
    def __init__(self, result=None):
        self.result = result or {"predictions": []}
        self.calls = 0

    def infer(self, frame, confidence):
        self.calls += 1
        return [self.result]


class Classifier:
    def __init__(self):
        self.calls = 0

    def predict(self, crops):
        self.calls += 1
        assert all(crop.size > 0 for crop in crops)
        return np.zeros(len(crops), dtype=int)


def field_result(full=False, confidence=0.95):
    ids = range(33) if full else overlay.ACTIVE_IDS
    keypoints = []
    for pid in ids:
        x, y = overlay.LANDMARKS.get(pid, (20, 10))
        keypoints.append({"x": 10 + 7 * x, "y": 10 + 7 * y, "confidence": confidence})
    return {"predictions": [{"class_id": 0, "class": "court", "keypoints": keypoints}]}


def player_result():
    return {
        "image": {"width": 320, "height": 180},
        "predictions": [
            {
                "x": 80,
                "y": 60,
                "width": 20,
                "height": 40,
                "confidence": 0.95,
                "class_id": 2,
                "class": "player",
            },
            {"x": 160, "y": 90, "width": 8, "height": 8, "confidence": 0.9, "class_id": 0, "class": "ball"},
            {
                "x": 230,
                "y": 60,
                "width": 20,
                "height": 40,
                "confidence": 0.95,
                "class_id": 1,
                "class": "goalkeeper",
            },
        ],
    }


def make_video(path, frames=7, size=(320, 180)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    assert writer.isOpened()
    try:
        for _ in range(frames):
            writer.write(np.full((size[1], size[0], 3), (30, 70, 30), np.uint8))
    finally:
        writer.release()
    return path


def decode_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return frames
            frames.append(frame)
    finally:
        cap.release()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"keypoint_interval": 0},
        {"radar_opacity": 1.1},
        {"radar_width_ratio": float("nan")},
        {"max_frames": -1},
        {"fallback_fps": float("inf")},
        {"max_homography_age": -1},
        {"active_ids": (0, 0, 1, 2)},
        {"landmarks": {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        overlay.OverlayConfig(**kwargs)


def test_calibration_age_counts_elapsed_frames_once():
    state = overlay.HomographyState()
    state.update(np.eye(3), 0, 12)
    # Failed periodic calibrations at 5 and 10 must not increment age twice.
    for i in range(11):
        if i in (5, 10):
            state.update(None, i, 0)
        assert state.current(i, 10) is not None
    assert state.current(11, 10) is None
    state.update(np.eye(3), 15, 8)
    assert state.current(15, 0) is not None
    assert state.current(16, 0) is None


@pytest.mark.parametrize("full", [False, True])
def test_homography_handles_both_skeleton_layouts(full):
    matrix, score = overlay.field_homography(
        np.zeros((180, 320, 3), np.uint8), Model(field_result(full)), overlay.OverlayConfig()
    )
    assert matrix is not None
    assert score == len(overlay.LANDMARKS)
    np.testing.assert_allclose(
        overlay.court_points([[80, 80], [150, 45]], matrix), [[10, 10], [20, 5]], atol=1e-4
    )


def test_homography_chooses_valid_instance_after_invalid_one():
    result = field_result(confidence=0.01)
    result["predictions"].extend(field_result()["predictions"])
    matrix, score = overlay.field_homography(
        np.zeros((180, 320, 3), np.uint8), Model(result), overlay.OverlayConfig()
    )
    assert matrix is not None and score >= 4


def test_homography_rejects_collinear_keypoints():
    result = field_result()
    for i, point in enumerate(result["predictions"][0]["keypoints"]):
        point.update(x=10 + i, y=40)
    assert overlay.field_homography(
        np.zeros((180, 320, 3), np.uint8), Model(result), overlay.OverlayConfig()
    ) == (None, 0)


def test_projection_rejects_horizon_without_losing_row_alignment():
    matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1]], float)
    projected = overlay.court_points([[1, 5], [2, 5]], matrix)
    assert projected.shape == (2, 2)
    assert np.isnan(projected[0]).all()
    np.testing.assert_allclose(projected[1], [2, 5])


def test_teams_skip_invalid_crops_cache_votes_and_expire_tracks():
    tracked = sv.Detections(
        xyxy=np.array([[5, 5, 20, 30], [50, 50, 80, 80], [20, 5, 30, 30]], float),
        confidence=np.ones(3),
        class_id=np.array([2, 2, 1]),
        tracker_id=np.array([1, 2, 3]),
    )
    history = overlay.TeamHistory(overlay.OverlayConfig(team_interval=5, track_ttl=3))
    classifier = Classifier()
    frame = np.zeros((40, 40, 3), np.uint8)
    np.testing.assert_array_equal(history.assign(frame, tracked, classifier, 0), [0, 2, 2])
    assert classifier.calls == 1
    np.testing.assert_array_equal(history.assign(frame, tracked, classifier, 1), [0, 2, 2])
    assert classifier.calls == 1
    history.assign(frame, tracked[:0], classifier, 5)
    assert history.votes == history.last_seen == {}


def test_invalid_classifier_length_fails_clearly():
    class BadClassifier:
        def predict(self, crops):
            return []

    tracked = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], float),
        confidence=np.array([1.0]),
        class_id=np.array([2]),
        tracker_id=np.array([1]),
    )
    with pytest.raises(ValueError, match="valid player crops"):
        overlay.TeamHistory(overlay.OverlayConfig()).assign(
            np.zeros((20, 20, 3), np.uint8), tracked, BadClassifier(), 0
        )


@pytest.mark.parametrize("position", ["bottom-center", "bottom-left", "bottom-right"])
@pytest.mark.parametrize("shape", [(180, 320, 3), (800, 100, 3), (9, 9, 3)])
def test_radar_fits_small_and_portrait_frames(position, shape):
    radar = overlay.Radar(shape, overlay.OverlayConfig(radar_position=position, radar_width_ratio=1))
    assert 0 <= radar.x and 0 <= radar.y
    assert radar.x + radar.base.shape[1] <= shape[1]
    assert radar.y + radar.base.shape[0] <= shape[0]


def test_render_unknown_frame_count_with_actual_detections(tmp_path, monkeypatch):
    source = make_video(tmp_path / "матч source.mp4")
    output = tmp_path / "render.mp4"
    real_capture = cv2.VideoCapture

    class UnknownLengthCapture:
        def __init__(self, path):
            self.inner = real_capture(path)
            self.released = False

        def get(self, prop):
            return 0 if prop == cv2.CAP_PROP_FRAME_COUNT else self.inner.get(prop)

        def isOpened(self):
            return self.inner.isOpened()

        def read(self):
            return self.inner.read()

        def release(self):
            self.released = True
            self.inner.release()

    monkeypatch.setattr(overlay.cv2, "VideoCapture", UnknownLengthCapture)
    monkeypatch.setattr(overlay, "find_ffmpeg", lambda: None)
    player, field, classifier = Model(player_result()), Model(field_result()), Classifier()
    stats = overlay.run_overlay(
        player, field, classifier, overlay.OverlayConfig(source, output, max_frames=4, progress=False)
    )
    assert stats.frames_processed == stats.frames_calibrated == 4
    assert stats.h264_encoded is False
    assert player.calls == 4 and field.calls == 1 and classifier.calls == 1
    monkeypatch.setattr(overlay.cv2, "VideoCapture", real_capture)
    frames = decode_video(output)
    assert len(frames) == 4 and frames[0].shape == (180, 320, 3)
    assert np.mean(np.abs(frames[0].astype(float) - decode_video(source)[0])) > 1
    assert not list(tmp_path.glob(".futsal-overlay-*"))


def test_render_empty_detections_and_no_calibration(tmp_path, monkeypatch):
    source = make_video(tmp_path / "source.mp4", frames=2)
    monkeypatch.setattr(overlay, "find_ffmpeg", lambda: None)
    stats = overlay.run_overlay(
        Model(), Model(), Classifier(), overlay.OverlayConfig(source, tmp_path / "out.mp4", progress=False)
    )
    assert stats.frames_processed == 2 and stats.frames_calibrated == 0


def test_model_failure_closes_resources_and_preserves_existing_output(tmp_path, monkeypatch):
    class FailingModel:
        def infer(self, frame, confidence):
            raise RuntimeError("model unavailable")

    source = make_video(tmp_path / "source.mp4")
    output = tmp_path / "previous.mp4"
    output.write_bytes(b"previous video")
    with pytest.raises(RuntimeError, match="model unavailable"):
        overlay.run_overlay(
            FailingModel(),
            Model(),
            Classifier(),
            overlay.OverlayConfig(source, output, overwrite=True, progress=False),
        )
    assert output.read_bytes() == b"previous video"
    assert not list(tmp_path.glob(".futsal-overlay-*"))
    # Windows refuses to rename files still held by OpenCV.
    source.rename(tmp_path / "renamed.mp4")


def test_source_cannot_be_overwritten(tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    with pytest.raises(ValueError, match="different files"):
        overlay.run_overlay(
            Model(), Model(), Classifier(), overlay.OverlayConfig(source, source, overwrite=True)
        )


def test_existing_output_requires_explicit_overwrite(tmp_path):
    source, output = tmp_path / "source.mp4", tmp_path / "out.mp4"
    source.touch()
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        overlay.run_overlay(Model(), Model(), Classifier(), overlay.OverlayConfig(source, output))
    assert output.read_bytes() == b"existing"


def test_failed_ffmpeg_falls_back_to_raw_video(tmp_path, monkeypatch):
    raw, staging = tmp_path / "raw.mp4", tmp_path / "final.mp4"
    raw.write_bytes(b"raw video")
    monkeypatch.setattr(overlay, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        overlay.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stderr="encoder failed"),
    )
    assert overlay.finalize_video(raw, tmp_path / "source.mp4", staging, 4, 10) is False
    assert staging.read_bytes() == raw.read_bytes()


def test_h264_preserves_short_audio_without_truncating_video(tmp_path):
    ffmpeg = overlay.find_ffmpeg()
    if not ffmpeg:
        pytest.skip("FFmpeg unavailable")
    silent = make_video(tmp_path / "silent.mp4", frames=7)
    source = tmp_path / "source-with-audio.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.2",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    output = tmp_path / "out.mp4"
    stats = overlay.run_overlay(
        Model(player_result()),
        Model(field_result()),
        Classifier(),
        overlay.OverlayConfig(source, output, progress=False),
    )
    assert stats.h264_encoded and len(decode_video(output)) == 7
    inspection = subprocess.run([ffmpeg, "-hide_banner", "-i", str(output)], capture_output=True, text=True)
    assert "Video: h264" in inspection.stderr
    assert "Audio: aac" in inspection.stderr


def test_publish_failure_keeps_concurrently_created_output(tmp_path, monkeypatch):
    source = make_video(tmp_path / "source.mp4", frames=1)
    output = tmp_path / "out.mp4"

    def finalize(raw, source, staging, count, fps):
        shutil.copy2(raw, staging)
        output.write_bytes(b"concurrent writer")
        return False

    monkeypatch.setattr(overlay, "finalize_video", finalize)
    with pytest.raises(FileExistsError):
        overlay.run_overlay(
            Model(), Model(), Classifier(), overlay.OverlayConfig(source, output, progress=False)
        )
    assert output.read_bytes() == b"concurrent writer"
