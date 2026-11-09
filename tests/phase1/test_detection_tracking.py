from unittest.mock import MagicMock

import numpy as np
import pytest
import requests
import supervision as sv

from futsal.phase1.config import DetectorConfig, TrackingConfig
from futsal.phase1.detection import RoboflowDetector, parse_predictions, person_detections
from futsal.phase1.tracking import PersonTracker, SceneCutDetector


def prediction(name="player", class_id=42, confidence=.9, **changes):
    return {"class": name, "class_id": class_id, "x": 40, "y": 50,
            "width": 20, "height": 40, "confidence": confidence, **changes}


def detections(*rows):
    return parse_predictions({"predictions": list(rows)}, (160, 320, 3), DetectorConfig())


def test_name_mapping_clipping_and_cross_role_nms():
    rows = detections(prediction(), prediction("goalkeeper", 17, .8),
                      prediction("ball", 99, .95), prediction(x=-5, width=20),
                      prediction(x=float("nan")))
    assert len(rows) == 4
    assert rows.xyxy[-1, 0] == 0
    people = person_detections(rows, .7)
    assert len(people) == 2
    assert set(people.class_id) == {42}
    assert set(people.data["role"]) == {"player"}


def test_invalid_payload_and_unknown_class_are_not_silent_empty_frames():
    with pytest.raises(ValueError, match="predictions"):
        parse_predictions({"error": "bad model"}, (100, 100), DetectorConfig())
    with pytest.raises(ValueError, match="Unmapped"):
        detections(prediction("unexpected"))
    assert len(detections(prediction(width=-1), prediction(confidence=.05))) == 0


def test_short_occlusion_and_low_score_match_preserve_id():
    tracker = PersonTracker(TrackingConfig(minimum_consecutive_frames=1), 30)
    first = tracker.update(detections(prediction()))
    identity = int(first.tracker_id[0])
    for _ in range(3):
        assert len(tracker.update(sv.Detections.empty())) == 0
    recovered = tracker.update(detections(prediction(x=41)))
    assert recovered.tracker_id.tolist() == [identity]
    low_score = tracker.update(detections(prediction(x=42, confidence=.2)))
    assert low_score.tracker_id.tolist() == [identity]


def test_cut_reset_does_not_reuse_ids_or_role_metadata():
    tracker = PersonTracker(TrackingConfig(minimum_consecutive_frames=1), 25)
    old = tracker.update(detections(prediction()))
    tracker.reset_segment()
    new = tracker.update(detections(prediction("goalkeeper", 17)))
    assert new.tracker_id[0] > old.tracker_id[0]
    assert tracker.segment_id == 1
    assert new.data["role"][0] == "goalkeeper"
    assert new.class_id[0] == 17


def test_empty_frames_expire_tracks_at_source_fps():
    tracker = PersonTracker(TrackingConfig(lost_seconds=.1, minimum_consecutive_frames=1), 10)
    old = tracker.update(detections(prediction()))
    for _ in range(5):
        tracker.update(sv.Detections.empty())
    for _ in range(3):
        new = tracker.update(detections(prediction()))
    assert new.tracker_id[0] != old.tracker_id[0]


def test_scene_cut_heuristic():
    cut = SceneCutDetector(.65)
    red = np.full((100, 200, 3), (0, 0, 220), np.uint8)
    blue = np.full((100, 200, 3), (220, 0, 0), np.uint8)
    assert not cut.update(red)
    assert not cut.update(red.copy())
    assert cut.update(blue)


def response(status, payload=None):
    value = MagicMock()
    value.status_code = status
    value.json.return_value = payload
    return value


def test_http_request_preserves_model_id_confidence_and_header(monkeypatch):
    detector = RoboflowDetector(DetectorConfig(), "test-secret")
    detector.session.post = MagicMock(return_value=response(200, {"predictions": [prediction()]}))
    result = detector.detect(np.zeros((100, 120, 3), np.uint8))
    args, kwargs = detector.session.post.call_args
    assert args[0].endswith("/s-workspace-bimfr/new_futsal-2-rfdetr-medium-t1")
    assert kwargs["params"]["confidence"] == 10
    assert "api_key" not in kwargs["params"]
    assert detector.session.headers["Authorization"] == "Bearer test-secret"
    assert result.data["role"][0] == "player"
    detector.close()


def test_transient_retry_and_fail_fast_auth(monkeypatch):
    monkeypatch.setattr("futsal.phase1.detection.time.sleep", lambda _: None)
    detector = RoboflowDetector(DetectorConfig(), "test-secret")
    detector.session.post = MagicMock(side_effect=[requests.Timeout("test-secret"),
                                                   response(200, {"predictions": []})])
    assert len(detector.detect(np.zeros((100, 100, 3), np.uint8))) == 0
    assert detector.session.post.call_count == 2
    detector.session.post = MagicMock(return_value=response(401))
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        detector.detect(np.zeros((100, 100, 3), np.uint8))
    assert "test-secret" not in str(error.value)
    assert detector.session.post.call_count == 1
    detector.close()
