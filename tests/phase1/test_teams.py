import cv2
import numpy as np
import pytest

from futsal.phase1.annotations import TrackingAnnotator
from futsal.phase1.config import TeamConfig
from futsal.phase1.teams import JerseyColorExtractor, TeamClassifier, TemporalTeamAssigner


def hsv_crop(h, s=240, v=200):
    return cv2.cvtColor(np.full((40, 30, 3), (h, s, v), np.uint8), cv2.COLOR_HSV2BGR)


def test_hue_wrap_and_achromatic_shirts():
    extractor = JerseyColorExtractor()
    assert np.linalg.norm(extractor.from_crop(hsv_crop(0)) - extractor.from_crop(hsv_crop(179))) < .06
    white = extractor.from_crop(hsv_crop(0, 0, 255))
    black = extractor.from_crop(hsv_crop(100, 0, 10))
    assert np.linalg.norm(white - black) > .5
    assert np.isfinite(white).all() and np.isfinite(black).all()


def test_crop_excludes_shorts_and_rejects_invalid_boxes():
    image = np.full((120, 60, 3), (0, 0, 220), np.uint8)
    image[60:] = (220, 0, 0)
    extractor = JerseyColorExtractor()
    assert np.all(extractor.crop(image, np.array([0, 0, 60, 120]))[:, :, 2] == 220)
    for box in ([60, 50, 20, 90], [-100, -100, -50, -20], [0, 0, 3, 3], [0, 0, np.nan, 10]):
        assert extractor.extract(image, np.array(box)) is None


def test_dominant_color_tolerates_small_background_contamination():
    extractor = JerseyColorExtractor()
    crop = hsv_crop(110)
    original = extractor.from_crop(crop)
    crop[:, :4] = (140, 160, 180)
    assert np.linalg.norm(extractor.from_crop(crop) - original) < .02


def test_fit_two_teams_and_reject_outlier():
    extractor = JerseyColorExtractor()
    red, blue, green = [extractor.from_crop(hsv_crop(h)) for h in (0, 110, 60)]
    classifier = TeamClassifier(TeamConfig(min_samples=8))
    classifier.fit([red] * 10 + [blue] * 10)
    assert {classifier.predict(red), classifier.predict(blue)} == {"team_a", "team_b"}
    assert classifier.predict(green) == "unknown"
    assert classifier.predict(None) == "unknown"


def test_three_clusters_respect_reviewed_mapping():
    extractor = JerseyColorExtractor()
    features = [extractor.from_crop(hsv_crop(h)) for h in (0, 60, 110)]
    classifier = TeamClassifier(TeamConfig(min_samples=9, cluster_labels=("team_b", "other", "team_a")))
    classifier.fit(np.repeat(features, 5, axis=0))
    assert [classifier.predict(f) for f in features] == ["team_b", "other", "team_a"]


def test_fit_fails_when_single_team_or_too_few_samples():
    classifier = TeamClassifier(TeamConfig(min_samples=8))
    with pytest.raises(ValueError, match="Not enough"):
        classifier.fit([np.zeros(4)])
    with pytest.raises(ValueError, match="indistinguishable"):
        classifier.fit(np.zeros((20, 4)))


def test_votes_suppress_flicker_then_release_stale_labels():
    config = TeamConfig(history_size=5, min_votes=3, vote_fraction=.6, history_ttl_seconds=1)
    assigner = TemporalTeamAssigner(config, fps=10)
    assert assigner.update(1, "team_a", 0)[0] == "unknown"
    for frame in range(1, 5):
        label, _ = assigner.update(1, "team_a", frame)
    assert label == "team_a"
    assert assigner.update(1, "team_b", 5)[0] == "team_a"
    for frame in range(6, 11):
        label, _ = assigner.update(1, "unknown", frame)
    assert label == "unknown"
    assigner.prune(30)
    assert not assigner.histories


def test_annotation_preserves_detection_classes():
    import supervision as sv

    tracks = sv.Detections(xyxy=np.array([[10, 30, 40, 90]], np.float32),
                           confidence=np.array([.9]), class_id=np.array([77]),
                           tracker_id=np.array([3]), data={"role": np.array(["player"])})
    frame = np.zeros((120, 200, 3), np.uint8)
    annotated = TrackingAnnotator().annotate(frame, tracks, ["team_a"], 1, 0)
    assert tracks.class_id.tolist() == [77]
    assert frame.sum() == 0 and annotated.sum() > 0
