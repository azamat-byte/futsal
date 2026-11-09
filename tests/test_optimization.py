import numpy as np
import pytest

from futsal.config import RunConfig
from futsal.inference import tile_windows, filter_detections, merge_detections
from futsal.tracking import AdaptiveBoxSmoother, BallTracker, PlayerTracker


def detection(box=(40, 25, 60, 70), score=.9, category="person"):
    return {"box": list(box), "score": score, "class_name": category,
            "role": "outfield_player" if category == "person" else "unknown", "model_class": category}


def test_tiles_cover_nondivisible_frame_edges():
    windows = tile_windows(1920, 1080, 640, .25)
    coverage = np.zeros((1080, 1920), bool)
    for x1, y1, x2, y2 in windows:
        coverage[y1:y2, x1:x2] = True
    assert coverage.all()
    assert len(windows) == len(set(windows))
    assert tile_windows(320, 180, 640) == [(0, 0, 320, 180)]


def test_roi_and_class_thresholds_reject_false_candidates_without_suppressing_ball():
    rows = [detection(), detection(box=(140, 25, 160, 70)),
            detection(box=(42, 55, 44, 57), score=.07, category="ball"),
            detection(box=(1, 1, 5, 5), score=.01)]
    output = filter_detections(rows, (100, 200, 3), .05, .05, [[20, 20], [100, 20], [100, 90], [20, 90]])
    assert len(output) == 2
    assert {d["class_name"] for d in output} == {"person", "ball"}


def test_multi_view_nms_keeps_separate_players_and_overlapping_ball():
    first = detection()
    second = detection(box=(41, 25, 61, 70), score=.8)
    other = detection(box=(80, 25, 100, 70))
    ball = detection(box=(40, 25, 60, 70), category="ball")
    output = merge_detections([first, second, other, ball])
    assert len(output) == 3
    assert sum(d["class_name"] == "ball" for d in output) == 1


def test_nested_ball_boxes_from_different_views_are_one_candidate():
    narrow = detection(box=(52, 50, 62, 70), category="ball", score=.8)
    wide = detection(box=(50, 50, 85, 70), category="ball", score=.4)
    separate = detection(box=(100, 50, 110, 70), category="ball", score=.5)
    output = merge_detections([narrow, wide, separate])
    assert len(output) == 2
    assert narrow in output


def test_ball_rejects_isolated_flash_and_confirms_contiguous_observations():
    tracker = BallTracker(confirmation_frames=2, start_confidence=.2)
    shape = (360, 640, 3)
    assert tracker.update([detection(box=(50, 50, 54, 54), category="ball")], 0, shape) is None
    assert tracker.update([], .05, shape) is None
    assert tracker.update([detection(box=(300, 50, 304, 54), category="ball")], .1, shape) is None
    accepted = tracker.update([detection(box=(305, 50, 309, 54), category="ball")], .15, shape)
    assert accepted is not None
    assert tracker.update([], .2, shape) is None


def test_weak_ball_cannot_start_track_even_when_spatially_consistent():
    tracker = BallTracker(confirmation_frames=2, start_confidence=.2)
    for t in [0, .05, .1]:
        assert tracker.update([detection(score=.09, category="ball")], t, (360, 640, 3)) is None


def test_display_smoothing_reduces_stationary_jitter_and_resets_after_gap():
    smoother = AdaptiveBoxSmoother()
    raw, filtered = [], []
    for i in range(80):
        x = 100 + (4 if i % 2 else -4)
        raw.append(x)
        filtered.append(smoother.update("p1", [x, 20, x + 30, 90], i / 20)[0])
    assert np.std(filtered[10:]) < np.std(raw[10:]) * .65
    assert smoother.update("p1", [300, 20, 330, 90], 10) == [300, 20, 330, 90]


def test_smoother_follows_fast_change_without_long_lag():
    smoother = AdaptiveBoxSmoother()
    for i in range(10):
        smoother.update("p1", [10, 20, 30, 70], i / 20)
    out = smoother.update("p1", [70, 20, 90, 70], .5)
    assert abs(out[0] - 70) < 12


def test_low_score_recovery_with_appearance_and_no_invented_missing_rows():
    class Encoder:
        calls = 0

        def encode(self, crops):
            self.calls += 1
            return np.tile([1., 0., .2], (len(crops), 1)).astype(np.float32)

    encoder = Encoder()
    config = RunConfig(tracker="botsort", appearance_reid=True, smooth_boxes=True,
                       confidence=.05, track_low_threshold=.05, track_buffer_seconds=1.5)
    tracker = PlayerTracker("botsort", 20, config, encoder)
    image = np.zeros((100, 200, 3), np.uint8)
    identities = []
    for i in range(14):
        found = [] if i in {6, 7} else [detection(box=(40 + i * 2, 25, 60 + i * 2, 70),
                                                score=.1 if i == 4 else .9)]
        rows = tracker.update(found, image, i / 20)
        if not found:
            assert rows == []
        identities.extend(row["local_id"] for row in rows)
        for row in rows:
            assert row["box"] == found[0]["box"]
            assert len(row["display_box"]) == 4
    assert len(set(identities)) == 1
    assert encoder.calls > 0


def test_inconsistent_tracking_options_rejected():
    with pytest.raises(ValueError, match="low threshold"):
        RunConfig(track_low_threshold=.5)
    with pytest.raises(ValueError, match="BoT-SORT"):
        RunConfig(appearance_reid=True, tracker="bytetrack")


def test_new_track_confirmation_rejects_first_frame_flash():
    image = np.zeros((100, 200, 3), np.uint8)
    tracker = PlayerTracker("bytetrack", 20, RunConfig(track_min_observations=2))
    assert tracker.update([detection()], image, 0) == []
    assert tracker.update([], image, .05) == []
    # A different, consistently observed object can still form a track.
    rows = []
    for i in range(2, 7):
        rows += tracker.update([detection(box=(110, 25, 130, 70))], image, i / 20)
    assert rows
