import numpy as np
import pandas as pd
import pytest

from futsal.analytics import pass_hypotheses
from futsal.court import line_candidates
from futsal.formations import formation_features
from futsal.geometry import DynamicCalibration
from futsal.models import class_kind
from futsal.tracking import BallTracker


def test_class_maps_not_class_numbers():
    assert class_kind("goalkeeper") == ("person", "goalkeeper")
    assert class_kind("sports ball") == ("ball", "unknown")
    assert class_kind("referee") == ("person", "referee")
    assert class_kind("car")[0] is None


def test_ambiguous_ball_is_unknown():
    tracker = BallTracker()
    found = [dict(box=[0, 0, 4, 4], score=0.8), dict(box=[20, 0, 24, 4], score=0.78)]
    assert tracker.update(found, 1, (100, 100, 3)) is None
    assert tracker.update([], 2, (100, 100, 3)) is None


def test_cut_invalidates_calibration():
    calibration = DynamicCalibration([])
    calibration.matrix = np.eye(3)
    calibration.info = {"valid": True, "reason": "accepted", "error_px": 0.0, "scale_verified": True}
    calibration.update(np.zeros((100, 100, 3), np.uint8), 1, [], cut=True)
    assert calibration.matrix is None
    assert not calibration.info["valid"]


def test_pass_cannot_bridge_missing_frames():
    rows = [
        dict(time_s=0.0, shot_id=0, state="control_candidate", entity_id="a", team="team_a"),
        dict(time_s=1.0, shot_id=0, state="free_ball", entity_id=None, team=None),
        dict(time_s=1.1, shot_id=0, state="control_candidate", entity_id="b", team="team_a"),
    ]
    assert pass_hypotheses(pd.DataFrame(rows)) == []


def test_formation_requires_four_players():
    with pytest.raises(ValueError):
        formation_features(np.zeros((5, 3, 2)))
    assert np.isfinite(formation_features(np.arange(80).reshape(10, 4, 2))).all()


def test_line_candidates_empty_frame():
    assert line_candidates(np.zeros((100, 100, 3), np.uint8)) == []
