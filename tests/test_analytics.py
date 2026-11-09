import pandas as pd
import pytest

from futsal.analytics import movement, eligible_positions, occupancy, apply_review, pass_hypotheses
from futsal.reid import validate_merge


def points():
    return pd.DataFrame(
        [
            dict(
                entity_id="p1",
                tracklet_id="s0p1",
                time_s=i * 0.1,
                dt_s=0.1,
                shot_id=0,
                court_x=2 + i * 0.1,
                court_y=2,
                class_name="person",
                observed=True,
                calibration_valid=True,
                scale_verified=True,
                segment_label="live_play_wide",
                team="team_a",
                frame_index=i,
            )
            for i in range(11)
        ]
    )


def test_speed_time_units():
    stats, _ = movement(points())
    assert 0.85 <= stats.iloc[0].observed_distance_m <= 1.05
    assert 0.85 <= stats.iloc[0].mean_speed_mps <= 1.05


def test_cut_never_adds_teleport_distance():
    data = points()
    data.loc[5:, "shot_id"] = 1
    data.loc[5:, "court_x"] += 30
    stats, _ = movement(data)
    assert stats.iloc[0].observed_distance_m < 2


@pytest.mark.parametrize(
    "column,value",
    [
        ("calibration_valid", False),
        ("scale_verified", False),
        ("segment_label", "replay"),
        ("segment_label", "unknown"),
        ("observed", False),
    ],
)
def test_quality_gate(column, value):
    data = points()
    data[column] = value
    assert eligible_positions(data).empty


def test_heatmap_uses_time():
    data = points()
    hist, _, _ = occupancy(data)
    assert hist.sum() == pytest.approx(1.1)


def test_review_keeps_unknown():
    data = points()
    corrected = apply_review(data, {"segments": {}, "identities": {}, "teams": {}})
    assert (corrected.segment_label == "unknown").all()


def test_simultaneous_merge_rejected():
    data = pd.DataFrame({"tracklet_id": ["a", "b"], "time_s": [1.0, 1.0]})
    with pytest.raises(ValueError):
        validate_merge(data, ["a", "b"])


def test_unknown_gap_cannot_make_pass():
    values = [
        dict(time_s=0, shot_id=0, state="control_candidate", entity_id="a", team="team_a"),
        dict(time_s=0.1, shot_id=0, state="unknown", entity_id=None, team=None),
        dict(time_s=0.2, shot_id=0, state="control_candidate", entity_id="b", team="team_a"),
    ]
    assert pass_hypotheses(pd.DataFrame(values)) == []
