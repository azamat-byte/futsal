"""Guard against presenting goalkeeper kits as team identity or trajectories as metric truth."""

import numpy as np
import pandas as pd

from futsal.walkthrough import cluster_outfield, goalkeeper_groups, track_diagnostics


def test_goalkeepers_do_not_contaminate_outfield_prototypes_or_receive_team_labels():
    samples = [dict(role="outfield_player", feature=[.4, .2, .3])] * 8
    samples += [dict(role="outfield_player", feature=[.7, .7, .7])] * 8
    samples += [dict(role="goalkeeper", feature=[0., 0., 0.])] * 5
    rows = pd.DataFrame(samples)
    labeled, model, report = cluster_outfield(rows)
    assert report["accuracy"] is None
    assert set(labeled.loc[labeled.role.eq("outfield_player"), "team"]) == {"team_a", "team_b"}
    assert labeled.loc[labeled.role.eq("goalkeeper"), "team"].eq("unknown").all()
    assert np.allclose(model.cluster_centers_, [[.4, .2, .3], [.7, .7, .7]])
    grouped = goalkeeper_groups(labeled)
    assert grouped.team.eq("unknown").all()


def test_similar_jersey_clusters_stay_unknown():
    rows = pd.DataFrame([dict(role="outfield_player", feature=[.5+i*.001, .5, .5]) for i in range(20)])
    labeled, _, report = cluster_outfield(rows)
    assert labeled.team.eq("unknown").all()
    assert report["accepted_crops"] == 0


def test_ball_rows_do_not_create_player_tracklets():
    rows = pd.DataFrame([dict(class_name="ball",tracklet_id=None,time_s=3.)])
    assert track_diagnostics(rows).empty
