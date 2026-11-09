"""Quality gates apply to every aggregation, including after user corrections."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from .jobs import corrections
from .storage import load_tracks, read_json, write_json


def apply_review(data, review):
    data = data.copy()
    data["segment_label"] = data.shot_id.astype(str).map(review.get("segments", {})).fillna("unknown")
    identity = review.get("identities", {})
    data["entity_id"] = data.tracklet_id.map(lambda k: identity.get(k, k))
    teams = review.get("teams", {})
    data["team"] = data.apply(lambda r: teams.get(r.tracklet_id, r.team), axis=1)
    roles = review.get("roles", {})
    if "role" in data.columns:
        data["role"] = data.apply(lambda r: roles.get(r.tracklet_id, r.role), axis=1)
    for pts, point in review.get("ball_ground", {}).items():
        mask = (data.frame_index == int(pts)) & (data.class_name == "ball")
        if point.get("reviewed") and point.get("scale_verified"):
            data.loc[mask, "court_x"] = point["x"]
            data.loc[mask, "court_y"] = point["y"]
            data.loc[mask, "calibration_valid"] = True
            data.loc[mask, "scale_verified"] = True
    return data


def eligible_positions(data):
    role_ok = data.role.isin(["outfield_player", "goalkeeper"]) if "role" in data.columns else True
    return data[
        (data.class_name == "person")
        & data.observed
        & data.calibration_valid
        & data.scale_verified
        & data.court_x.notna()
        & data.court_y.notna()
        & (data.segment_label == "live_play_wide")
        & role_ok
    ].copy()


def movement(data, max_gap=0.25, max_speed=12):
    """Observed segments only. Median smoothing requires contiguous observations."""
    results = []
    points = []
    for entity, group in data.groupby("entity_id"):
        group = group.sort_values("time_s").copy()
        if group.duplicated("time_s").any():
            raise ValueError(f"Identity {entity} assigned to simultaneous tracks")
        breaks = (
            group.shot_id.ne(group.shot_id.shift())
            | group.time_s.diff().gt(max_gap)
            | group.time_s.diff().le(0)
        )
        group["piece"] = breaks.cumsum()
        distance = duration = 0.0
        speeds = []
        for _, part in group.groupby("piece"):
            part = part.copy()
            for coord in ("court_x", "court_y"):
                part["filtered_" + coord] = part[coord].rolling(3, center=True, min_periods=1).median()
            dt = part.time_s.diff()
            ds = np.hypot(part.filtered_court_x.diff(), part.filtered_court_y.diff())
            speed = ds / dt
            valid = dt.gt(0) & dt.le(max_gap) & speed.le(max_speed)
            part["speed_mps"] = speed.where(valid)
            part["distance_step_m"] = ds.where(valid)
            distance += float(ds[valid].sum())
            duration += float(dt[valid].sum())
            speeds.extend(speed[valid].tolist())
            points.append(part)
        results.append(
            {
                "entity_id": entity,
                "observed_distance_m": distance if duration else None,
                "valid_movement_duration_s": duration,
                "mean_speed_mps": distance / duration if duration else None,
                "p95_speed_mps": float(np.percentile(speeds, 95)) if speeds else None,
                "accuracy_status": "physical_accuracy_not_independently_validated",
            }
        )
    return pd.DataFrame(results), pd.concat(points, ignore_index=True) if points else pd.DataFrame()


def occupancy(data, length=40, width=20, cell=1):
    weights = np.minimum(data.dt_s.to_numpy(float), 0.25)
    hist, x, y = np.histogram2d(
        data.court_x,
        data.court_y,
        bins=[np.arange(0, length + cell, cell), np.arange(0, width + cell, cell)],
        weights=weights,
    )
    return hist.T, x, y


def tactical_geometry(data):
    records = []
    for (time, team), part in data[data.team.isin(["team_a", "team_b"])].groupby(["time_s", "team"]):
        outfield = part[part.role == "outfield_player"]
        complete = len(outfield) == 4 and outfield.entity_id.nunique() == 4
        points = part[["court_x", "court_y"]].to_numpy()
        center = points.mean(0)
        area = None
        if len(points) >= 3:
            try:
                area = float(ConvexHull(points).volume)
            except QhullError:
                pass
        records.append(
            {
                "time_s": time,
                "team": team,
                "visible_players": len(part),
                "centroid_x": center[0],
                "centroid_y": center[1],
                "width_m": float(np.ptp(points[:, 1])),
                "depth_m": float(np.ptp(points[:, 0])),
                "hull_area_m2": area,
                "scope": "four_outfield_players_visible" if complete else "visible_subset",
                "formation": "unknown",
            }
        )
    return pd.DataFrame(records)


def possession_hypotheses(data, max_distance=1.2, hold_s=0.3):
    """Conservative proximity/persistence hypotheses; never official possession statistics."""
    rows = []
    current = None
    since = None
    previous_time = None
    previous_shot = None
    for time, frame in data.groupby("time_s", sort=True):
        state = "unknown"
        candidate = None
        shot = int(frame.shot_id.iloc[0])
        live = frame.segment_label.iloc[0] == "live_play_wide"
        ball = frame[
            (frame.class_name == "ball")
            & frame.calibration_valid
            & frame.scale_verified
            & frame.court_x.notna()
            & frame.court_y.notna()
        ]
        people = frame[
            (frame.class_name == "person")
            & frame.calibration_valid
            & frame.scale_verified
            & frame.court_x.notna()
            & frame.court_y.notna()
            & frame.team.ne("unknown")
        ]
        if live and len(ball) == 1 and len(people):
            p = ball[["court_x", "court_y"]].iloc[0].to_numpy(float)
            d = np.linalg.norm(people[["court_x", "court_y"]].to_numpy(float) - p, axis=1)
            order = np.argsort(d)
            if d[order[0]] <= max_distance:
                if len(order) > 1 and d[order[1]] - d[order[0]] < 0.35:
                    state = "contested"
                else:
                    who = people.iloc[order[0]]
                    candidate = (who.entity_id, who.team)
            else:
                state = "free_ball"
        if (
            previous_time is None
            or time - previous_time > 0.25
            or shot != previous_shot
            or candidate != current
        ):
            current = candidate
            since = time
        if candidate:
            state = "control_candidate" if time - since >= hold_s else "control_pending"
        rows.append(
            {
                "time_s": float(time),
                "shot_id": shot,
                "state": state,
                "entity_id": candidate[0] if state == "control_candidate" else None,
                "team": candidate[1] if state == "control_candidate" else None,
                "status": "hypothesis_requires_review",
            }
        )
        previous_time, previous_shot = time, shot
    return pd.DataFrame(rows)


def pass_hypotheses(possession):
    result = []
    last = None
    transfer = []
    previous_time = None
    previous_shot = None
    for row in possession.to_dict("records"):
        if previous_time is not None and (
            row["time_s"] - previous_time > 0.25 or row["shot_id"] != previous_shot
        ):
            last = None
            transfer = []
        previous_time = row["time_s"]
        previous_shot = row["shot_id"]
        if row["state"] in {"unknown", "contested"}:
            last = None
            transfer = []
            continue
        if row["state"] == "free_ball":
            if last:
                transfer.append(row)
            continue
        if row["state"] != "control_candidate":
            continue
        if last and last["entity_id"] != row["entity_id"]:
            if (
                transfer
                and row["shot_id"] == last["shot_id"]
                and row["team"] == last["team"]
                and row["time_s"] - last["time_s"] <= 3
            ):
                result.append(
                    {
                        "event_id": f"pass-{len(result)}",
                        "start_s": last["time_s"],
                        "end_s": row["time_s"],
                        "source": last["entity_id"],
                        "receiver": row["entity_id"],
                        "team": row["team"],
                        "status": "candidate_requires_review",
                    }
                )
            transfer = []
        last = row
    return result


def derive_run(run):
    run = Path(run)
    raw = load_tracks(run)
    if raw.empty:
        write_json(run / "analytics_summary.json", {"status": "no_tracks"})
        return
    data = apply_review(raw, corrections(run))
    valid = eligible_positions(data)
    stats, trajectory = movement(valid)
    tactical = tactical_geometry(valid)
    possession = possession_hypotheses(data)
    events = pass_hypotheses(possession)
    event_review = corrections(run).get("events", {})
    for event in events:
        event["review_status"] = event_review.get(event["event_id"], "unreviewed")
    for name, table in [
        ("movement", stats),
        ("trajectory", trajectory),
        ("tactics", tactical),
        ("possession", possession),
    ]:
        table.to_parquet(run / f"{name}.parquet", index=False)
    write_json(run / "pass_candidates.json", events)
    total_times = read_json(run / "manifest.json", {}).get("frames", data.time_s.nunique())
    summary = {
        "status": "estimates" if len(valid) else "metric_analytics_unavailable",
        "reason": None
        if len(valid)
        else "Requires reviewed live-wide segments, verified scale and valid calibration",
        "detection_rows": len(data),
        "tracklets": int(data.tracklet_id.nunique()),
        "eligible_position_rows": len(valid),
        "sampled_times": total_times,
        "times_with_metric_positions": int(valid.time_s.nunique()),
        "metric_time_coverage": float(valid.time_s.nunique() / total_times) if total_times else 0,
        "automatic_pass_candidates": len(events),
        "reviewed_accepted_passes": sum(e["review_status"] == "accepted" for e in events),
        "event_accuracy": "not_evaluated",
    }
    write_json(run / "analytics_summary.json", summary)
    return summary
