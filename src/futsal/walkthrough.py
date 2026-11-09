"""Readable visual diagnostics used by the companion notebook; no training or new detector."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .models import class_kind
from .teams import jersey_feature
from .video import load_image

TEAM_COLORS = {"team_a": "#3498ff", "team_b": "#f05252", "unknown": "#aaaaaa"}


def crop_box(image, box, pad=0):
    x1, y1, x2, y2 = np.asarray(box, int)
    height, width = image.shape[:2]
    return image[max(0, y1-pad):min(height, y2+pad), max(0, x1-pad):min(width, x2+pad)].copy()


def overlay(image, detections, kind=None):
    out = image.copy()
    for d in detections:
        category = d.get("class_name", class_kind(d.get("model_class", ""))[0])
        if kind and category != kind:
            continue
        x1, y1, x2, y2 = np.asarray(d["box"], int)
        color = (0, 160, 255) if category == "ball" else (255, 180, 40)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.get('model_class', category)} {d['score']:.2f}"
        label_y = max(24, y1-6)
        (text_width, text_height), baseline = cv2.getTextSize(label, 0, .75, 2)
        cv2.rectangle(out, (x1, label_y-text_height-4), (x1+text_width+4, label_y+baseline), (25,25,25), -1)
        cv2.putText(out, label, (x1+2, label_y), 0, .75, color, 2, cv2.LINE_AA)
    return out


def gather_crops(records, directory, score=.25):
    """Keep role provenance, timestamps and box locations; never force GK candidates into teams."""
    rows, crops = [], []
    for frame in records:
        image = load_image(Path(directory) / frame["image"])
        for detection in frame["detections"]:
            if detection["class_name"] != "person" or detection["score"] < score:
                continue
            feature = jersey_feature(image, detection["box"])
            crop = crop_box(image, detection["box"])
            if feature is None or not crop.size:
                continue
            rows.append({"time_s": frame["time_s"], "role": detection["role"],
                         "model_class": detection["model_class"], "score": detection["score"],
                         "box": detection["box"], "feature": feature.tolist()})
            crops.append(crop)
    return pd.DataFrame(rows), crops


def team_overlay(image, rows):
    """Display per-crop appearance assignment, keeping model roles and unknown teams visible."""
    out = image.copy()
    for row in rows.itertuples():
        color_hex = TEAM_COLORS[row.team].lstrip("#")
        color = tuple(int(color_hex[i:i+2],16) for i in (4,2,0))
        x1,y1,x2,y2 = np.asarray(row.box,int)
        label = f"{row.team} / {row.model_class}"
        cv2.rectangle(out,(x1,y1),(x2,y2),color,3)
        y = max(24,y1-6)
        (width,height),baseline = cv2.getTextSize(label,0,.65,2)
        cv2.rectangle(out,(x1,y-height-4),(x1+width+4,y+baseline),(25,25,25),-1)
        cv2.putText(out,label,(x1+2,y),0,.65,color,2,cv2.LINE_AA)
    return out


def cluster_outfield(rows):
    """LAB KMeans reference prototypes with ambiguity rejection, like the active color pipeline."""
    data = rows.copy()
    data["team"] = "unknown"
    data["cluster_margin"] = np.nan
    eligible = data.role.eq("outfield_player")
    features = np.asarray(data.loc[eligible, "feature"].tolist())
    if len(features) < 12 or len(np.unique(features, axis=0)) < 2:
        return data, None, {"status": "insufficient_outfield_crops", "accuracy": None}
    model = KMeans(2, n_init=10, random_state=42).fit(features)
    model.cluster_centers_ = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 1])]
    distance = np.linalg.norm(features[:, None, :] - model.cluster_centers_[None, :, :], axis=2)
    labels = np.argmin(distance, axis=1)
    margin = np.abs(distance[:, 0] - distance[:, 1])
    separated = np.linalg.norm(model.cluster_centers_[0] - model.cluster_centers_[1]) >= .12
    accepted = (distance.min(axis=1) <= .28) & (margin >= .05) & separated
    teams = np.array(["team_a", "team_b"])[labels].astype(object)
    teams[~accepted] = "unknown"
    data.loc[eligible, "team"] = teams
    data.loc[eligible, "cluster_margin"] = margin
    silhouette = float(silhouette_score(features, labels)) if len(np.unique(labels)) == 2 else None
    return data, model, {"status": "unverified_appearance_clusters", "silhouette": silhouette,
                         "accepted_crops": int(accepted.sum()), "accuracy": None}


def goalkeeper_groups(rows):
    """Appearance groups among detector-labelled GK candidates, not role/team truth."""
    data = rows.loc[rows.role.eq("goalkeeper")].copy()
    if len(data) < 4 or len(np.unique(np.asarray(data.feature.tolist()), axis=0)) < 2:
        data["kit_group"] = "unavailable"
        return data
    features = np.asarray(data.feature.tolist())
    labels = KMeans(2, n_init=10, random_state=42).fit_predict(features)
    data["kit_group"] = [f"kit_{i}" for i in labels]
    # Deliberately no kit_group -> team mapping: GK kits can differ from both outfield kits.
    return data


def track_diagnostics(data):
    people = data.loc[data.class_name.eq("person")].dropna(subset=["tracklet_id"])
    if people.empty:
        return pd.DataFrame(columns=["observations", "first_s", "last_s", "observed_span_s"])
    grouped = people.groupby("tracklet_id").time_s.agg(observations="size", first_s="min", last_s="max")
    grouped["observed_span_s"] = grouped.last_s - grouped.first_s
    return grouped.sort_values("observations", ascending=False)
