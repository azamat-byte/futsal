"""Dominant torso HSV colors, fit-once K-Means, and bounded track histories."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .config import TeamConfig


class JerseyColorExtractor:
    """Crop the central upper half, avoiding most head, arms, shorts and background.

    Hue is circular and unreliable for unsaturated pixels. The four-dimensional
    embedding uses saturation-weighted sin/cos hue, saturation, and value. White,
    gray and black shirts are intentionally retained rather than masked away.
    """

    def crop(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            return None
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 24:
            return None
        h, w = frame.shape[:2]
        left, right = np.clip([int(x1 + .20 * bw), int(x2 - .20 * bw)], 0, w)
        top, bottom = np.clip([int(y1 + .15 * bh), int(y1 + .50 * bh)], 0, h)
        if right - left < 4 or bottom - top < 6:
            return None
        return frame[top:bottom, left:right]

    def from_crop(self, crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        hue, saturation, value = pixels.T
        # Wrap red's 0/179 boundary into the same bin. Suppress arbitrary hue in gray pixels.
        hbin = ((hue + 5) // 10).astype(int) % 18
        hbin[saturation < 30] = 0
        sbin = (saturation // 64).astype(int)
        vbin = (value // 64).astype(int)
        codes = hbin * 16 + sbin * 4 + vbin
        winner = int(np.bincount(codes, minlength=18 * 16).argmax())
        dominant = pixels[codes == winner]
        angle = dominant[:, 0] * (2 * np.pi / 180)
        weights = dominant[:, 1] / 255
        hue_angle = np.arctan2(np.sum(np.sin(angle) * weights), np.sum(np.cos(angle) * weights))
        s = float(np.median(dominant[:, 1]) / 255)
        v = float(np.median(dominant[:, 2]) / 255)
        return np.array([s * np.cos(hue_angle), s * np.sin(hue_angle), s, .6 * v], dtype=np.float32)

    def extract(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        crop = self.crop(frame, box)
        return None if crop is None else self.from_crop(crop)


class TeamClassifier:
    """Fit once on representative high-confidence player detections; never per frame."""

    def __init__(self, config: TeamConfig) -> None:
        self.config = config
        self.centers: np.ndarray | None = None
        self.counts: np.ndarray | None = None

    def fit(self, features: list[np.ndarray] | np.ndarray) -> None:
        data = np.asarray(features, dtype=np.float32)
        k = len(self.config.cluster_labels)
        if data.ndim != 2 or data.shape[1] != 4 or len(data) < self.config.min_samples:
            raise ValueError("Not enough valid jersey crops; choose a calibration segment showing both teams")
        if not np.isfinite(data).all() or len(np.unique(data, axis=0)) < k:
            raise ValueError("Calibration colors are invalid or indistinguishable")
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(data)
        centers = model.cluster_centers_
        # Color ordering removes K-Means' arbitrary label permutation; club names still require review.
        # Put the ordering seam away from pure red; tiny floating errors around
        # hue zero must not swap the first and last cluster.
        hue = np.mod(np.arctan2(centers[:, 1], centers[:, 0]) + np.pi / 18, 2 * np.pi)
        hue[centers[:, 2] < .12] = 0
        order = np.lexsort((centers[:, 3], hue))
        centers = centers[order]
        counts = np.bincount(model.labels_, minlength=k)[order]
        separations = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        separations[np.diag_indices(k)] = np.inf
        if separations.min() < self.config.min_center_distance:
            raise ValueError("Shirt clusters are too similar; choose a better calibration segment")
        if counts.min() / counts.sum() < self.config.min_cluster_fraction:
            raise ValueError("Calibration is too imbalanced; show both teams or review the cluster count")
        self.centers, self.counts = centers, counts

    def predict(self, feature: np.ndarray | None) -> str:
        if self.centers is None:
            raise RuntimeError("Fit the team classifier before processing")
        if feature is None or not np.isfinite(feature).all():
            return "unknown"
        distances = np.linalg.norm(self.centers - feature, axis=1)
        nearest = np.argsort(distances)
        if distances[nearest[0]] > self.config.max_color_distance:
            return "unknown"
        if distances[nearest[1]] - distances[nearest[0]] < self.config.min_margin:
            return "unknown"
        return self.config.cluster_labels[int(nearest[0])]

    def report(self) -> dict:
        if self.centers is None or self.counts is None:
            raise RuntimeError("Classifier is not fitted")
        return {
            "feature": "s*cos(h), s*sin(h), s, 0.6*v; OpenCV HSV; dominant torso bin",
            "clusters": [{"cluster_id": i, "label": label, "samples": int(count), "center": center.tolist()}
                         for i, (label, count, center) in enumerate(
                             zip(self.config.cluster_labels, self.counts, self.centers))],
            "review_required": True,
            "note": "Color groups do not identify clubs, referees, or goalkeeper affiliation.",
        }


@dataclass
class _History:
    votes: deque[str]
    last_frame: int


class TemporalTeamAssigner:
    """Unknown votes age out stale colors; state expires after missing tracks."""

    def __init__(self, config: TeamConfig, fps: float) -> None:
        self.config = config
        self.ttl_frames = max(1, round(config.history_ttl_seconds * fps))
        self.histories: dict[int, _History] = {}

    def prune(self, frame_index: int) -> None:
        self.histories = {key: value for key, value in self.histories.items()
                          if frame_index - value.last_frame <= self.ttl_frames}

    def reset(self) -> None:
        self.histories.clear()

    def update(self, track_id: int, candidate: str, frame_index: int) -> tuple[str, float]:
        state = self.histories.get(track_id)
        if state is None or frame_index - state.last_frame > self.ttl_frames:
            state = _History(deque(maxlen=self.config.history_size), frame_index)
            self.histories[track_id] = state
        state.votes.append(candidate)
        state.last_frame = frame_index
        label, votes = Counter(state.votes).most_common(1)[0]
        agreement = votes / len(state.votes)
        if label == "unknown" or votes < self.config.min_votes or agreement < self.config.vote_fraction:
            return "unknown", 0.0
        return label, agreement


def calibration_montage(classifier: TeamClassifier, features: list[np.ndarray], crops: list[np.ndarray]) -> np.ndarray:
    """Representative real torso crops near each center, with explicit cluster indices."""
    if classifier.centers is None:
        raise RuntimeError("Classifier is not fitted")
    data = np.asarray(features)
    membership = np.linalg.norm(data[:, None] - classifier.centers[None, :], axis=2).argmin(axis=1)
    canvas = np.full((len(classifier.centers) * 140, 800, 3), 28, dtype=np.uint8)
    for index, center in enumerate(classifier.centers):
        y = index * 140
        label = classifier.config.cluster_labels[index]
        cv2.putText(canvas, f"Cluster {index}: {label} (review)", (12, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
        candidates = np.flatnonzero(membership == index)
        nearest = candidates[np.argsort(np.linalg.norm(data[candidates] - center, axis=1))[:8]]
        for column, sample in enumerate(nearest):
            x = 12 + column * 96
            canvas[y + 38:y + 126, x:x + 80] = cv2.resize(crops[int(sample)], (80, 88))
    return canvas
