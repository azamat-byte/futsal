import cv2
import numpy as np
from sklearn.cluster import KMeans


def jersey_feature(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, float)
    bw, bh = x2 - x1, y2 - y1
    crop = image[
        max(0, int(y1 + bh * 0.18)) : min(h, int(y1 + bh * 0.55)),
        max(0, int(x1 + bw * 0.2)) : min(w, int(x2 - bw * 0.2)),
    ]
    if crop.size < 60 or bh < 25:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    mask = (hsv[:, :, 2] > 35) & (hsv[:, :, 2] < 250)
    if mask.sum() < 20:
        return None
    # Robust color prototype; unlabeled clustering is explicitly experimental.
    return np.median(lab[mask], axis=0) / 255


class TeamClassifier:
    def __init__(self):
        self.samples = []
        self.model = None
        self.history = {}

    def fit(self, features):
        data = np.asarray(features)
        if len(data) < 20:
            return False
        model = KMeans(2, n_init=10, random_state=42).fit(data)
        if np.linalg.norm(model.cluster_centers_[0] - model.cluster_centers_[1]) < 0.12:
            return False
        # Deterministic cluster ordering; actual club mapping still needs review.
        order = np.argsort(model.cluster_centers_[:, 1])
        model.cluster_centers_ = model.cluster_centers_[order]
        self.model = model
        return True

    def predict(self, feature, tracklet):
        if feature is None:
            return "unknown", None
        history = self.history.setdefault(tracklet, [])
        history.append(feature)
        del history[:-15]
        if self.model is None:
            if len(self.samples) < 1000:
                self.samples.append(feature)
            if len(self.samples) >= 60:
                self.fit(self.samples)
        if self.model is None or len(history) < 3:
            return "unknown", None
        vector = np.median(history, axis=0)
        distance = np.linalg.norm(self.model.cluster_centers_ - vector, axis=1)
        order = np.argsort(distance)
        margin = float(distance[order[1]] - distance[order[0]])
        if distance[order[0]] > 0.28 or margin < 0.05:
            return "unknown", margin
        return ["team_a", "team_b"][order[0]], margin
