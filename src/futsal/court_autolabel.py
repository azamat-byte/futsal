"""Automatic court-keypoint labelling: model proposals -> homography -> line-alignment refinement.

A homography is accepted only when the projected futsal markings sit on painted white lines in the
image (independent of the network's confidence). Labels for all schema landmarks are then obtained by
projecting the template, so every landmark that falls inside the image is labelled visible even if a
player stands on it. These are *derived* labels: review samples before trusting them for evaluation.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares


def load_schema(path):
    schema = json.loads(Path(path).read_text("utf-8"))
    schema["court_points"] = np.asarray(schema["court_points"], float)
    return schema


def wireframe(length=40.0, width=20.0, step=0.25):
    """Painted markings as {name: Nx2 metre samples}; goal frames are not painted so are excluded."""
    paths = {
        "far_touch": [[0, 0], [length, 0]],
        "near_touch": [[0, width], [length, width]],
        "left_goal_line": [[0, 0], [0, width]],
        "right_goal_line": [[length, 0], [length, width]],
        "halfway": [[length / 2, 0], [length / 2, width]],
    }
    out = {}
    for name, pts in paths.items():
        a, b = np.asarray(pts, float)
        n = max(2, int(np.hypot(*(b - a)) / step))
        out[name] = a + (b - a) * np.linspace(0, 1, n)[:, None]
    t = np.linspace(0, 2 * np.pi, int(2 * np.pi * 3 / step))
    out["center_circle"] = np.c_[length / 2 + 3 * np.cos(t), width / 2 + 3 * np.sin(t)]
    cy = width / 2
    for end, sign, tag in ((0.0, 1.0, "left"), (length, -1.0, "right")):
        for post_y, a0, a1, side in ((cy - 1.58, -np.pi / 2, 0.0, "far"), (cy + 1.58, 0.0, np.pi / 2, "near")):
            t = np.linspace(a0, a1, int(0.5 * np.pi * 6 / step))
            out[f"{tag}_arc_{side}"] = np.c_[end + sign * 6 * np.cos(t), post_y + 6 * np.sin(t)]
        a, b = np.array([end + sign * 6, cy - 1.58]), np.array([end + sign * 6, cy + 1.58])
        n = int(3.16 / step) + 2
        out[f"{tag}_penalty_line"] = a + (b - a) * np.linspace(0, 1, n)[:, None]
    return out


def line_distance_map(image):
    """Distance (px) to the nearest bright thin structure; returns (dt, mask)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = gray.shape[1]
    k = int(round(13 * width / 960)) | 1
    k = max(7, k)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    top = cv2.subtract(gray, opened)
    mask = (top > 22).astype(np.uint8)
    # remove specks (text, noise) but keep thin long lines
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= max(12, width // 80)) | (
        np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= width // 40
    )
    mask = keep[labels].astype(np.uint8)
    dt = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
    return dt, mask


def apply_h(H, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    h = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = h[:, :2] / h[:, 2:3]
    return xy, h[:, 2]


class Refiner:
    """Refine court->image homography by aligning the wireframe with detected white lines."""

    ANCHORS = np.array([[10.0, 4.0], [30.0, 4.0], [30.0, 16.0], [10.0, 16.0]])

    def __init__(self, image, length=40.0, width=20.0):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.scale = self.w / 960.0
        self.dt, self.mask = line_distance_map(image)
        self.wire = wireframe(length, width)
        self.names = list(self.wire)
        self.samples = np.vstack([self.wire[n] for n in self.names])
        self.owner = np.concatenate([[i] * len(self.wire[n]) for i, n in enumerate(self.names)])

    def _inside(self, H, margin=4):
        xy, w = apply_h(H, self.samples)
        ok = (w > 1e-6) & np.isfinite(xy).all(1)
        ok &= (xy[:, 0] >= margin) & (xy[:, 0] <= self.w - 1 - margin)
        ok &= (xy[:, 1] >= margin) & (xy[:, 1] <= self.h - 1 - margin)
        return xy, ok

    def _dt(self, xy):
        return map_coordinates(self.dt, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")

    def _h_from_disp(self, base_img, d):
        return cv2.getPerspectiveTransform(self.ANCHORS.astype(np.float32),
                                           (base_img + d.reshape(4, 2)).astype(np.float32))

    def refine(self, H0, prior_xy=None, prior_court=None, prior_w=0.3, rounds=3):
        H = np.asarray(H0, float)
        base = apply_h(H, self.ANCHORS)[0]
        if not np.isfinite(base).all() or np.abs(base).max() > 20 * max(self.w, self.h):
            return None
        cap = 10.0 * self.scale
        d = np.zeros(8)
        for _ in range(rounds):
            Hc = self._h_from_disp(base, d)
            _, ok = self._inside(Hc)
            if ok.sum() < 60:
                return None
            pts = self.samples[ok]
            base_d = d.copy()

            def resid(x):
                Hx = self._h_from_disp(base, x)
                xy, w = apply_h(Hx, pts)
                bad = (w <= 1e-6) | ~np.isfinite(xy).all(1)
                xy = np.nan_to_num(xy, nan=-1e4, posinf=1e4, neginf=-1e4)
                xy[:, 0] = np.clip(xy[:, 0], 0, self.w - 1)
                xy[:, 1] = np.clip(xy[:, 1], 0, self.h - 1)
                r = np.minimum(self._dt(xy), cap)
                r[bad] = cap
                parts = [r]
                if prior_xy is not None and len(prior_xy):
                    pxy, pw = apply_h(Hx, prior_court)
                    pr = ((pxy - prior_xy) * prior_w).ravel()
                    pr = np.nan_to_num(pr, nan=cap, posinf=cap, neginf=-cap)
                    parts.append(np.clip(pr, -cap * 3, cap * 3))
                return np.concatenate(parts)

            try:
                sol = least_squares(resid, base_d, loss="soft_l1", f_scale=2.0 * self.scale,
                                    max_nfev=60, x_scale=10.0)
            except (ValueError, np.linalg.LinAlgError):
                return None
            d = sol.x
        return self._h_from_disp(base, d), sol

    def score(self, H):
        xy, ok = self._inside(H)
        if ok.sum() < 20:
            return {"inside": int(ok.sum()), "close": 0.0, "mean_dt": 99.0, "lines": 0, "spread": 0.0}
        dist = self._dt(xy[ok])
        s = self.scale
        close = float(np.mean(dist <= 2.5 * s))
        lines = 0
        owner = self.owner[ok]
        per_line = {}
        for o, dd in zip(owner, dist):
            per_line.setdefault(o, []).append(dd)
        for o, dd in per_line.items():
            if len(dd) >= 12 and np.mean(np.asarray(dd) <= 3.0 * s) >= 0.55:
                lines += 1
        pts = xy[ok]
        spread = float(max(np.ptp(pts[:, 0]) / self.w, np.ptp(pts[:, 1]) / self.h))
        return {"inside": int(ok.sum()), "close": close, "mean_dt": float(np.mean(np.minimum(dist, 10 * s)) / s),
                "lines": lines, "spread": spread}


def init_homography(kxy, kconf, court, thr_conf=0.35, thr_px=6.0, min_pts=5):
    """RANSAC court->image homography from detected keypoints (pixel coords)."""
    use = np.flatnonzero((kconf >= thr_conf) & (kxy[:, 0] > 1) & (kxy[:, 1] > 1))
    if len(use) < min_pts:
        return None, None, None
    src, dst = court[use], kxy[use]
    if np.linalg.matrix_rank(src - src.mean(0)) < 2:
        return None, None, None
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, thr_px)
    if H is None or inl is None or inl.sum() < min_pts:
        return None, None, None
    inl = inl.ravel().astype(bool)
    good = src[inl]
    if np.linalg.matrix_rank(good - good.mean(0)) < 2:
        return None, None, None
    return H, dst[inl], good


def project_labels(H, court, width, height, margin=0):
    """All schema landmarks -> (N,3) pixel xy + visibility (2 inside the frame, else 0)."""
    xy, w = apply_h(H, court)
    out = np.zeros((len(court), 3))
    for i, ((x, y), ww) in enumerate(zip(xy, w)):
        if ww > 1e-6 and np.isfinite([x, y]).all() and margin <= x <= width - 1 - margin and margin <= y <= height - 1 - margin:
            out[i] = [x, y, 2]
    return out
