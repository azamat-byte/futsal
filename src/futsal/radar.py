"""Metric court geometry and synchronized raster radar, with explicit missing calibration."""

from collections import defaultdict, deque

import cv2
import numpy as np


def court_paths(length=40, width=20):
    """FIFA futsal markings in metres; 8 cm posts, 3 m clear goal opening.

    Penalty quarter circles start at the outside of each post (3.16 m joining line).
    Coordinates follow x along the length and y along the width.
    """
    if length < 25 or width < 16:
        raise ValueError("This standard marking template needs length >=25 and width >=16 m")
    paths = [np.array([[0, 0], [length, 0], [length, width], [0, width], [0, 0]]),
             np.array([[length / 2, 0], [length / 2, width]])]
    theta = np.linspace(0, 2 * np.pi, 100)
    paths.append(np.c_[length / 2 + 3 * np.cos(theta), width / 2 + 3 * np.sin(theta)])
    for end, sign in ((0, 1), (length, -1)):
        for y, angles in ((width / 2 - 1.58, (-np.pi / 2, 0)),
                          (width / 2 + 1.58, (0, np.pi / 2))):
            theta = np.linspace(*angles, 50)
            paths.append(np.c_[end + sign * 6 * np.cos(theta), y + 6 * np.sin(theta)])
        paths.append(np.array([[end + sign * 6, width / 2 - 1.58],
                               [end + sign * 6, width / 2 + 1.58]]))
        paths.append(np.array([[end, width / 2 - 1.5], [end - sign * .65, width / 2 - 1.5],
                               [end - sign * .65, width / 2 + 1.5], [end, width / 2 + 1.5]]))
    marks = np.array([[length / 2, width / 2], [6, width / 2], [10, width / 2],
                      [length - 6, width / 2], [length - 10, width / 2]])
    return paths, marks


def palette(colors=None):
    defaults = {"team_a": "#3498ff", "team_b": "#f05252", "unknown": "#a0a0a0",
                "referee": "#ffe04b", "ball": "#ff9f35"}
    defaults.update(colors or {})
    result = {}
    for name, color in defaults.items():
        if len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"Invalid hex visualization color: {color}")
        result[name] = tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))
    return result


class RadarRenderer:
    def __init__(self, length=40, width=20, colors=None, show_ids=True, show_trails=False):
        self.length, self.width = length, width
        self.scale, self.pad, self.header = 16, 24, 68
        self.colors = palette(colors)
        self.show_ids, self.show_trails = show_ids, show_trails
        self.paths, self.marks = court_paths(length, width)
        self.history = defaultdict(lambda: deque(maxlen=20))
        self.previous_shot = None

    def pixels(self, points):
        return np.rint(np.asarray(points) * self.scale + [self.pad, self.pad + self.header]).astype(int)

    def render(self, rows, time_s, shot, calibration):
        if shot != self.previous_shot or not calibration.get("valid"):
            self.history.clear()
        self.previous_shot = shot
        shape = (round(self.width * self.scale + 2 * self.pad + self.header),
                 round(self.length * self.scale + 2 * self.pad), 3)
        out = np.full(shape, (31, 27, 17), dtype=np.uint8)
        for path in self.paths:
            cv2.polylines(out, [self.pixels(path)], False, (170, 188, 185), 1, cv2.LINE_AA)
        for point in self.pixels(self.marks):
            cv2.circle(out, tuple(point), 2, (190, 205, 200), -1, cv2.LINE_AA)
        cv2.putText(out, f"FUTSAL LAB   |   {time_s:.2f}s   |   shot {shot}", (20, 25), 0, .56,
                    (235, 240, 240), 1, cv2.LINE_AA)
        if not calibration.get("valid"):
            self.history.clear()
            cv2.putText(out, "Positions unavailable: " + calibration.get("reason", "no_calibration"),
                        (20, 51), 0, .45, (90, 180, 255), 1, cv2.LINE_AA)
            return out
        unit = "verified scale" if calibration.get("scale_verified") else "assumed dimensions; no metric stats"
        cv2.putText(out, f"Visible detections only | {unit}", (20, 51), 0, .45,
                    (160, 210, 210), 1, cv2.LINE_AA)
        for row in rows:
            x, y = row.get("court_x"), row.get("court_y")
            if not row.get("calibration_valid") or x is None or y is None or not np.isfinite([x, y]).all():
                continue
            if not (0 <= x <= self.length and 0 <= y <= self.width):
                continue
            color = self.colors["ball"] if row["class_name"] == "ball" else self.colors[
                "referee" if row.get("role") == "referee" else row.get("team", "unknown")]
            point = tuple(self.pixels([[x, y]])[0])
            identity = row.get("tracklet_id")
            if identity and self.show_trails:
                trail = self.history[identity]
                if trail and time_s - trail[-1][0] > .25:
                    trail.clear()
                trail.append((time_s, point))
                if len(trail) > 1:
                    cv2.polylines(out, [np.array([p for _, p in trail])], False, color, 1, cv2.LINE_AA)
            cv2.circle(out, point, 6, color, -1, cv2.LINE_AA)
            cv2.circle(out, point, 6, (240, 240, 240), 1, cv2.LINE_AA)
            if self.show_ids and identity:
                cv2.putText(out, identity.split(":")[-1], (point[0] + 8, point[1] - 5),
                            0, .36, color, 1, cv2.LINE_AA)
        return out


def inset_radar(frame, radar, fraction=.36):
    width = max(2, int(frame.shape[1] * fraction) // 2 * 2)
    height = max(2, round(radar.shape[0] * width / radar.shape[1]))
    height = min(height, frame.shape[0] - 20)
    small = cv2.resize(radar, (width, height), interpolation=cv2.INTER_AREA)
    out = frame.copy()
    out[-height - 10:-10, -width - 10:-10] = small
    return out
