"""A metric futsal floor diagram, player dots, and a compact video overlay."""

import cv2
import numpy as np

COLORS = {"team_a": (255, 191, 0), "team_b": (129, 64, 255), "goalkeeper": (255, 136, 179),
          "referee": (64, 215, 255), "other": (0, 145, 255), "unknown": (197, 190, 176)}


class PitchRadar:
    def __init__(self, length=40.0, width=20.0, scale=12):
        self.length, self.width, self.scale = length, width, scale
        self.margin, self.header = 24, 48
        self.size = (int(length * scale) + 2*self.margin, int(width * scale) + self.header + self.margin)

    def point(self, x, y):
        return (round(self.margin + x*self.scale), round(self.header + y*self.scale))

    def render(self, observations=(), status="anchor", units="assumed_m"):
        w, h = self.size
        canvas = np.full((h, w, 3), (28, 33, 35), np.uint8)
        cv2.rectangle(canvas, self.point(0, 0), self.point(self.length, self.width), (66, 94, 64), -1)
        line = (213, 225, 218)
        cv2.rectangle(canvas, self.point(0, 0), self.point(self.length, self.width), line, 1)
        cv2.line(canvas, self.point(self.length/2, 0), self.point(self.length/2, self.width), line, 1)
        cv2.circle(canvas, self.point(self.length/2, self.width/2), round(3*self.scale), line, 1, cv2.LINE_AA)
        # Two 6 m quarter-circles joined by the line parallel to each 3 m goal mouth.
        top, bottom = self.width/2 - 1.5, self.width/2 + 1.5
        for x, direction in ((0, 1), (self.length, -1)):
            a = np.linspace(-np.pi/2, 0, 50)
            b = np.linspace(0, np.pi/2, 50)
            curve = [(x + direction*6*np.cos(t), top + 6*np.sin(t)) for t in a]
            curve += [(x + direction*6, bottom)]
            curve += [(x + direction*6*np.cos(t), bottom + 6*np.sin(t)) for t in b]
            cv2.polylines(canvas, [np.array([self.point(*p) for p in curve])], False, line, 1, cv2.LINE_AA)
            for distance in (6, 10):
                cv2.circle(canvas, self.point(x + direction*distance, self.width/2), 2, line, -1)
            cv2.rectangle(canvas, self.point(x, top), self.point(x - direction*.8, bottom), line, 1)
        for row in observations:
            if not row.get("position_valid", False):
                continue
            role, team = row["role"], row["team"]
            group = (
                team if (role == "goalkeeper" and team in {"team_a", "team_b"})
                else (role if role in {"goalkeeper", "referee"} else team)
            )
            color = COLORS.get(group, COLORS["unknown"])
            p = self.point(row["pitch_x_m"], row["pitch_y_m"])
            cv2.circle(canvas, p, 5, (15, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(row["track_id"]), (p[0]+6, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (255, 255, 255), 1, cv2.LINE_AA)
        title = f"RADAR | {self.length:g} x {self.width:g} m"
        if units == "assumed_m":
            title += " (assumed)"
        cv2.putText(canvas, title, (12, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
        good = status in {"anchor", "optical_flow"}
        caption = status if good else f"No calibration: {status}"
        cv2.putText(canvas, caption, (12, 37), cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (170, 225, 170) if good else (80, 190, 255), 1, cv2.LINE_AA)
        return canvas

    def overlay(self, frame, radar):
        scene = frame.copy()
        h, w = frame.shape[:2]
        ratio = min(.32*w/radar.shape[1], .35*h/radar.shape[0], 1)
        small = cv2.resize(radar, (max(1, round(radar.shape[1]*ratio)), max(1, round(radar.shape[0]*ratio))))
        sh, sw = small.shape[:2]
        margin = min(12, max(0, (min(w-sw, h-sh))//2))
        scene[h-sh-margin:h-margin, w-sw-margin:w-margin] = small
        return scene
