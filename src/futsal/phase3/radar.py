"""2D Futsal Radar extended with Voronoi Pitch Control cells and Team Convex Hulls."""

from __future__ import annotations

import cv2
import numpy as np

from futsal.phase2.radar import COLORS, PitchRadar

TEAM_FILL_BGR = {
    "team_a": (255, 191, 0),    # Cyan / Blue
    "team_b": (129, 64, 255),   # Pink / Magenta
}


class SpatialRadar(PitchRadar):
    """Renders Pitch Control Voronoi cells, Team Convex Hulls, and player dots on the 40x20m radar."""

    def render_spatial(
        self,
        observations: list[dict] = (),
        spatial: dict | None = None,
        status: str = "anchor",
        units: str = "assumed_m",
        voronoi_alpha: float = 0.25,
        hull_alpha: float = 0.20,
    ) -> np.ndarray:
        canvas = self.render(observations=(), status=status, units=units)
        spatial = spatial or {}

        # 1. Draw pitch-clipped Voronoi cells (Pitch Control)
        cells = spatial.get("cells", [])
        if cells and spatial.get("control_valid", False):
            overlay = canvas.copy()
            for cell in cells:
                poly = cell.get("polygon", [])
                team = cell.get("team")
                if len(poly) >= 3 and team in TEAM_FILL_BGR:
                    pts = np.array([self.point(x, y) for x, y in poly], dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], TEAM_FILL_BGR[team])
                    cv2.polylines(canvas, [pts], True, (200, 215, 205), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, voronoi_alpha, canvas, 1.0 - voronoi_alpha, 0, canvas)

        # 2. Draw Team Convex Hulls (Team Compactness)
        hulls = spatial.get("hulls", [])
        if hulls:
            hull_overlay = canvas.copy()
            for hull in hulls:
                poly = hull.get("polygon", [])
                team = hull.get("team")
                color = TEAM_FILL_BGR.get(team, (200, 200, 200))
                if len(poly) >= 3:
                    pts = np.array([self.point(x, y) for x, y in poly], dtype=np.int32)
                    cv2.fillPoly(hull_overlay, [pts], color)
                    cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
                elif len(poly) == 2:
                    p1, p2 = self.point(*poly[0]), self.point(*poly[1])
                    cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
            cv2.addWeighted(hull_overlay, hull_alpha, canvas, 1.0 - hull_alpha, 0, canvas)

        # 3. Draw player dots and IDs on top of tactical layers
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
            raw_j = row.get("jersey_number")
            disp_id = (
                str(int(float(raw_j)))
                if raw_j not in ("", None) and not (isinstance(raw_j, float) and np.isnan(raw_j))
                else str(row["track_id"])
            )
            cv2.putText(
                canvas,
                disp_id,
                (p[0] + 6, p[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 4. Pitch Control % header badge
        ctrl_a = spatial.get("team_a_control_pct")
        if ctrl_a is not None:
            ctrl_b = 100.0 - ctrl_a
            badge = f"Control A: {ctrl_a:.1f}% | B: {ctrl_b:.1f}%"
            cv2.putText(
                canvas,
                badge,
                (canvas.shape[1] - 245, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        return canvas
