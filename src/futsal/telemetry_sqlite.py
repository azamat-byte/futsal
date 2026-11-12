"""Epics 4 & 5: SQLite Telemetry Streaming (`TelemetrySQLiteStreamer`) & Temporal Shot Detection (`ShotEventDetector`).

Epic 4 (`TelemetrySQLiteStreamer`):
- Replaces monolithic JSON dumps with real-time SQLite WAL streaming (`player_telemetry` & `match_events` tables).
- Provides `PLAYER_SUMMARY_BY_JERSEY_SQL` and `query_player_match_totals()` to group by persistent `(team, jersey_number)`
  (excluding inactive bench/sub-zone rows) and compute true total distance covered and top sprint speed across the match.

Epic 5 (`ShotEventDetector`):
- Temporal state machine that detects when a player performs a shooting action (high acceleration / ball strike
  near the attacking penalty area) followed by the ball intersecting the 3m goal net coordinates (`x <= 0.8m` or
  `x >= 39.2m`, `y in [8.2m, 11.8m]`), logging `"Shot Made"` (or `"Shot Attempt"`) to SQLite in real time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from futsal.player_id_ocr import normalize_team_key

PLAYER_SUMMARY_BY_JERSEY_SQL = """
SELECT
    team,
    jersey_number,
    MAX(player_name) AS player_name,
    COUNT(DISTINCT tracker_id) AS stitched_tracks_count,
    GROUP_CONCAT(DISTINCT tracker_id) AS tracker_ids,
    COUNT(*) AS active_frames,
    ROUND(COALESCE(SUM(CASE WHEN is_active = 1 THEN distance_step_m ELSE 0.0 END), 0.0), 2) AS total_distance_m,
    ROUND(COALESCE(MAX(CASE WHEN is_active = 1 THEN speed_kmh ELSE NULL END), 0.0), 1) AS top_sprint_speed_kmh,
    ROUND(COALESCE(AVG(CASE WHEN is_active = 1 THEN speed_kmh ELSE NULL END), 0.0), 1) AS avg_speed_kmh,
    SUM(CASE WHEN is_active = 1 AND possession_state = 'controlled' AND is_ball_carrier = 1 THEN 1 ELSE 0 END) AS possession_frames
FROM player_telemetry
WHERE jersey_number IS NOT NULL
  AND is_active = 1
GROUP BY team, jersey_number
ORDER BY team ASC, total_distance_m DESC;
"""


class TelemetrySQLiteStreamer:
    """Real-time SQLite telemetry & event streamer using Write-Ahead Logging (WAL)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS player_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                timestamp_s REAL NOT NULL,
                segment_id INTEGER NOT NULL,
                tracker_id INTEGER NOT NULL,
                team TEXT NOT NULL,
                jersey_number INTEGER,
                player_name TEXT,
                role TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                pitch_x_m REAL,
                pitch_y_m REAL,
                speed_kmh REAL,
                distance_step_m REAL,
                possession_state TEXT,
                is_ball_carrier INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_team_jersey
            ON player_telemetry (team, jersey_number, is_active, frame_id);
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS match_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                timestamp_s REAL NOT NULL,
                event_type TEXT NOT NULL,
                team TEXT NOT NULL,
                tracker_id INTEGER,
                jersey_number INTEGER,
                player_name TEXT,
                start_x_m REAL,
                start_y_m REAL,
                goal_x_m REAL,
                goal_y_m REAL,
                peak_speed_kmh REAL,
                ball_speed_mps REAL
            );
            """
        )
        self.conn.commit()

    def stream_frame(
        self,
        frame_id: int,
        timestamp_s: float,
        rows: list[dict[str, Any]],
        possession: dict[str, Any] | None = None,
        segment_id: int = 0,
    ) -> int:
        """Insert a single frame's player telemetry rows into `player_telemetry` in a single transaction."""
        if not rows:
            return 0
        poss = possession or {}
        poss_state = str(poss.get("state") or row_default_poss(rows))
        owner_id = poss.get("owner_id")

        records = []
        for r in rows:
            tid = int(r["track_id"])
            raw_j = r.get("jersey_number")
            j_num = (
                int(float(raw_j))
                if raw_j not in ("", None) and not (isinstance(raw_j, float) and np.isnan(raw_j))
                else None
            )
            # Use jersey_number directly as the stitched tracker_id when resolved
            stitched_tid = j_num if j_num is not None else tid
            px = _opt_float(r.get("pitch_x_m"))
            py = _opt_float(r.get("pitch_y_m"))
            spd = _opt_float(r.get("speed_kmh"))
            step = _opt_float(r.get("distance_step_m"))
            is_act = 1 if bool(r.get("is_active", True)) else 0
            row_owner = _opt_int(r.get("possession_owner_id", owner_id))
            raw_in_poss = r.get("in_possession", False)
            in_poss_flag = (
                bool(raw_in_poss)
                if not (isinstance(raw_in_poss, float) and np.isnan(raw_in_poss))
                else False
            )
            in_poss = in_poss_flag or (row_owner is not None and row_owner == tid)
            records.append(
                (
                    int(frame_id),
                    float(timestamp_s),
                    int(r.get("segment_id", segment_id)),
                    stitched_tid,
                    normalize_team_key(r.get("team")),
                    j_num,
                    r.get("player_name"),
                    str(r.get("role", "player")),
                    is_act,
                    px,
                    py,
                    spd,
                    step,
                    str(r.get("possession_state") or poss_state),
                    1 if in_poss else 0,
                )
            )

        self.conn.executemany(
            """
            INSERT INTO player_telemetry (
                frame_id, timestamp_s, segment_id, tracker_id, team,
                jersey_number, player_name, role, is_active,
                pitch_x_m, pitch_y_m, speed_kmh, distance_step_m,
                possession_state, is_ball_carrier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            records,
        )
        self.conn.commit()
        return len(records)

    def log_event(self, event: dict[str, Any]) -> int:
        """Insert a tactical or shot event into `match_events`."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO match_events (
                frame_id, timestamp_s, event_type, team, tracker_id,
                jersey_number, player_name, start_x_m, start_y_m,
                goal_x_m, goal_y_m, peak_speed_kmh, ball_speed_mps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(event["frame_id"]),
                float(event["timestamp_s"]),
                str(event["event_type"]),
                normalize_team_key(event.get("team")),
                _opt_int(event.get("tracker_id")),
                _opt_int(event.get("jersey_number")),
                event.get("player_name"),
                _opt_float(event.get("start_x_m")),
                _opt_float(event.get("start_y_m")),
                _opt_float(event.get("goal_x_m")),
                _opt_float(event.get("goal_y_m")),
                _opt_float(event.get("peak_speed_kmh")),
                _opt_float(event.get("ball_speed_mps")),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def query_player_match_totals(self) -> list[dict[str, Any]]:
        """Execute `PLAYER_SUMMARY_BY_JERSEY_SQL` grouping by persistent `(team, jersey_number)`."""
        cur = self.conn.cursor()
        cur.execute(PLAYER_SUMMARY_BY_JERSEY_SQL)
        return [dict(row) for row in cur.fetchall()]

    def query_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        if event_type:
            cur.execute("SELECT * FROM match_events WHERE event_type = ? ORDER BY frame_id ASC;", (event_type,))
        else:
            cur.execute("SELECT * FROM match_events ORDER BY frame_id ASC;")
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> TelemetrySQLiteStreamer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _opt_float(val: Any) -> float | None:
    if val in ("", None):
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _opt_int(val: Any) -> int | None:
    f = _opt_float(val)
    return int(round(f)) if f is not None else None


def row_default_poss(rows: list[dict[str, Any]]) -> str:
    for r in rows:
        if r.get("possession_state"):
            return str(r["possession_state"])
    return "unknown"


@dataclass
class PendingShotCandidate:
    """Tracks an initiated shot action waiting for goal-net intersection confirmation."""

    frame_id: int
    timestamp_s: float
    tracker_id: int
    team: str
    jersey_number: int | None
    player_name: str | None
    start_x_m: float
    start_y_m: float
    peak_speed_kmh: float
    ball_speed_mps: float


class ShotEventDetector:
    """Epic 5: Temporal Shot Action & Goal-Net Intersection Detector.

    1. Initiation: Detects a player in possession (or within `strike_radius_m` of the ball)
       near the penalty area (`x <= penalty_zone_x_m` or `x >= length - penalty_zone_x_m`)
       exhibiting high acceleration / sprint speed (`speed_kmh >= min_shooter_speed_kmh`)
       or imparting high ball velocity (`ball_speed_mps >= min_ball_speed_mps`).
    2. Confirmation: Within `max_flight_s` (default 1.5s), if the 2D ball coordinate `(ball_x_m, ball_y_m)`
       intersects either 3m goal mouth (`x <= goal_depth_tol_m` or `x >= length - goal_depth_tol_m`,
       `y in [goal_y_min_m, goal_y_max_m]`), emits a `"Shot Made"` event (and logs to SQLite if attached).
    """

    def __init__(
        self,
        pitch_length_m: float = 40.0,
        pitch_width_m: float = 20.0,
        penalty_zone_x_m: float = 12.0,
        goal_y_min_m: float = 8.2,
        goal_y_max_m: float = 11.8,
        goal_depth_tol_m: float = 0.85,
        min_shooter_speed_kmh: float = 14.0,
        min_ball_speed_mps: float = 9.0,
        strike_radius_m: float = 2.0,
        max_flight_s: float = 1.5,
        cooldown_s: float = 2.0,
        db_streamer: TelemetrySQLiteStreamer | None = None,
    ) -> None:
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)
        self.penalty_zone_x_m = float(penalty_zone_x_m)
        self.goal_y_min_m = float(goal_y_min_m)
        self.goal_y_max_m = float(goal_y_max_m)
        self.goal_depth_tol_m = float(goal_depth_tol_m)
        self.min_shooter_speed_kmh = float(min_shooter_speed_kmh)
        self.min_ball_speed_mps = float(min_ball_speed_mps)
        self.strike_radius_m = float(strike_radius_m)
        self.max_flight_s = float(max_flight_s)
        self.cooldown_s = float(cooldown_s)
        self.db_streamer = db_streamer

        self._candidate: PendingShotCandidate | None = None
        self._prev_ball: tuple[float, float, float] | None = None  # (time_s, x_m, y_m)
        self._last_logged_time_s: float = -999.0

    def _is_in_goal_net(self, ball_x_m: float, ball_y_m: float) -> bool:
        in_y = self.goal_y_min_m <= ball_y_m <= self.goal_y_max_m
        in_left_goal = ball_x_m <= self.goal_depth_tol_m
        in_right_goal = ball_x_m >= (self.pitch_length_m - self.goal_depth_tol_m)
        return bool(in_y and (in_left_goal or in_right_goal))

    def _is_near_penalty_box(self, x_m: float) -> bool:
        return bool(x_m <= self.penalty_zone_x_m or x_m >= (self.pitch_length_m - self.penalty_zone_x_m))

    def update(
        self,
        frame_id: int,
        timestamp_s: float,
        rows: list[dict[str, Any]],
        ball_pitch_m: tuple[float, float] | list[float] | None,
        possession: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Process one frame and return a `"Shot Made"` event dict if confirmed."""
        ball_speed_mps = 0.0
        bx, by = (None, None)
        if ball_pitch_m is not None and len(ball_pitch_m) == 2:
            bx, by = _opt_float(ball_pitch_m[0]), _opt_float(ball_pitch_m[1])
            if bx is not None and by is not None:
                if self._prev_ball is not None:
                    dt = timestamp_s - self._prev_ball[0]
                    if 0.0 < dt <= 0.5:
                        dist = float(np.hypot(bx - self._prev_ball[1], by - self._prev_ball[2]))
                        ball_speed_mps = dist / dt
                self._prev_ball = (timestamp_s, bx, by)

        # Expire stale shot candidate
        if self._candidate is not None and (timestamp_s - self._candidate.timestamp_s) > self.max_flight_s:
            self._candidate = None

        # 1. Check if active candidate intersects the goal net
        if (
            self._candidate is not None
            and bx is not None
            and by is not None
            and (timestamp_s - self._last_logged_time_s) >= self.cooldown_s
            and self._is_in_goal_net(bx, by)
        ):
            cand = self._candidate
            event = {
                "frame_id": int(frame_id),
                "timestamp_s": float(timestamp_s),
                "event_type": "Shot Made",
                "team": cand.team,
                "tracker_id": cand.tracker_id,
                "jersey_number": cand.jersey_number,
                "player_name": cand.player_name,
                "start_x_m": round(cand.start_x_m, 2),
                "start_y_m": round(cand.start_y_m, 2),
                "goal_x_m": round(bx, 2),
                "goal_y_m": round(by, 2),
                "peak_speed_kmh": round(cand.peak_speed_kmh, 1),
                "ball_speed_mps": round(max(cand.ball_speed_mps, ball_speed_mps), 2),
            }
            self._candidate = None
            self._last_logged_time_s = timestamp_s
            if self.db_streamer is not None:
                self.db_streamer.log_event(event)
            return event

        # 2. Detect shot initiation by an active player near the penalty box
        owner_id = (possession or {}).get("owner_id")
        for r in rows:
            if not r.get("is_active", True) or not r.get("position_valid", True):
                continue
            px, py = _opt_float(r.get("pitch_x_m")), _opt_float(r.get("pitch_y_m"))
            if px is None or py is None or not self._is_near_penalty_box(px):
                continue
            tid = int(r["track_id"])
            spd = _opt_float(r.get("speed_kmh")) or 0.0
            near_ball = (
                bx is not None
                and by is not None
                and float(np.hypot(px - bx, py - by)) <= self.strike_radius_m
            )
            is_owner = owner_id is not None and int(owner_id) == tid
            if (is_owner or near_ball) and (
                spd >= self.min_shooter_speed_kmh or ball_speed_mps >= self.min_ball_speed_mps
            ):
                self._candidate = PendingShotCandidate(
                    frame_id=int(frame_id),
                    timestamp_s=float(timestamp_s),
                    tracker_id=tid,
                    team=normalize_team_key(r.get("team")),
                    jersey_number=_opt_int(r.get("jersey_number")),
                    player_name=r.get("player_name"),
                    start_x_m=px,
                    start_y_m=py,
                    peak_speed_kmh=spd,
                    ball_speed_mps=ball_speed_mps,
                )
                break

        return None
