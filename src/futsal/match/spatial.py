"""Streaming movement estimates and pitch-clipped, visible-player spatial geometry."""

from collections import deque

import cv2
import numpy as np


class MovementEngine:
    def __init__(self, window=5, max_gap_s=.25, max_speed_mps=12):
        self.window, self.max_gap, self.max_speed = window, max_gap_s, max_speed_mps
        self.states, self.totals = {}, {}

    def reset(self):
        self.states.clear()

    def update(self, row, time_s, anchor):
        identity = (row['segment_id'], row['track_id'])
        total = self.totals.setdefault(identity, dict(segment_id=identity[0], track_id=identity[1],
                                                    team=row['team'], observed_distance_m=0., valid_duration_s=0.,
                                                    max_speed_kmh=None))
        if row['team'] in {'team_a', 'team_b'}:
            total['team'] = row['team']
        empty = dict(speed_kmh=None, distance_step_m=None, observed_distance_m=total['observed_distance_m'])
        if not row.get('position_valid'):
            self.states.pop(identity, None)
            return empty
        point = np.array([row['pitch_x_m'], row['pitch_y_m']], float)
        state = self.states.get(identity)
        if state is not None:
            dt = time_s-state['time']
            if (anchor != state['anchor'] or not 0 < dt <= self.max_gap or
                    np.linalg.norm(point-state['raw'])/dt > self.max_speed):
                state = None
        if state is None:
            self.states[identity] = dict(time=time_s, anchor=anchor, raw=point,
                                         samples=deque([(time_s, point)], maxlen=self.window), filtered=None)
            return empty
        state['samples'].append((time_s, point))
        # Position and timestamp use the same rolling mean. Startup windows therefore
        # do not divide a half-frame movement by a full-frame interval.
        smoothed = np.mean([p for _, p in state['samples']], axis=0)
        smooth_time = float(np.mean([t for t, _ in state['samples']]))
        previous = state['filtered'] or (state['time'], state['raw'])
        dt = smooth_time-previous[0]
        step = float(np.linalg.norm(smoothed-previous[1]))
        state.update(time=time_s, raw=point, filtered=(smooth_time, smoothed))
        if dt <= 0 or step/dt > self.max_speed:
            return empty
        speed = step/dt*3.6
        total['observed_distance_m'] += step
        total['valid_duration_s'] += dt
        total['max_speed_kmh'] = max(total['max_speed_kmh'] or 0, speed)
        return dict(speed_kmh=speed, distance_step_m=step, observed_distance_m=total['observed_distance_m'])

    def summary(self):
        return [{**row, 'observed_distance_m': row['observed_distance_m'] if row['valid_duration_s'] else None,
                 'mean_speed_kmh': row['observed_distance_m']/row['valid_duration_s']*3.6
                 if row['valid_duration_s'] else None} for row in self.totals.values()]


def polygon_area(points):
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 3:
        return 0.
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], 1))-np.dot(p[:, 1], np.roll(p[:, 0], 1)))/2)


def clip_halfplane(polygon, normal, limit):
    result = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        a, b = float(first @ normal-limit), float(second @ normal-limit)
        if a <= 1e-9:
            result.append(first)
        if (a <= 0) != (b <= 0):
            result.append(first + (second-first)*a/(a-b))
    return np.asarray(result, float).reshape(-1, 2)


class SpatialEngine:
    def __init__(self, length=40., width=20., cell=.5):
        self.length, self.width, self.cell = length, width, cell
        self.heatmaps, self.last_seen = {}, {}

    def update(self, rows, time_s):
        eligible = [
            r for r in rows
            if r.get("position_valid")
            and r["role"] in {"player", "goalkeeper"}
            and r["team"] in {"team_a", "team_b"}
        ]
        for row in rows:
            key = f"{row['segment_id']}:{row['track_id']}"
            previous = self.last_seen.pop(key, None)
            if not row.get('position_valid'):
                continue
            grid = self.heatmaps.setdefault(key, np.zeros((int(np.ceil(self.width/self.cell)),
                                                          int(np.ceil(self.length/self.cell))), np.float32))
            self.last_seen[key] = time_s
            if previous is not None and 0 < time_s-previous <= .25:
                x, y = row['pitch_x_m'], row['pitch_y_m']
                grid[min(grid.shape[0]-1, int(y/self.cell)), min(grid.shape[1]-1, int(x/self.cell))] += time_s-previous
        hulls = []
        for team in ('team_a', 'team_b'):
            group = [r for r in eligible if r['team'] == team]
            if not group:
                continue
            points = np.array([[r['pitch_x_m'], r['pitch_y_m']] for r in group], np.float32)
            hull = cv2.convexHull(points).reshape(-1, 2)
            hulls.append(dict(team=team, visible_players=len(group), area_m2=polygon_area(hull),
                              width_m=float(np.ptp(points[:, 1])), depth_m=float(np.ptp(points[:, 0])),
                              polygon=hull.tolist(), scope='visible_outfield_subset'))
        cells = []
        points = np.array([[r['pitch_x_m'], r['pitch_y_m']] for r in eligible]).reshape(-1, 2)
        valid = {r['team'] for r in eligible} == {'team_a', 'team_b'}
        if len(points) > 1:
            distances = np.linalg.norm(points[:, None]-points[None], axis=2)
            np.fill_diagonal(distances, np.inf)
            valid = valid and distances.min() > .01
        if valid:
            for i, (row, p) in enumerate(zip(eligible, points)):
                polygon = np.array([[0, 0], [self.length, 0], [self.length, self.width], [0, self.width]], float)
                for j, q in enumerate(points):
                    if i != j and len(polygon):
                        polygon = clip_halfplane(polygon, 2*(q-p), float(q@q-p@p))
                cells.append(dict(track_id=row['track_id'], team=row['team'], polygon=polygon.tolist(),
                                  area_m2=polygon_area(polygon)))
        return dict(hulls=hulls, cells=cells, control_valid=bool(valid),
                    team_a_control_pct=sum(c['area_m2'] for c in cells if c['team'] == 'team_a')/
                    (self.length*self.width)*100 if valid else None,
                    scope='nearest_visible_outfield_player_proxy')
