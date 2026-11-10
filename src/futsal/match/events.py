"""Pose proximity and temporal pass/shot hypotheses, never official event labels."""

import numpy as np


class BallSelector:
    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def select(self, detections, time_s, shape):
        if not len(detections):
            return None
        indices = np.flatnonzero((detections.data['role'] == 'ball') & (detections.confidence >= .25))
        candidates = []
        for i in indices:
            box = detections.xyxy[i]
            center = (box[:2]+box[2:])/2
            if self.previous is not None and time_s-self.previous[0] <= .2:
                gate = np.hypot(*shape[:2])*(.025+1.2*(time_s-self.previous[0]))
                if np.linalg.norm(center-self.previous[1]) > gate:
                    continue
            candidates.append((float(detections.confidence[i]), center, box))
        candidates.sort(key=lambda c: -c[0])
        if not candidates or (len(candidates) > 1 and candidates[0][0]-candidates[1][0] < .1):
            return None
        score, center, box = candidates[0]
        continuous = self.previous is not None and 0 < time_s-self.previous[0] <= .2
        self.previous = time_s, center
        return dict(x=float(center[0]), y=float(center[1]), confidence=score,
                    box=box.tolist(), continuous=continuous)


class PossessionEngine:
    def __init__(self, confirm_s=.12):
        self.confirm_s = confirm_s
        self.pending = self.pending_since = self.last_time = None

    def reset(self):
        self.pending = self.pending_since = self.last_time = None

    def update(self, time_s, ball, rows, feet):
        if self.last_time is not None and not 0 < time_s-self.last_time <= .25:
            self.reset()
        self.last_time = time_s
        candidate = None
        state = 'unknown'
        if ball is not None:
            center = np.array([ball['x'], ball['y']])
            distances = []
            for row in rows:
                if row['role'] not in {'player', 'goalkeeper'} or row['track_id'] not in feet:
                    continue
                height = max(1., row['y2']-row['y1'])
                distance = float(np.linalg.norm(np.asarray(feet[row['track_id']])-center, axis=1).min()/height)
                distances.append((distance, row['track_id'], row['team']))
            distances.sort()
            if distances:
                state = 'free_ball'
                if distances[0][0] <= .65:
                    if len(distances) > 1 and distances[1][0]-distances[0][0] < .10:
                        state = 'contested'
                    else:
                        candidate = distances[0][1:]
            # Without confident foot keypoints, no box-bottom surrogate is invented.
        if candidate != self.pending:
            self.pending, self.pending_since = candidate, time_s
        if candidate:
            state = 'controlled' if time_s-self.pending_since >= self.confirm_s else 'pending'
        return dict(state=state, owner_id=candidate[0] if state == 'controlled' else None,
                    team=candidate[1] if state == 'controlled' else None,
                    method='ankle_proximity_temporal', status='hypothesis_requires_review')


class EventEngine:
    def __init__(self, length=40., width=20.):
        self.length, self.width = length, width
        self.events = []
        self.reset()

    def reset(self):
        self.last_ball = self.velocity = self.last_owner = self.flight = None

    def update(self, time_s, segment_id, possession, ball_pitch, continuous, anchor):
        emitted = []
        if ball_pitch is None or not continuous:
            self.reset()
        velocity = None
        if ball_pitch is not None:
            point = np.asarray(ball_pitch, float)
            if self.last_ball is not None:
                old_time, old_point, old_anchor = self.last_ball
                dt = time_s-old_time
                if 0 < dt <= .2 and anchor == old_anchor:
                    velocity = (point-old_point)/dt
                    if np.linalg.norm(velocity) > 45:
                        self.reset()
                        velocity = None
                else:
                    self.reset()
            if velocity is not None and self.last_owner is not None and self.velocity is not None:
                speed = float(np.linalg.norm(velocity))
                change = float(np.linalg.norm(velocity-self.velocity))
                if (self.flight is None and time_s-self.last_owner['time'] <= .4 and speed >= 2.5 and change >= 1.5):
                    self.flight = {**self.last_owner, 'launch_time': time_s, 'origin': point.copy(), 'shot': False}
            if self.flight is not None:
                if time_s-self.flight['launch_time'] > 3:
                    self.flight = None
                elif velocity is not None and not self.flight['shot'] and abs(velocity[0]) > 2.5:
                    goal_x = self.length if velocity[0] > 0 else 0
                    eta = (goal_x-point[0])/velocity[0]
                    goal_y = point[1]+eta*velocity[1]
                    if 0 < eta < .8 and abs(goal_x-point[0]) < 15 and abs(goal_y-self.width/2) <= 1.5:
                        emitted.append(dict(type='shot_candidate', time_s=time_s, segment_id=segment_id,
                                            source_id=self.flight['owner_id'], receiver_id=None,
                                            team=self.flight['team'], evidence='post_control_acceleration_toward_goal'))
                        self.flight['shot'] = True
            self.last_ball = time_s, point, anchor
            self.velocity = velocity
        if possession['state'] == 'controlled':
            if (self.flight is not None and possession['owner_id'] != self.flight['owner_id'] and
                    possession['team'] in {'team_a', 'team_b'} and possession['team'] == self.flight['team'] and
                    ball_pitch is not None and np.linalg.norm(np.asarray(ball_pitch)-self.flight['origin']) >= 1):
                emitted.append(dict(type='pass_candidate', time_s=time_s, segment_id=segment_id,
                                    source_id=self.flight['owner_id'], receiver_id=possession['owner_id'],
                                    team=possession['team'], evidence='release_acceleration_then_same_team_control'))
                self.flight = None
            self.last_owner = dict(time=time_s, owner_id=possession['owner_id'], team=possession['team'])
        elif possession['state'] in {'unknown', 'contested'}:
            self.last_owner = self.flight = None
        for item in emitted:
            item.update(event_id=len(self.events)+1, status='candidate_requires_review',
                        trajectory_basis='ball_center_projected_to_floor_assumption')
            self.events.append(item)
        return emitted
