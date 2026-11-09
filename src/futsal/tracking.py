from types import SimpleNamespace
import numpy as np


class AdaptiveBoxSmoother:
    """Speed-adaptive exponential filtering of display boxes, never filling missing observations."""

    def __init__(self, min_cutoff=2.0, beta=.05, max_gap=.3):
        self.min_cutoff, self.beta, self.max_gap = min_cutoff, beta, max_gap
        self.states = {}

    @staticmethod
    def alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))

    def update(self, identity, box, time_s):
        box = np.asarray(box, float)
        previous = self.states.get(identity)
        if previous is None or not 0 < time_s - previous[0] <= self.max_gap:
            filtered, derivative = box.copy(), np.zeros(4)
        else:
            old_time, old_raw, old_filtered, old_derivative = previous
            dt = time_s - old_time
            a = self.alpha(1.0, dt)
            derivative = a * (box - old_raw) / dt + (1 - a) * old_derivative
            a = self.alpha(self.min_cutoff + self.beta * np.abs(derivative), dt)
            filtered = a * box + (1 - a) * old_filtered
        self.states[identity] = (time_s, box.copy(), filtered.copy(), derivative)
        self.states = {k: v for k, v in self.states.items() if time_s - v[0] <= self.max_gap}
        return filtered.tolist()


class PlayerTracker:
    def __init__(self, mode="botsort", fps=10, config=None, encoder=None):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT

        self.fps = fps
        self.last_time = None
        self.min_observations = config.track_min_observations if config else 1
        self.confirmed_ids = set()
        self.encoder = encoder
        if encoder is not None and mode != "botsort":
            raise ValueError("Appearance features require BoT-SORT")
        args = SimpleNamespace(
            track_high_thresh=config.track_high_threshold if config else .25,
            track_low_thresh=config.track_low_threshold if config else .07,
            new_track_thresh=config.track_new_threshold if config else .3,
            track_buffer=round(30 * config.track_buffer_seconds) if config else 30,
            match_thresh=config.track_match_threshold if config else .8,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.3,
            appearance_thresh=0.9,
            with_reid=encoder is not None,
            model="auto",
        )
        self.tracker = (BOTSORT if mode == "botsort" else BYTETracker)(args, frame_rate=round(fps))
        self.smoother = AdaptiveBoxSmoother() if config and config.smooth_boxes else None

    def update(self, detections, image, time_s=None):
        from ultralytics.engine.results import Boxes

        people = [d for d in detections if d["class_name"] == "person"]
        data = np.array([d["box"] + [d["score"], 0] for d in people], dtype=np.float32).reshape(-1, 6)
        if time_s is not None and self.last_time is not None:
            missed = max(0, round((time_s - self.last_time) * self.fps) - 1)
            empty = Boxes(np.empty((0, 6), np.float32), image.shape[:2])
            for _ in range(min(missed, self.tracker.max_time_lost + 1)):
                self.tracker.update(empty, None)
        self.last_time = time_s
        features = None
        if self.encoder and people:
            import torch

            crops = []
            h, w = image.shape[:2]
            for d in people:
                x1, y1, x2, y2 = np.asarray(d["box"], int)
                crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                crops.append(crop if crop.size else np.zeros((32, 16, 3), np.uint8))
            features = torch.from_numpy(np.asarray(self.encoder.encode(crops), np.float32))
        result = self.tracker.update(Boxes(data, image.shape[:2]), image, feats=features)
        for track in self.tracker.tracked_stracks:
            if track.tracklet_len + 1 >= self.min_observations:
                self.confirmed_ids.add(track.track_id)
        output = []
        for row in result:
            # output: x1,y1,x2,y2,track_id,score,class_id,detection_index
            index = int(row[7])
            if int(row[4]) not in self.confirmed_ids:
                continue
            if index >= len(people) or index < 0:
                continue
            item = dict(people[index])
            # The raw measurement remains authoritative for analytics; smoothing is display-only.
            item.update(local_id=int(row[4]), score=float(row[5]))
            if self.smoother:
                tick = time_s if time_s is not None else self.tracker.frame_id / self.fps
                item["display_box"] = self.smoother.update(item["local_id"], row[:4], tick)
            output.append(item)
        return output


class BallTracker:
    """Short-horizon image-space candidate filter; unknown instead of invented observations."""

    def __init__(self, max_gap=0.4, confirmation_frames=1, start_confidence=0):
        self.time = None
        self.position = None
        self.velocity = np.zeros(2)
        self.max_gap = max_gap
        self.confirmation_frames = confirmation_frames
        self.start_confidence = start_confidence
        self.pending_position = self.pending_time = None
        self.pending_count = 0

    def _confirm_start(self, candidate, center, time, diagonal):
        if candidate["score"] < self.start_confidence:
            return False
        contiguous = (self.pending_time is not None and 0 < time - self.pending_time <= self.max_gap
                      and np.linalg.norm(center - self.pending_position) <=
                      diagonal * (.01 + .7 * (time - self.pending_time)))
        self.pending_count = self.pending_count + 1 if contiguous else 1
        self.pending_position, self.pending_time = center.copy(), time
        return self.pending_count >= self.confirmation_frames

    def update(self, candidates, time, image_shape):
        if not candidates:
            self.pending_count = 0
            return None
        if self.time is not None and time <= self.time:
            return None
        centers = np.array(
            [[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2] for d in candidates]
        )
        elapsed = time - self.time if self.time is not None else None
        if elapsed is None or elapsed > self.max_gap:
            order = np.argsort([-d["score"] for d in candidates])
            if len(order) > 1 and candidates[order[0]]["score"] - candidates[order[1]]["score"] < 0.1:
                self.pending_count = 0
                return None
            index = order[0]
            if not self._confirm_start(candidates[index], centers[index], time, np.hypot(*image_shape[:2])):
                return None
        else:
            predicted = self.position + self.velocity * elapsed
            distances = np.linalg.norm(centers - predicted, axis=1)
            order = np.argsort(distances)
            gate = np.hypot(*image_shape[:2]) * (0.04 + 0.5 * elapsed)
            if distances[order[0]] > gate:
                # A new flight/possession can break the velocity prediction. Confirm a new observed
                # sequence rather than connecting the old and new positions with interpolation.
                order = np.argsort([-d["score"] for d in candidates])
                index = order[0]
                if len(order) > 1 and candidates[index]["score"] - candidates[order[1]]["score"] < .1:
                    return None
                if not self._confirm_start(candidates[index], centers[index], time,
                                           np.hypot(*image_shape[:2])):
                    return None
                elapsed = None
                self.velocity[:] = 0
                self.pending_count = 0
                self.position, self.time = centers[index], time
                return {**candidates[index], "continuity": "reacquired"}
            if len(order) > 1 and distances[order[1]] - distances[order[0]] < 10:
                return None
            index = order[0]
        if elapsed is not None and 0 < elapsed <= self.max_gap:
            self.velocity = 0.5 * self.velocity + 0.5 * (centers[index] - self.position) / elapsed
        else:
            self.velocity[:] = 0
        self.position, self.time = centers[index], time
        self.pending_count = 0
        return candidates[index]
