"""Person-only ByteTrack and explicit identity boundaries at camera cuts."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from .config import TrackingConfig


class SceneCutDetector:
    """A cheap hard-cut heuristic; same-court cuts/dissolves can require manual boundaries."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.previous: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        hist /= max(float(hist.sum()), 1)
        changed = (self.previous is not None and
                   cv2.compareHist(self.previous, hist, cv2.HISTCMP_BHATTACHARYYA) > self.threshold)
        self.previous = hist
        return bool(changed)


class PersonTracker:
    """Track player/GK/referee together, preserving role metadata and observed boxes.

    IDs remain unique within a run after resets, but do not assert roster identity.
    The tracker must receive every decoded frame, including empty detections.
    """

    def __init__(self, config: TrackingConfig, fps: float) -> None:
        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.activation_threshold,
            # Supervision scales this 30-fps reference buffer by the actual FPS.
            lost_track_buffer=max(1, round(config.lost_seconds * 30)),
            minimum_matching_threshold=config.matching_threshold,
            frame_rate=max(1, round(fps)),
            minimum_consecutive_frames=config.minimum_consecutive_frames,
        )
        self.segment_id = 0
        self.offset = 0
        self.max_id = 0

    def reset_segment(self) -> None:
        self.tracker.reset()
        self.offset = self.max_id
        self.segment_id += 1

    def update(self, people: sv.Detections) -> sv.Detections:
        tracked = self.tracker.update_with_detections(people)
        if len(tracked):
            tracked.tracker_id = tracked.tracker_id + self.offset
            self.max_id = max(self.max_id, int(tracked.tracker_id.max()))
        return tracked
