"""Epic 1: Broadcast Scene Management (`BroadcastSceneManager`).

Integrates `scenedetect` (`ContentDetector` when installed, with a built-in HSV-luma content delta detector)
and classifies frames as `"Tactical Wide Angle"` vs. `"Replay/Close-up"`.
Safely pauses tracking, homography optical flow, and movement integrators during non-tactical cuts,
and cleanly re-initializes engines (`reset()`) when the tactical wide angle returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import cv2
import numpy as np


class ResettableEngine(Protocol):
    def reset(self) -> Any:
        ...


@dataclass(frozen=True)
class SceneDecision:
    """Per-frame classification output from `BroadcastSceneManager`."""

    frame_index: int
    scene_label: str          # "Tactical Wide Angle" | "Replay/Close-up"
    is_tactical: bool
    is_hard_cut: bool
    reinitialized: bool
    segment_id: int
    court_ratio: float
    median_box_height_ratio: float
    content_delta: float


class BroadcastSceneManager:
    """Detects broadcast camera cuts and classifies Tactical Wide Angle vs. Replay/Close-up scenes.

    Heuristics:
    1. Hard Cut Detection: Uses `scenedetect.detectors.ContentDetector` (if installed) or HSV-Luma
       mean absolute frame delta (`content_delta >= cut_threshold`) to detect camera switches.
    2. Court Dominance (`court_ratio >= min_court_ratio`): Measures the fraction of pixels in the
       lower 75% of the frame matching the futsal pitch color (blue FIFA court or green/wood floor).
    3. Player Scale (`median_box_height_ratio <= max_box_height_ratio`): In a wide tactical angle,
       players span ~8%–22% of frame height; in close-ups/replays, players span >26% of frame height.
    """

    TACTICAL_LABEL = "Tactical Wide Angle"
    NON_TACTICAL_LABEL = "Replay/Close-up"

    def __init__(
        self,
        min_court_ratio: float = 0.28,
        max_box_height_ratio: float = 0.26,
        cut_threshold: float = 27.0,
        confirm_tactical_frames: int = 2,
        engines_to_reset: list[ResettableEngine | Callable[[], Any]] | None = None,
    ) -> None:
        self.min_court_ratio = float(min_court_ratio)
        self.max_box_height_ratio = float(max_box_height_ratio)
        self.cut_threshold = float(cut_threshold)
        self.confirm_tactical_frames = max(1, int(confirm_tactical_frames))
        self.engines_to_reset = list(engines_to_reset or [])

        self.segment_id: int = 0
        self.in_tactical: bool = True
        self._tactical_streak: int = 0
        self._prev_hsv_small: np.ndarray | None = None
        self._scenedetect_detector = self._init_scenedetect()

    def _init_scenedetect(self) -> Any:
        try:
            from scenedetect.detectors import ContentDetector

            return ContentDetector(threshold=self.cut_threshold)
        except Exception:
            return None

    def register_engine(self, engine: ResettableEngine | Callable[[], Any]) -> None:
        """Register a stateful engine (tracker, homography, movement) to reset upon tactical re-entry."""
        self.engines_to_reset.append(engine)

    def _trigger_resets(self) -> None:
        for eng in self.engines_to_reset:
            if hasattr(eng, "reset") and callable(eng.reset):
                eng.reset()
            elif callable(eng):
                eng()

    @staticmethod
    def compute_court_ratio(frame_bgr: np.ndarray) -> float:
        """Compute fraction of pixels in the lower 75% of the frame matching futsal court colors."""
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0
        h, w = frame_bgr.shape[:2]
        roi = frame_bgr[int(0.25 * h) :, :]
        small = cv2.resize(roi, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # Blue futsal court (H: 90..130) OR green/parquet futsal court (H: 15..85)
        blue_mask = cv2.inRange(hsv, (90, 45, 50), (132, 255, 255))
        green_wood_mask = cv2.inRange(hsv, (15, 35, 50), (85, 255, 255))
        combined = cv2.bitwise_or(blue_mask, green_wood_mask)
        return float(np.count_nonzero(combined)) / float(combined.size)

    @staticmethod
    def compute_median_box_height_ratio(
        boxes_xyxy: np.ndarray | list[list[float]] | None,
        frame_height: int,
    ) -> float:
        """Compute median bounding-box height as a fraction of `frame_height`."""
        if boxes_xyxy is None or frame_height <= 0:
            return 0.0
        arr = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(arr) == 0:
            return 0.0
        heights = np.clip(arr[:, 3] - arr[:, 1], 0.0, float(frame_height))
        return float(np.median(heights) / float(frame_height))

    def _compute_content_delta(self, frame_bgr: np.ndarray, frame_index: int) -> tuple[float, bool]:
        small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        delta = 0.0
        if self._prev_hsv_small is not None:
            delta = float(np.mean(np.abs(hsv - self._prev_hsv_small)))
        self._prev_hsv_small = hsv

        is_cut = delta >= self.cut_threshold
        if self._scenedetect_detector is not None:
            try:
                cuts = self._scenedetect_detector.process_frame(frame_index, small)
                if cuts:
                    is_cut = True
            except Exception:
                pass
        return delta, is_cut

    def update(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        boxes_xyxy: np.ndarray | list[list[float]] | None = None,
    ) -> SceneDecision:
        """Classify `frame_bgr` and pause or re-initialize registered engines on scene transitions."""
        h = frame_bgr.shape[0] if frame_bgr is not None and frame_bgr.size else 1080
        delta, is_cut = self._compute_content_delta(frame_bgr, frame_index)
        court_ratio = self.compute_court_ratio(frame_bgr)
        box_h_ratio = self.compute_median_box_height_ratio(boxes_xyxy, h)

        raw_tactical = (court_ratio >= self.min_court_ratio) and (
            box_h_ratio == 0.0 or box_h_ratio <= self.max_box_height_ratio
        )

        reinitialized = False
        if not raw_tactical:
            # Immediately pause tracking & homography when cutting to Replay/Close-up
            self.in_tactical = False
            self._tactical_streak = 0
        else:
            self._tactical_streak += 1
            if not self.in_tactical and self._tactical_streak >= self.confirm_tactical_frames:
                # Tactical wide angle has returned after a replay/close-up: increment segment & reset engines
                self.in_tactical = True
                self.segment_id += 1
                self._trigger_resets()
                reinitialized = True
            elif self.in_tactical and is_cut:
                # Hard camera cut between two wide angles: increment segment & reset engines
                self.segment_id += 1
                self._trigger_resets()
                reinitialized = True

        label = self.TACTICAL_LABEL if self.in_tactical else self.NON_TACTICAL_LABEL
        return SceneDecision(
            frame_index=int(frame_index),
            scene_label=label,
            is_tactical=bool(self.in_tactical),
            is_hard_cut=bool(is_cut),
            reinitialized=bool(reinitialized),
            segment_id=int(self.segment_id),
            court_ratio=round(court_ratio, 4),
            median_box_height_ratio=round(box_h_ratio, 4),
            content_delta=round(delta, 3),
        )
