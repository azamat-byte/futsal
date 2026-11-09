"""Validated runtime settings. API credentials are never configuration fields."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class DetectorConfig:
    model_id: str = "s-workspace-bimfr/new_futsal-2-rfdetr-medium-t1"
    api_url: str = "https://serverless.roboflow.com"
    confidence: float = 0.10
    nms_iou: float = 0.70
    timeout_s: float = 45.0
    attempts: int = 3
    class_map: dict[str, str] = field(default_factory=lambda: {
        "player": "player", "goalkeeper": "goalkeeper", "ball": "ball", "referee": "referee",
    })

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.model_id):
            raise ValueError("model_id must be workspace/model-slug or project/version")
        url = urlsplit(self.api_url)
        if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
            raise ValueError("api_url must be an HTTP(S) base URL without query or fragment")
        if url.username or url.password:
            raise ValueError("Keep credentials out of api_url; use ROBOFLOW_API_KEY")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS for remote inference; HTTP is allowed for localhost only")
        if not 0 < self.confidence <= 0.10:
            raise ValueError("Use confidence <= 0.10 to preserve ByteTrack's low-score association pool")
        if not 0 < self.nms_iou <= 1 or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Invalid detector NMS or timeout")
        if not 1 <= self.attempts <= 5:
            raise ValueError("attempts must be between 1 and 5")
        if not self.class_map or not set(self.class_map.values()) <= {"player", "goalkeeper", "referee", "ball"}:
            raise ValueError("class_map values must be player, goalkeeper, referee, or ball")


@dataclass(frozen=True)
class TrackingConfig:
    activation_threshold: float = 0.30
    lost_seconds: float = 1.0
    matching_threshold: float = 0.80
    minimum_consecutive_frames: int = 2
    cut_threshold: float = 0.65

    def __post_init__(self) -> None:
        if not 0.10 < self.activation_threshold <= 0.85:
            raise ValueError("activation_threshold must be in (0.10, 0.85]")
        if not 0 < self.matching_threshold <= 1 or not 0 < self.cut_threshold <= 1:
            raise ValueError("Matching and cut thresholds must be in (0, 1]")
        if not math.isfinite(self.lost_seconds) or self.lost_seconds <= 0:
            raise ValueError("lost_seconds must be positive")
        if self.minimum_consecutive_frames < 1:
            raise ValueError("minimum_consecutive_frames must be positive")


@dataclass(frozen=True)
class TeamConfig:
    cluster_labels: tuple[str, ...] = ("team_a", "team_b")
    calibration_frames: int = 40
    calibration_stride: int = 15
    min_samples: int = 40
    max_samples: int = 1200
    calibration_confidence: float = 0.60
    min_cluster_fraction: float = 0.08
    min_center_distance: float = 0.20
    max_color_distance: float = 0.55
    min_margin: float = 0.08
    history_size: int = 15
    min_votes: int = 3
    vote_fraction: float = 0.65
    history_ttl_seconds: float = 5.0

    def __post_init__(self) -> None:
        expected = {"team_a", "team_b"} if len(self.cluster_labels) == 2 else {"team_a", "team_b", "other"}
        if len(self.cluster_labels) not in {2, 3} or set(self.cluster_labels) != expected:
            raise ValueError("cluster_labels must be a permutation of team_a/team_b[/other]")
        if min(self.calibration_frames, self.calibration_stride, self.history_size, self.min_votes) < 1:
            raise ValueError("Frame, stride, history and vote counts must be positive")
        if not 2 * len(self.cluster_labels) <= self.min_samples <= self.max_samples:
            raise ValueError("Require 2 * clusters <= min_samples <= max_samples")
        if self.min_votes > self.history_size:
            raise ValueError("min_votes cannot exceed history_size")
        for name in ("calibration_confidence", "min_cluster_fraction", "vote_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("min_center_distance", "max_color_distance", "min_margin", "history_ttl_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Phase1Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)

    @classmethod
    def load(cls, path: Path | None) -> Phase1Config:
        if path is None:
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) - {"detector", "tracking", "teams"}:
            raise ValueError("Config accepts detector, tracking, and teams objects only")
        teams = dict(raw.get("teams", {}))
        if "cluster_labels" in teams:
            teams["cluster_labels"] = tuple(teams["cluster_labels"])
        return cls(DetectorConfig(**raw.get("detector", {})),
                   TrackingConfig(**raw.get("tracking", {})), TeamConfig(**teams))

    def to_dict(self) -> dict:
        return asdict(self)
