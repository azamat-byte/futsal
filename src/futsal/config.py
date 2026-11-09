from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "best.pt"
    device: str = "cpu"
    imgsz: int = Field(1280, ge=320, le=2560)
    sample_fps: float = Field(10, gt=0, le=60)
    confidence: float = Field(0.08, ge=0, le=1)
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"
    start: float = Field(0, ge=0)
    duration: float | None = Field(60, gt=0)
    team_mode: Literal["color", "off"] = "color"
    ball_model: str | None = None
    tile_ball: bool = False
    render: bool = True
    radar: bool = True
    radar_overlay: bool = True
    show_boxes: bool = True
    show_ids: bool = True
    show_trails: bool = False
    show_court_lines: bool = False
    team_colors: dict[str, str] = Field(default_factory=lambda: {
        "team_a": "#3498ff", "team_b": "#f05252", "unknown": "#a0a0a0",
        "referee": "#ffe04b", "ball": "#ff9f35",
    })
    team_overrides_file: str | None = None
    reuse_detections_from: str | None = None
    court_length: float = Field(40, gt=0)
    court_width: float = Field(20, gt=0)
    scale_verified: bool = False
    cut_threshold: float = Field(0.55, gt=0, le=1)
    calibration_max_age: float = Field(2, gt=0, le=10)
    calibration_file: str | None = None
    cuts_file: str | None = None
    court_model: str | None = None
    court_schema: str | None = None
    inference_mode: Literal["full", "full+tiled"] = "full"
    inference_tile_size: int = Field(960, ge=320, le=1920)
    inference_tile_overlap: float = Field(.25, ge=0, lt=.8)
    enhance_contrast: bool = False
    person_confidence: float | None = Field(None, ge=0, le=1)
    ball_confidence: float | None = Field(None, ge=0, le=1)
    ball_tile_imgsz: int = Field(960, ge=320, le=1280)
    court_roi_file: str | None = None
    track_high_threshold: float = Field(.25, gt=0, le=1)
    track_low_threshold: float = Field(.07, ge=0, le=1)
    track_new_threshold: float = Field(.30, gt=0, le=1)
    track_buffer_seconds: float = Field(1.0, gt=0, le=5)
    track_min_observations: int = Field(1, ge=1, le=5)
    track_match_threshold: float = Field(.8, gt=0, le=1)
    appearance_reid: bool = False
    reid_checkpoint: str = "models/osnet-msmt17.pth"
    smooth_boxes: bool = False
    ball_confirmation_frames: int = Field(1, ge=1, le=5)
    ball_start_confidence: float = Field(0, ge=0, le=1)
    ball_max_gap: float = Field(.4, gt=0, le=1)

    @model_validator(mode="after")
    def validate_tracking(self):
        if self.track_low_threshold >= self.track_high_threshold:
            raise ValueError("Tracking low threshold must be below high threshold")
        if self.track_new_threshold < self.track_high_threshold:
            raise ValueError("New-track threshold must be >= tracking high threshold")
        if self.appearance_reid and self.tracker != "botsort":
            raise ValueError("Appearance Re-ID requires BoT-SORT")
        return self

    @classmethod
    def load(cls, path: str | Path | None):
        return cls.model_validate(yaml.safe_load(Path(path).read_text("utf-8")) or {}) if path else cls()
