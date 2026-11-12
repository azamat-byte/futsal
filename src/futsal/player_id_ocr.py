"""Phase 6: Player Identification via Lightweight VLM Jersey Number OCR.

Adapts Roboflow's Basketball AI VLM OCR + temporal mode voting pipeline for futsal:
1. Upper-torso bounding-box cropping from tracked player detections.
2. Lightweight VLM OCR (`SmolVLM` / `Qwen2-VL` via Hugging Face `transformers`)
   using the strict prompt: "Read the jersey number on this player. Return only the digits."
3. Per-track rolling buffer + mode voting (`TemporalJerseyVoter`) to filter motion blur.
4. Team + Jersey Number -> Player Name roster lookup (`ROSTER_MAP`).
5. `sv.EllipseAnnotator` + `sv.LabelAnnotator` visualization displaying resolved player names.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import re
from typing import Callable, Mapping

import cv2
import numpy as np
import supervision as sv

from futsal.phase1.annotations import COLORS, GROUPS

TEAM_NAMES: dict[str, str] = {
    "team_a": "Team A",
    "team_b": "Team B",
    "goalkeeper": "GK",
    "referee": "Ref",
    "other": "Other",
    "unknown": "Unknown",
}

STRICT_VLM_PROMPT = "Read the jersey number on this player. Return only the digits."

DEFAULT_ROSTER_MAP: dict[str, dict[int, str]] = {
    "Team A": {
        1: "Guitta",
        2: "Roncaglio",
        3: "Willian",
        4: "Marlon",
        5: "Neguinho",
        6: "Marcel",
        7: "Dyego",
        8: "Marcenio",
        9: "Leandro Lino",
        10: "Pito",
        11: "Ferrao",
        12: "Arthur",
        13: "Rafa Santos",
        14: "Felipe Valerio",
    },
    "Team B": {
        1: "Sarmiento",
        2: "Tripodi",
        3: "Claudino",
        4: "Bolo Alemany",
        5: "Arrieta",
        6: "Corso",
        7: "Menzeguez",
        8: "Gauna",
        9: "Borruto",
        10: "Vaporaki",
        11: "Brandi",
        12: "Kiki Vaporaki",
        13: "Plaza",
        14: "Taborda",
        18: "Matias Rosa",
    },
}

_DIGIT_PATTERN = re.compile(r"\b(\d{1,2})\b")


def normalize_team_key(team: str | None) -> str:
    """Normalize internal ('team_a', 'team_b') or display ('Team A', 'Team B') keys."""
    if not team:
        return "Unknown"
    low = str(team).strip().lower()
    if low in ("team_a", "team a", "a", "0"):
        return "Team A"
    if low in ("team_b", "team b", "b", "1"):
        return "Team B"
    return str(team).strip()


def parse_jersey_digits(raw_text: str | None) -> int | None:
    """Extract a valid futsal jersey number (1..99) from raw VLM output text."""
    if not raw_text:
        return None
    cleaned = str(raw_text).strip()
    # Strip echoed prompt if model repeats the instruction
    if STRICT_VLM_PROMPT in cleaned:
        cleaned = cleaned.split(STRICT_VLM_PROMPT)[-1].strip()
    if "Assistant:" in cleaned:
        cleaned = cleaned.split("Assistant:")[-1].strip()
    match = _DIGIT_PATTERN.search(cleaned)
    if not match:
        return None
    number = int(match.group(1))
    return number if 1 <= number <= 99 else None


def crop_upper_torso(
    frame: np.ndarray,
    box_xyxy: np.ndarray | list[float] | tuple[float, float, float, float],
    top_ratio: float = 0.12,
    bottom_ratio: float = 0.58,
    side_margin_ratio: float = 0.10,
    min_width: int = 16,
    min_height: int = 24,
) -> np.ndarray | None:
    """Crop the upper-torso region of a tracked player where the jersey number sits."""
    box = np.asarray(box_xyxy, dtype=float).ravel()
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < min_width or bh < min_height:
        return None

    h, w = frame.shape[:2]
    cx1 = int(np.clip(round(x1 + side_margin_ratio * bw), 0, w))
    cx2 = int(np.clip(round(x2 - side_margin_ratio * bw), 0, w))
    cy1 = int(np.clip(round(y1 + top_ratio * bh), 0, h))
    cy2 = int(np.clip(round(y1 + bottom_ratio * bh), 0, h))

    if (cx2 - cx1) < 8 or (cy2 - cy1) < 12:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


def crop_shorts_region(
    frame: np.ndarray,
    box_xyxy: np.ndarray | list[float] | tuple[float, float, float, float],
    top_ratio: float = 0.52,
    bottom_ratio: float = 0.78,
    side_margin_ratio: float = 0.08,
) -> np.ndarray | None:
    """Crop the shorts/thigh region where futsal front-facing kit numbers are printed."""
    return crop_upper_torso(
        frame,
        box_xyxy,
        top_ratio=top_ratio,
        bottom_ratio=bottom_ratio,
        side_margin_ratio=side_margin_ratio,
        min_width=16,
        min_height=20,
    )


def compute_crop_quality(crop_bgr: np.ndarray | None) -> float:
    """Score crop sharpness (Laplacian variance) and contrast for quality-gated VLM voting."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    area = float(gray.shape[0] * gray.shape[1])
    return sharpness * contrast * np.sqrt(area)


def preprocess_for_ocr(crop_bgr: np.ndarray, scale: int = 3) -> np.ndarray:
    """Upscale crop with bicubic interpolation and apply LAB CLAHE local contrast enhancement."""
    h, w = crop_bgr.shape[:2]
    upscaled = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_enhanced = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_enhanced, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


class VLMJerseyOCR:
    """Lightweight Vision-Language Model (SmolVLM / Qwen2-VL) wrapper for jersey number OCR."""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        device: str | None = None,
        max_new_tokens: int = 8,
        predict_fn: Callable[[np.ndarray, str], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt = STRICT_VLM_PROMPT
        self._predict_fn = predict_fn
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._predict_fn is not None or self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if "cuda" in str(resolved_device) else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(resolved_device)
        self._model.eval()
        self.device = resolved_device

    def read_number(self, torso_crop_bgr: np.ndarray | None) -> int | None:
        """Run VLM OCR on a BGR upper-torso crop and return the parsed jersey number (1..99)."""
        if torso_crop_bgr is None or torso_crop_bgr.size == 0:
            return None

        if self._predict_fn is not None:
            raw_output = self._predict_fn(torso_crop_bgr, self.prompt)
            return parse_jersey_digits(raw_output)

        self._lazy_load()
        from PIL import Image
        import torch

        rgb = cv2.cvtColor(torso_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt_text, images=[pil_image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        trimmed_ids = generated_ids[:, input_len:]
        decoded = self._processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        return parse_jersey_digits(decoded)


@dataclass
class TemporalJerseyVoter:
    """Rolling buffer + mode voting for associating stable jersey numbers & player names to tracks."""

    roster_map: Mapping[str, Mapping[int, str]] = field(default_factory=lambda: dict(DEFAULT_ROSTER_MAP))
    buffer_size: int = 25
    min_votes: int = 3
    min_agreement: float = 0.40
    ocr_stride: int = 5
    histories: dict[int, deque[int]] = field(default_factory=dict)
    last_ocr_frame: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.histories.clear()
        self.last_ocr_frame.clear()

    def should_query_vlm(self, tracker_id: int, frame_index: int) -> bool:
        """Throttle VLM calls per track while allowing immediate queries on new tracks."""
        last_frame = self.last_ocr_frame.get(int(tracker_id))
        return last_frame is None or (frame_index - last_frame) >= self.ocr_stride

    def update(self, tracker_id: int, number_prediction: int | None, frame_index: int | None = None) -> int | None:
        """Record a VLM jersey number prediction for `tracker_id` and return its current mode."""
        tid = int(tracker_id)
        if frame_index is not None:
            self.last_ocr_frame[tid] = int(frame_index)
        if number_prediction is not None and 1 <= int(number_prediction) <= 99:
            buf = self.histories.setdefault(tid, deque(maxlen=self.buffer_size))
            buf.append(int(number_prediction))
        return self.get_jersey_number(tid)

    def get_jersey_number(self, tracker_id: int) -> int | None:
        """Return the most frequent valid jersey number (mode) if it meets `min_votes`."""
        buf = self.histories.get(int(tracker_id))
        if not buf or len(buf) < self.min_votes:
            return None
        winner, count = Counter(buf).most_common(1)[0]
        if count / len(buf) < self.min_agreement:
            return None
        return int(winner)

    def resolve_player_name(self, tracker_id: int, team: str | None) -> str | None:
        """Map `(team, mode_jersey_number)` to a player name from `roster_map`."""
        jersey = self.get_jersey_number(tracker_id)
        if jersey is None:
            return None
        team_key = normalize_team_key(team)
        team_roster = self.roster_map.get(team_key, {})
        return team_roster.get(jersey, f"#{jersey}")

    def format_label(
        self,
        tracker_id: int,
        team: str | None,
        role: str = "player",
        speed_kmh: float | None = None,
    ) -> str:
        """Build the display label for `sv.EllipseAnnotator` / `sv.LabelAnnotator`."""
        tid = int(tracker_id)
        jersey = self.get_jersey_number(tid)
        name = self.resolve_player_name(tid, team)
        team_display = TEAM_NAMES.get(str(team), normalize_team_key(team))

        if role == "referee":
            base = f"#{tid} Referee"
        elif name is not None and jersey is not None:
            if name.startswith("#"):
                base = f"{name} {team_display}"
            else:
                base = f"#{jersey} {name}"
        elif role == "goalkeeper":
            base = f"#{tid} GK {team_display}" if team in ("team_a", "team_b", "Team A", "Team B") else f"#{tid} GK"
        else:
            base = f"#{tid} {team_display}"

        if speed_kmh is not None and np.isfinite(speed_kmh):
            base += f" | {float(speed_kmh):.1f} km/h"
        return base


class PlayerIdentityEllipseAnnotator:
    """Supervision `sv.EllipseAnnotator` + `sv.LabelAnnotator` displaying VLM-resolved Player Names."""

    def __init__(
        self,
        ocr: VLMJerseyOCR,
        voter: TemporalJerseyVoter | None = None,
    ) -> None:
        self.ocr = ocr
        self.voter = voter or TemporalJerseyVoter()
        palette = sv.ColorPalette.from_hex(list(COLORS))
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=palette,
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.BLACK,
            text_scale=0.5,
            text_padding=4,
            text_position=sv.Position.BOTTOM_CENTER,
        )

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        teams: list[str],
        frame_index: int = 0,
        speeds_kmh: list[float | None] | None = None,
    ) -> np.ndarray:
        """Crop upper torsos on scheduled frames, update mode votes, and draw ground ellipses with player names."""
        scene = frame.copy()
        if not len(detections):
            return scene

        roles = detections.data.get("role", np.repeat("player", len(detections)))
        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else np.arange(1, len(detections) + 1, dtype=int)
        )

        groups: list[int] = []
        labels: list[str] = []

        for i in range(len(detections)):
            tid = int(tracker_ids[i])
            role = str(roles[i])
            team = teams[i] if i < len(teams) else "unknown"

            # Run VLM OCR on upper-torso crop for players/goalkeepers at the configured stride
            if role in ("player", "goalkeeper") and self.voter.should_query_vlm(tid, frame_index):
                torso = crop_upper_torso(frame, detections.xyxy[i])
                pred_num = self.ocr.read_number(torso)
                self.voter.update(tid, pred_num, frame_index=frame_index)

            if role == "referee":
                group = GROUPS.index("referee")
            elif role == "goalkeeper":
                group = GROUPS.index(team if team in ("team_a", "team_b") else "goalkeeper")
            else:
                group = GROUPS.index(team if team in GROUPS else "unknown")
            groups.append(group)

            spd = speeds_kmh[i] if (speeds_kmh is not None and i < len(speeds_kmh)) else None
            labels.append(self.voter.format_label(tid, team, role=role, speed_kmh=spd))

        color_indices = np.asarray(groups, dtype=int)
        scene = self.ellipse_annotator.annotate(scene, detections, custom_color_lookup=color_indices)
        scene = self.label_annotator.annotate(scene, detections, labels=labels, custom_color_lookup=color_indices)
        return scene
