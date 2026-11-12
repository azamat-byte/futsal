"""Epics 2 & 3: Rolling Substitutions Re-ID (`PlayerIdentityManager`) & In-Play Boundary Filtering (`InPlayBoundaryFilter`).

Epic 2 (`PlayerIdentityManager`):
- Stitches broken ByteTrack `tracker_id` trajectories across rolling futsal substitutions and camera re-entries.
- Combines lightweight VLM OCR (`SmolVLM` / `Qwen2-VL`) upon track entry with a SigLIP Visual Memory Bank
  (cosine similarity over L2-normalized embeddings) to map temporary `tracker_id`s back to a persistent
  `(team, jersey_number)` profile.
- Enforces same-frame team mutual exclusion so two simultaneously visible teammates cannot claim the same identity.

Epic 3 (`InPlayBoundaryFilter`):
- Projects each detection's bottom-center foot coordinate `(0.5 * (x1 + x2), y2)` through the 3x3 homography matrix
  `H_img_to_pitch` onto the 40x20m FIFA futsal court.
- Flags players stepping into the touchline substitution zones (`X in [10m, 15m]` or `[25m, 30m]`) or outside the
  playing court boundary (team benches / run-off area) with `is_active = False`.
- Integrates with `MovementEngine` and `SpatialEngine` so inactive players are excluded from Voronoi Pitch Control,
  Team Convex Hulls, and total distance / speed calculations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Mapping

import cv2
import numpy as np
import supervision as sv

from futsal.player_id_ocr import (
    DEFAULT_ROSTER_MAP,
    TemporalJerseyVoter,
    VLMJerseyOCR,
    compute_crop_quality,
    crop_shorts_region,
    crop_upper_torso,
    normalize_team_key,
    preprocess_for_ocr,
)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


class SigLIPEmbedder:
    """Extracts L2-normalized visual appearance embeddings for player Re-ID.

    Uses `google/siglip-base-patch16-224` via Hugging Face `transformers` when loaded,
    or a fast spatial-HSV + color-moment descriptor fallback for lightweight CPU/offline testing.
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        use_pretrained: bool = False,
        embed_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.use_pretrained = use_pretrained
        self._embed_fn = embed_fn
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if not self.use_pretrained or self._embed_fn is not None or self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(resolved_device)
        self._model.eval()
        self.device = resolved_device

    def embed(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        """Return a 1D L2-normalized float32 embedding vector for a player bounding-box crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if self._embed_fn is not None:
            vec = np.asarray(self._embed_fn(crop_bgr), dtype=np.float32).ravel()
            return _l2_normalize(vec)

        if self.use_pretrained:
            self._lazy_load()
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = self._processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                feats = self._model.get_image_features(**inputs)
            vec = feats[0].detach().cpu().numpy().astype(np.float32).ravel()
            return _l2_normalize(vec)

        # Lightweight 3-strip spatial HSV + LAB appearance descriptor (upper torso, shorts, socks)
        resized = cv2.resize(crop_bgr, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        strips = [
            hsv[12:56, 8:56],   # Jersey torso
            hsv[56:96, 8:56],   # Shorts
            hsv[96:124, 8:56],  # Socks / shoes
        ]
        parts: list[np.ndarray] = []
        for strip in strips:
            hist_h = cv2.calcHist([strip], [0], None, [16], [0, 180]).ravel()
            hist_s = cv2.calcHist([strip], [1], None, [8], [0, 256]).ravel()
            hist_v = cv2.calcHist([strip], [2], None, [8], [0, 256]).ravel()
            parts.extend([_l2_normalize(hist_h), _l2_normalize(hist_s), _l2_normalize(hist_v)])
        vec = np.concatenate(parts).astype(np.float32)
        return _l2_normalize(vec)


@dataclass
class PlayerProfile:
    """Persistent player identity across rolling substitutions and broken ByteTrack trajectories."""

    team: str
    jersey_number: int
    player_name: str
    active_tracker_id: int | None = None
    historical_tracker_ids: list[int] = field(default_factory=list)
    embeddings: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=16))
    last_seen_frame: int = -1
    is_active: bool = True

    @property
    def profile_id(self) -> str:
        return f"{self.team}:#{self.jersey_number}"

    def add_embedding(self, embedding: np.ndarray | None) -> None:
        if embedding is not None:
            self.embeddings.append(_l2_normalize(np.asarray(embedding, dtype=np.float32).ravel()))

    @property
    def prototype_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        mean_vec = np.mean(np.stack(list(self.embeddings), axis=0), axis=0)
        return _l2_normalize(mean_vec)


class PlayerIdentityManager:
    """Stitches broken ByteTrack IDs to persistent `(team, jersey_number)` profiles across substitutions.

    Workflow per frame:
    1. If a `tracker_id` is already bound to a locked `(team, jersey_number)` profile, reuse it and update
       its SigLIP visual memory bank on high-quality crops.
    2. For new or unresolved `tracker_id`s (e.g., player subbing in or re-entering camera view):
       a. Query VLM OCR (`SmolVLM` / `Qwen2-VL`) on 3x CLAHE-enhanced upper-torso and shorts crops.
       b. Once `TemporalJerseyVoter` reaches mode consensus, bind the `tracker_id` to that `(team, jersey_number)`
          `PlayerProfile` and stitch `historical_tracker_ids`.
       c. Prior to OCR convergence (or when the jersey number is occluded), match the track's SigLIP embedding
          against inactive/subbed-out `PlayerProfile`s on the same team via cosine similarity.
    """

    def __init__(
        self,
        ocr: VLMJerseyOCR | None = None,
        embedder: SigLIPEmbedder | None = None,
        voter: TemporalJerseyVoter | None = None,
        roster_map: Mapping[str, Mapping[int, str]] | None = None,
        reid_similarity_threshold: float = 0.78,
        min_crop_quality: float = 50.0,
        entry_ocr_frames: int = 10,
    ) -> None:
        self.ocr = ocr
        self.embedder = embedder or SigLIPEmbedder()
        self.roster_map: dict[str, dict[int, str]] = {
            normalize_team_key(team): {int(k): str(v) for k, v in players.items()}
            for team, players in (roster_map or DEFAULT_ROSTER_MAP).items()
        }
        self.voter = voter or TemporalJerseyVoter(roster_map=self.roster_map, min_votes=2, ocr_stride=3)
        self.reid_similarity_threshold = float(reid_similarity_threshold)
        self.min_crop_quality = float(min_crop_quality)
        self.entry_ocr_frames = int(entry_ocr_frames)

        # Persistent profiles keyed by (normalized_team, jersey_number)
        self.profiles: dict[tuple[str, int], PlayerProfile] = {}
        # Mapping from temporary ByteTrack tracker_id -> (normalized_team, jersey_number)
        self.tracker_to_profile: dict[int, tuple[str, int]] = {}
        # Frame index when each tracker_id was first observed
        self.tracker_first_frame: dict[int, int] = {}

    def register_known_vote(
        self,
        tracker_id: int,
        team: str,
        jersey_number: int | None,
        frame_index: int = 0,
        crop_bgr: np.ndarray | None = None,
    ) -> PlayerProfile | None:
        """Feed an external or VLM jersey number observation and stitch into a persistent `PlayerProfile`."""
        tid = int(tracker_id)
        team_key = normalize_team_key(team)
        self.tracker_first_frame.setdefault(tid, frame_index)
        mode_num = self.voter.update(tid, jersey_number, frame_index=frame_index)
        if mode_num is not None:
            return self._bind_tracker_to_profile(tid, team_key, mode_num, frame_index, crop_bgr)
        return None

    def _bind_tracker_to_profile(
        self,
        tracker_id: int,
        team_key: str,
        jersey_number: int,
        frame_index: int,
        crop_bgr: np.ndarray | None = None,
    ) -> PlayerProfile:
        key = (team_key, int(jersey_number))
        player_name = self.roster_map.get(team_key, {}).get(int(jersey_number), f"#{int(jersey_number)}")
        profile = self.profiles.get(key)
        if profile is None:
            profile = PlayerProfile(
                team=team_key,
                jersey_number=int(jersey_number),
                player_name=player_name,
            )
            self.profiles[key] = profile

        profile.active_tracker_id = int(tracker_id)
        if int(tracker_id) not in profile.historical_tracker_ids:
            profile.historical_tracker_ids.append(int(tracker_id))
        profile.last_seen_frame = int(frame_index)
        if crop_bgr is not None:
            profile.add_embedding(self.embedder.embed(crop_bgr))
        self.tracker_to_profile[int(tracker_id)] = key
        return profile

    def _match_visual_memory_bank(
        self,
        crop_bgr: np.ndarray,
        team_key: str,
        occupied_keys: set[tuple[str, int]],
    ) -> PlayerProfile | None:
        """Match a new track's SigLIP embedding against inactive same-team profiles in the memory bank."""
        query_emb = self.embedder.embed(crop_bgr)
        if query_emb is None:
            return None
        best_profile: PlayerProfile | None = None
        best_sim = -1.0
        for key, profile in self.profiles.items():
            if profile.team != team_key or key in occupied_keys:
                continue
            proto = profile.prototype_embedding
            if proto is None:
                continue
            sim = float(np.dot(query_emb, proto))
            if sim >= self.reid_similarity_threshold and sim > best_sim:
                best_sim = sim
                best_profile = profile
        return best_profile

    def update_frame(
        self,
        frame: np.ndarray,
        rows: list[dict],
        frame_index: int,
    ) -> list[dict]:
        """Resolve persistent `(team, jersey_number)` identities for all tracks in `frame_index`."""
        occupied_keys: set[tuple[str, int]] = set()
        h, w = frame.shape[:2]

        # Pass 1: Tracks already bound to a confirmed profile keep their identity
        for row in rows:
            tid = int(row["track_id"])
            if tid in self.tracker_to_profile:
                occupied_keys.add(self.tracker_to_profile[tid])

        # Pass 2: Resolve new / unbound tracks via VLM OCR + SigLIP visual memory bank
        enriched: list[dict] = []
        for row in rows:
            item = dict(row)
            tid = int(item["track_id"])
            role = str(item.get("role", "player"))
            team_key = normalize_team_key(item.get("team"))
            self.tracker_first_frame.setdefault(tid, frame_index)

            x1, y1 = max(0, int(round(float(item["x1"])))), max(0, int(round(float(item["y1"]))))
            x2, y2 = min(w, int(round(float(item["x2"])))), min(h, int(round(float(item["y2"]))))
            full_crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

            profile: PlayerProfile | None = None
            if tid in self.tracker_to_profile:
                profile = self.profiles[self.tracker_to_profile[tid]]
                profile.last_seen_frame = frame_index
                if full_crop is not None and (frame_index % 10 == 0):
                    profile.add_embedding(self.embedder.embed(full_crop))
            elif role in ("player", "goalkeeper") and team_key in ("Team A", "Team B"):
                # Aggressive VLM OCR during entry window or scheduled stride
                age = frame_index - self.tracker_first_frame[tid]
                should_ocr = self.ocr is not None and (
                    age <= self.entry_ocr_frames or self.voter.should_query_vlm(tid, frame_index)
                )
                if should_ocr and self.ocr is not None:
                    box = (float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]))
                    torso = crop_upper_torso(frame, box)
                    shorts = crop_shorts_region(frame, box)
                    # Pick higher sharpness/contrast kit region or query torso then shorts
                    pred = None
                    if torso is not None and compute_crop_quality(torso) >= self.min_crop_quality:
                        pred = self.ocr.read_number(preprocess_for_ocr(torso))
                    if pred is None and shorts is not None and compute_crop_quality(shorts) >= self.min_crop_quality:
                        pred = self.ocr.read_number(preprocess_for_ocr(shorts))
                    mode_num = self.voter.update(tid, pred, frame_index=frame_index)
                    if mode_num is not None and (team_key, mode_num) not in occupied_keys:
                        profile = self._bind_tracker_to_profile(tid, team_key, mode_num, frame_index, full_crop)
                        occupied_keys.add((team_key, mode_num))

                # Fallback to SigLIP Visual Memory Bank if OCR hasn't locked yet
                if profile is None and full_crop is not None:
                    matched = self._match_visual_memory_bank(full_crop, team_key, occupied_keys)
                    if matched is not None:
                        profile = self._bind_tracker_to_profile(
                            tid, team_key, matched.jersey_number, frame_index, full_crop
                        )
                        occupied_keys.add((team_key, matched.jersey_number))

            if profile is not None:
                profile.is_active = bool(item.get("is_active", True))
                item["raw_tracker_id"] = tid
                item["track_id"] = profile.jersey_number
                item["stitched_tracker_id"] = profile.jersey_number
                item["jersey_number"] = profile.jersey_number
                item["player_name"] = profile.player_name
                item["profile_id"] = profile.profile_id
            else:
                item["raw_tracker_id"] = tid
                item["stitched_tracker_id"] = tid
                item.setdefault("jersey_number", None)
                item.setdefault("player_name", None)
                item.setdefault("profile_id", f"{team_key}:track_{tid}")

            enriched.append(item)
        return enriched


@dataclass(frozen=True)
class SubstitutionZone:
    """Metric 2D court interval defining a touchline substitution zone."""

    x_min_m: float
    x_max_m: float
    y_edge_m: float  # 0.0 (top touchline) or 20.0 (bottom touchline)
    corridor_depth_m: float = 0.65


class InPlayBoundaryFilter:
    """Epic 3: Filters bench and substitution-zone players out of in-play tactical & physical analytics.

    Uses the 3x3 homography matrix `H_img_to_pitch` (or pre-projected `(pitch_x_m, pitch_y_m)`)
    to test whether each player's bottom-center coordinate `(0.5 * (x1 + x2), y2)` lies inside
    the active 40x20m court boundary and outside the touchline substitution zones.
    """

    def __init__(
        self,
        pitch_length_m: float = 40.0,
        pitch_width_m: float = 20.0,
        boundary_margin_m: float = 0.35,
        substitution_zones: tuple[SubstitutionZone, ...] | None = None,
        exit_confirm_frames: int = 2,
        enter_confirm_frames: int = 2,
    ) -> None:
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)
        self.boundary_margin_m = float(boundary_margin_m)
        # FIFA Futsal Law 3: Substitution zones are 5m long (10m..15m and 25m..30m from left goal line)
        self.substitution_zones = substitution_zones or (
            SubstitutionZone(x_min_m=10.0, x_max_m=15.0, y_edge_m=0.0, corridor_depth_m=0.45),
            SubstitutionZone(x_min_m=25.0, x_max_m=30.0, y_edge_m=0.0, corridor_depth_m=0.45),
        )
        self.exit_confirm_frames = max(1, int(exit_confirm_frames))
        self.enter_confirm_frames = max(1, int(enter_confirm_frames))
        self._active_state: dict[int, bool] = {}
        self._streak_count: dict[int, int] = {}

    @staticmethod
    def project_bottom_center(
        box_xyxy: np.ndarray | list[float] | tuple[float, float, float, float],
        homography_img_to_pitch: np.ndarray,
    ) -> tuple[float, float] | None:
        """Project bottom-center pixel `(0.5*(x1+x2), y2)` through 3x3 `homography_img_to_pitch`."""
        H = np.asarray(homography_img_to_pitch, dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            return None
        x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
        foot_px = np.array([[[0.5 * (x1 + x2), y2]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(foot_px, H)[0, 0]
        if not np.isfinite(projected).all():
            return None
        return float(projected[0]), float(projected[1])

    def classify_point(self, pitch_x_m: float | None, pitch_y_m: float | None) -> tuple[bool, str]:
        """Return `(raw_is_active, boundary_status)` for a 2D metric coordinate `(pitch_x_m, pitch_y_m)`."""
        if pitch_x_m is None or pitch_y_m is None or not (np.isfinite(pitch_x_m) and np.isfinite(pitch_y_m)):
            return False, "uncalibrated"

        x, y = float(pitch_x_m), float(pitch_y_m)
        eps = self.boundary_margin_m

        # 1. Outside the outer playing court + touchline margin -> bench / run-off area
        if x < -eps or x > self.pitch_length_m + eps or y < -eps or y > self.pitch_width_m + eps:
            return False, "out_of_bounds_bench"

        # 2. Check FIFA touchline substitution zones
        for zone in self.substitution_zones:
            if zone.x_min_m <= x <= zone.x_max_m:
                if abs(y - zone.y_edge_m) <= zone.corridor_depth_m and (y < 0.0 or y > self.pitch_width_m):
                    return False, "substitution_zone"
                if y < 0.15 and zone.y_edge_m == 0.0:
                    return False, "substitution_zone"
                if y > self.pitch_width_m - 0.15 and zone.y_edge_m == self.pitch_width_m:
                    return False, "substitution_zone"

        return True, "in_play"

    def update_row(
        self,
        row: dict,
        homography_img_to_pitch: np.ndarray | None = None,
    ) -> dict:
        """Evaluate a player track row and set `is_active` and `boundary_status` with hysteresis."""
        item = dict(row)
        tid = int(item["track_id"])

        px = item.get("pitch_x_m")
        py = item.get("pitch_y_m")
        if (px in ("", None) or py in ("", None)) and homography_img_to_pitch is not None:
            proj = self.project_bottom_center(
                (item["x1"], item["y1"], item["x2"], item["y2"]),
                homography_img_to_pitch,
            )
            if proj is not None:
                px, py = proj
                item["pitch_x_m"], item["pitch_y_m"] = px, py
                item["position_valid"] = True

        px_val = float(px) if px not in ("", None) and not (isinstance(px, float) and np.isnan(px)) else None
        py_val = float(py) if py not in ("", None) and not (isinstance(py, float) and np.isnan(py)) else None

        raw_active, status = self.classify_point(px_val, py_val)
        current = self._active_state.get(tid, raw_active)

        if raw_active == current:
            self._streak_count[tid] = 0
        else:
            self._streak_count[tid] = self._streak_count.get(tid, 0) + 1
            needed = self.enter_confirm_frames if raw_active else self.exit_confirm_frames
            if self._streak_count[tid] >= needed:
                current = raw_active
                self._active_state[tid] = current
                self._streak_count[tid] = 0

        self._active_state.setdefault(tid, current)
        item["is_active"] = bool(current)
        item["boundary_status"] = status if not current else "in_play"
        return item

    def filter_frame_rows(
        self,
        rows: list[dict],
        homography_img_to_pitch: np.ndarray | None = None,
    ) -> list[dict]:
        """Annotate all rows in a frame with `is_active` and `boundary_status`."""
        return [self.update_row(r, homography_img_to_pitch=homography_img_to_pitch) for r in rows]
