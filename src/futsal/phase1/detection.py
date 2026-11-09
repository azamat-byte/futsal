"""Small Roboflow HTTP boundary; all downstream code uses Supervision detections."""

from __future__ import annotations

import base64
import logging
import time
from typing import Protocol

import cv2
import numpy as np
import requests
import supervision as sv

from .config import DetectorConfig

LOG = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Return source-pixel boxes, confidences, class_name and role data."""
        ...

    def infer(self, frame: np.ndarray) -> dict:
        """Raw response for other tasks, including the separate pitch keypoint model."""
        ...


def parse_predictions(payload: dict, shape: tuple[int, ...], config: DetectorConfig) -> sv.Detections:
    """Validate the model response, clip boxes, and resolve roles by class NAME."""
    if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
        raise ValueError("Expected a Roboflow object-detection response with a predictions list")
    rows = []
    height, width = shape[:2]
    class_map = {name.strip().lower(): role for name, role in config.class_map.items()}
    for prediction in payload["predictions"]:
        try:
            name = str(prediction["class"]).strip().lower()
            role = class_map.get(name)
            if role is None:
                raise ValueError(f"Unmapped detection class {name!r}; update detector.class_map")
            x, y, w, h, score = (float(prediction[k]) for k in ("x", "y", "width", "height", "confidence"))
            if not np.isfinite([x, y, w, h, score]).all() or not 0 <= score <= 1 or w <= 0 or h <= 0:
                continue
            if score < config.confidence:
                continue
            box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if np.any(box[2:] - box[:2] < 1):
                continue
            # Class IDs are retained as metadata, never used to guess team/role.
            rows.append((box, score, int(prediction["class_id"]), name, role))
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed Roboflow detection prediction") from error
    if not rows:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([r[0] for r in rows], dtype=np.float32),
        confidence=np.array([r[1] for r in rows], dtype=np.float32),
        class_id=np.array([r[2] for r in rows], dtype=int),
        data={"class_name": np.array([r[3] for r in rows]), "role": np.array([r[4] for r in rows])},
    )


def person_detections(detections: sv.Detections, nms_iou: float) -> sv.Detections:
    """Deduplicate across person roles without allowing a ball to suppress a person."""
    if not len(detections):
        return sv.Detections.empty()
    people = detections[np.isin(detections.data["role"], ["player", "goalkeeper", "referee"])]
    return people.with_nms(threshold=nms_iou, class_agnostic=True)


class RoboflowDetector:
    """Serverless /model-id API, also supported by current local Inference servers.

    Sends source-resolution JPEGs with a bearer header (server release >=1.5.0).
    Retry only transient transport/429/5xx failures. Credentials, response bodies,
    and request objects are never included in logs or raised exception messages.
    """

    def __init__(self, config: DetectorConfig, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("Set ROBOFLOW_API_KEY before running hosted inference")
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/x-www-form-urlencoded",
        })

    def detect(self, frame: np.ndarray) -> sv.Detections:
        return parse_predictions(self.infer(frame), frame.shape, self.config)

    def infer(self, frame: np.ndarray) -> dict:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("Could not encode video frame for inference")
        body = base64.b64encode(encoded.tobytes()).decode("ascii")
        endpoint = f"{self.config.api_url.rstrip('/')}/{self.config.model_id}"
        for attempt in range(self.config.attempts):
            try:
                response = self.session.post(
                    endpoint, data=body,
                    params={"confidence": round(100 * self.config.confidence, 4),
                            "overlap": round(100 * self.config.nms_iou, 4), "format": "json"},
                    timeout=(10, self.config.timeout_s), allow_redirects=False,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt + 1 == self.config.attempts:
                    raise RuntimeError("Inference connection failed after bounded retries") from None
            else:
                with response:
                    if response.status_code == 200:
                        try:
                            payload = response.json()
                        except ValueError:
                            raise RuntimeError("Inference returned invalid JSON") from None
                        return payload
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        raise RuntimeError(
                            f"Inference HTTP {response.status_code}; check model ID, key and endpoint access"
                        )
                    if attempt + 1 == self.config.attempts:
                        raise RuntimeError(f"Inference HTTP {response.status_code} after bounded retries")
            LOG.warning("Transient inference failure; retry %d/%d", attempt + 2, self.config.attempts)
            time.sleep(min(2 ** attempt, 4))
        raise RuntimeError("Inference exhausted retries")

    def close(self) -> None:
        self.session.close()
