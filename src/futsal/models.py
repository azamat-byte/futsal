"""Detector adapters and explicit model provenance. No implicit accuracy claims."""

from pathlib import Path
import urllib.request

import cv2
import numpy as np

from .storage import sha256, write_json

SPORTS_COMMIT = "42c80c06b6b65a7f89455b89fe31cdf4c38ba227"
MODEL_SOURCES = {
    "yolo11n": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", "yolo11n.pt"),
    "football-player": ("17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q", "football-player-detection.pt"),
    "football-ball": ("1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V", "football-ball-detection.pt"),
}


def download_model(name, directory="models"):
    if name not in MODEL_SOURCES:
        raise ValueError(f"Unknown model {name}; choose {list(MODEL_SOURCES)}")
    source, filename = MODEL_SOURCES[name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists():
        temporary = target.with_suffix(".download")
        if source.startswith("https:"):
            urllib.request.urlretrieve(source, temporary)
        else:
            import gdown

            result = gdown.download(id=source, output=str(temporary), quiet=False)
            if not result:
                raise RuntimeError("Model download failed")
        if temporary.stat().st_size < 100000:
            raise ValueError("Downloaded file is not a plausible model checkpoint")
        temporary.replace(target)
    write_json(
        target.with_suffix(".manifest.json"),
        {
            "source": source,
            "sha256": sha256(target),
            "filename": filename,
            "reference_commit": SPORTS_COMMIT if name.startswith("football") else None,
            "license": "Ultralytics AGPL-3.0; verify separate checkpoint terms before redistribution",
            "validation": "not_evaluated",
        },
    )
    return target


def class_kind(name):
    value = name.lower().replace("_", " ").replace("-", " ")
    if value in {"ball", "sports ball", "football"}:
        return "ball", "unknown"
    if value in {"goalkeeper", "goal keeper"}:
        return "person", "goalkeeper"
    if value in {"player", "players", "outfield player"}:
        return "person", "outfield_player"
    if value in {"referee", "referees"}:
        return "person", "referee"
    if value == "person":
        return "person", "unknown"
    return None, "unknown"


def suppress_duplicate_people(detections, threshold=0.8):
    """Deduplicate near-identical boxes across role labels; never suppress an overlapping ball."""
    people = [(i, d) for i, d in enumerate(detections) if d["class_name"] == "person"]
    if len(people) < 2:
        return detections
    boxes = [[d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]]
             for _, d in people]
    keep = cv2.dnn.NMSBoxes(boxes, [d["score"] for _, d in people], 0.0, threshold)
    kept = {people[int(index)][0] for index in np.asarray(keep).ravel()}
    return [d for i, d in enumerate(detections) if d["class_name"] != "person" or i in kept]


class Detector:
    def __init__(self, checkpoint, device="cpu", imgsz=640, confidence=0.08):
        from ultralytics import YOLO

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing supplied checkpoint: {checkpoint}. Select your local best.pt.")
        self.model = YOLO(str(checkpoint))
        if self.model.task != "detect":
            raise ValueError(f"Primary detector requires a detect checkpoint, received {self.model.task}")
        self.device, self.imgsz, self.confidence = device, imgsz, confidence
        self.names = self.model.names
        self.allowed = [i for i, n in self.names.items() if class_kind(n)[0]]
        if not self.allowed:
            raise ValueError(f"No supported classes in checkpoint: {self.names}")
        self.manifest = {
            "checkpoint": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
            "names": self.names,
            "backend": "ultralytics",
        }
        self.mode = "full"
        self.tile_size, self.tile_overlap = 960, .25
        self.ball_tile_imgsz = 960
        self.enhance = False
        self.person_confidence = self.ball_confidence = confidence
        self.roi = None

    def configure(self, config):
        from .storage import read_json

        self.mode = config.inference_mode
        self.tile_size, self.tile_overlap = config.inference_tile_size, config.inference_tile_overlap
        self.ball_tile_imgsz = config.ball_tile_imgsz
        self.enhance = config.enhance_contrast
        self.person_confidence = config.person_confidence if config.person_confidence is not None else config.confidence
        self.ball_confidence = config.ball_confidence if config.ball_confidence is not None else config.confidence
        self.roi = read_json(config.court_roi_file) if config.court_roi_file else None
        if self.roi:
            points = np.asarray(self.roi.get("polygon"), float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("Court ROI needs 3+ finite pixel-coordinate polygon points")
        self.manifest["inference"] = {"mode": self.mode, "tile_size": self.tile_size,
                                      "tile_overlap": self.tile_overlap, "contrast": self.enhance,
                                      "person_confidence": self.person_confidence,
                                      "ball_confidence": self.ball_confidence,
                                      "ball_tile_imgsz": self.ball_tile_imgsz,
                                      "court_roi": self.roi}

    def _predict_view(self, image, inference_size=None):
        result = self.model.predict(
            image,
            device=self.device,
            imgsz=inference_size if inference_size else self.imgsz,
            conf=min(self.confidence, self.person_confidence, self.ball_confidence),
            classes=self.allowed,
            verbose=False,
        )[0]
        rows = []
        for b in result.boxes.cpu().numpy().data:
            name = self.names[int(b[5])]
            kind, role = class_kind(name)
            rows.append(
                {
                    "box": b[:4].astype(float).tolist(),
                    "score": float(b[4]),
                    "class_name": kind,
                    "role": role,
                    "model_class": name,
                }
            )
        return rows

    def predict(self, image):
        from .inference import tile_windows, enhance_luminance, merge_detections, filter_detections

        prepared = enhance_luminance(image) if self.enhance else image
        rows = self._predict_view(prepared)
        height, width = image.shape[:2]
        if self.mode == "full+tiled":
            for x1, y1, x2, y2 in tile_windows(width, height, self.tile_size, self.tile_overlap):
                if x1 == y1 == 0 and x2 == width and y2 == height:
                    continue
                for item in self._predict_view(prepared[y1:y2, x1:x2]):
                    box = np.asarray(item["box"])
                    # Ignore boxes cut at internal tile edges; overlapping neighbours supply full boxes.
                    if ((x1 > 0 and box[0] < 2) or (y1 > 0 and box[1] < 2)
                            or (x2 < width and box[2] > x2 - x1 - 2)
                            or (y2 < height and box[3] > y2 - y1 - 2)):
                        continue
                    rows.append({**item, "box": (box + [x1, y1, x1, y1]).tolist()})
            rows = merge_detections(rows)
        if self.roi and self.roi.get("image_size") != [width, height]:
            raise ValueError("Court ROI image_size differs from this video")
        return filter_detections(rows, image.shape, self.person_confidence, self.ball_confidence,
                                 self.roi["polygon"] if self.roi else None)

    def predict_ball(self, image, tiled=False, base_detections=None):
        if not tiled:
            return [d for d in self.predict(image) if d["class_name"] == "ball"]
        from .inference import tile_windows, enhance_luminance, merge_detections, filter_detections

        height, width = image.shape[:2]
        prepared = enhance_luminance(image) if self.enhance else image
        size, overlap = 640, 128
        output = []
        baseline = base_detections if base_detections is not None else self.predict(image)
        output = [d for d in baseline if d["class_name"] == "ball"]
        for x, y, x2, y2 in tile_windows(width, height, size, overlap / size):
            if x == y == 0 and x2 == width and y2 == height:
                continue
            for item in self._predict_view(prepared[y:y2, x:x2], self.ball_tile_imgsz):
                if item["class_name"] == "ball":
                    output.append({**item, "box": (np.asarray(item["box"]) + [x, y, x, y]).tolist()})
        return filter_detections(merge_detections(output), image.shape, self.person_confidence,
                                 self.ball_confidence, self.roi["polygon"] if self.roi else None)


class RFDETRDetector:
    """Optional challenger; dependency installed only when requested."""

    def __init__(self, checkpoint=None, device="cpu", confidence=0.08):
        from rfdetr import RFDETRSmall
        from rfdetr.assets.coco_classes import COCO_CLASSES

        self.model = RFDETRSmall(device=device, **({"pretrain_weights": checkpoint} if checkpoint else {}))
        self.names = COCO_CLASSES
        self.confidence = confidence

    def predict(self, image):
        detections = self.model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=self.confidence)
        rows = []
        for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            name = self.names[int(class_id)]
            kind, role = class_kind(name)
            if kind:
                rows.append(
                    {
                        "box": box.tolist(),
                        "score": float(score),
                        "class_name": kind,
                        "role": role,
                        "model_class": name,
                    }
                )
        return rows
