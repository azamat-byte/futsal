"""Futsal landmark annotation/training and reviewed geometric diagnostics."""

from pathlib import Path
import cv2
import numpy as np
import yaml

from .storage import read_json, write_json
from .video import save_image


def line_candidates(image):
    """Image-space line proposals only; does not claim semantic court correspondences."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 85, 255]))
    white[: int(image.shape[0] * 0.15)] = 0
    edges = cv2.Canny(white, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=image.shape[1] * 0.08, maxLineGap=15)
    return [] if lines is None else lines[:, 0].tolist()


class CourtLandmarkDetector:
    def __init__(self, checkpoint, schema, device="cpu"):
        from ultralytics import YOLO

        self.schema = read_json(schema)
        if "court_points" not in self.schema or len(self.schema["court_points"]) != len(self.schema["names"]):
            raise ValueError("Court schema requires names and matching court_points")
        self.model = YOLO(str(checkpoint))
        self.device = device

    def propose(self, image, time_s):
        from .geometry import CalibrationSpec

        result = self.model.predict(image, device=self.device, verbose=False)[0]
        if result.keypoints is None or not len(result.keypoints.xy):
            return None
        points = result.keypoints.xy[0].cpu().numpy()
        confidence = (
            result.keypoints.conf[0].cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(len(points))
        )
        court = np.asarray(self.schema["court_points"], float)
        if len(points) != len(court):
            raise ValueError("Court model output count differs from semantic landmark schema")
        good = np.flatnonzero((confidence >= 0.65) & (points[:, 0] > 0) & (points[:, 1] > 0))
        if len(good) < 6:
            return None
        # Geometric held-out landmark checks do not replace independent model evaluation.
        check = good[::3]
        fit = np.array([i for i in good if i not in check])
        if len(fit) < 4:
            return None
        return CalibrationSpec(
            time_s=time_s,
            image_points=points[fit].tolist(),
            court_points=court[fit].tolist(),
            validation_image_points=points[check].tolist(),
            validation_court_points=court[check].tolist(),
            length=self.schema.get("length", 40),
            width=self.schema.get("width", 20),
            scale_verified=self.schema.get("scale_verified", False),
            method="trained_landmarks",
            landmark_model_accepted=self.schema.get("acceptance_status") == "acceptance_passed",
        )


def keypoint_export(annotation_file, target):
    """JSON: landmark_names plus reviewed frames with image, split, group, points [[x,y,v]]."""
    spec = read_json(annotation_file)
    names = spec["landmark_names"]
    if len(names) < 4 or len(set(names)) != len(names):
        raise ValueError("Need >=4 unique semantic court landmarks")
    target = Path(target).resolve()
    groups = {}
    records = spec["frames"]
    for frame in records:
        if not frame.get("reviewed") or not frame.get("group"):
            raise ValueError("Every keypoint frame requires review and source group")
        groups.setdefault(frame["group"], set()).add(frame["split"])
        if len(frame["points"]) != len(names):
            raise ValueError("Landmark count mismatch")
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError("Keypoint source group leaks across splits")
    target.mkdir(parents=True, exist_ok=False)
    for index, frame in enumerate(records):
        from .video import load_image

        image = load_image(frame["image"])
        if image is None:
            raise ValueError("Unreadable keypoint source image")
        height, width = image.shape[:2]
        points = np.asarray(frame["points"], float)
        if points.shape != (len(names), 3) or not np.isfinite(points).all():
            raise ValueError("Invalid keypoint array")
        if not np.isin(points[:, 2], [0, 1, 2]).all():
            raise ValueError("Visibility must be 0, 1 or 2")
        for point in points:
            if point[2] == 0:
                point[:2] = 0
            elif not (0 <= point[0] <= width and 0 <= point[1] <= height):
                raise ValueError("Visible landmark outside image")
        points[:, 0] /= width
        points[:, 1] /= height
        images = target / "images" / frame["split"]
        labels = target / "labels" / frame["split"]
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        save_image(images / f"{index:06d}.jpg", image)
        # Whole image represents one court instance, with named sparse landmarks.
        (labels / f"{index:06d}.txt").write_text(
            "0 0.5 0.5 1 1 " + " ".join(map(str, points.ravel())), "utf-8"
        )
    (target / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(target),
                "train": "images/train",
                "val": "images/val",
                "names": ["court"],
                "kpt_shape": [len(names), 3],
            }
        ),
        "utf-8",
    )
    write_json(
        target / "landmark_schema.json",
        {
            "names": names,
            "source": str(Path(annotation_file).resolve()),
            "flip_augmentation": "disabled unless semantic landmark remapping is provided",
        },
    )
    return target


def train_court(dataset, checkpoint, output, device="cpu", epochs=50):
    from ultralytics import YOLO

    path = Path(dataset)
    if not (path / "landmark_schema.json").exists():
        raise ValueError("Use a reviewed futsal keypoint export")
    return YOLO(checkpoint).train(
        data=str((path / "data.yaml").resolve()),
        device=device,
        epochs=epochs,
        imgsz=640,
        batch=2,
        workers=0,
        seed=42,
        fliplr=0,
        flipud=0,
        degrees=0,
        project=str(output),
        name="court-candidate",
        exist_ok=False,
    )
