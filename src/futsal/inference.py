"""Configurable inference views and conservative, role-aware duplicate filtering."""

import cv2
import numpy as np


def tile_windows(width, height, size=960, overlap=.25):
    if size < 64 or not 0 <= overlap < .8:
        raise ValueError("Tile size must be >=64 and overlap in [0, .8)")
    step = max(1, round(size * (1 - overlap)))

    def starts(length):
        return sorted(set(range(0, max(1, length - size + 1), step)) | {max(0, length - size)})

    return [(x, y, min(x + size, width), min(y + size, height)) for y in starts(height) for x in starts(width)]


def enhance_luminance(image):
    """Mild optional CLAHE on luminance; no changes to court/jersey semantics."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def merge_detections(rows, person_iou=.7, ball_iou=.35):
    """NMS separately for all person roles and balls. Keep original confidence and class names."""
    output = []
    for category, threshold in (("person", person_iou), ("ball", ball_iou)):
        items = [r for r in rows if r["class_name"] == category]
        if not items:
            continue
        boxes = [[r["box"][0], r["box"][1], r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]]
                 for r in items]
        keep = cv2.dnn.NMSBoxes(boxes, [r["score"] for r in items], 0.0, threshold)
        selected = [items[int(i)] for i in np.asarray(keep).ravel()]
        if category == "ball":
            unique = []
            for item in sorted(selected, key=lambda d: -d["score"]):
                box = np.asarray(item["box"])
                duplicate = False
                for prior in unique:
                    other = np.asarray(prior["box"])
                    intersection = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])).prod()
                    small_area = min(np.prod(box[2:] - box[:2]), np.prod(other[2:] - other[:2]))
                    if intersection / max(small_area, 1e-6) >= .8:
                        duplicate = True
                        break
                if not duplicate:
                    unique.append(item)
            selected = unique
        output.extend(selected)
    return output


def filter_detections(rows, image_shape, person_confidence=.08, ball_confidence=.08, court_polygon=None):
    """Optional reviewed image-space court ROI plus basic malformed-box rejection.

    ROI is deliberately manual: an incorrect automatic court mask can delete real players.
    People use feet; balls use their centre. A fixed ROI is only appropriate for a fixed camera.
    """
    height, width = image_shape[:2]
    contour = np.asarray(court_polygon, np.float32) if court_polygon else None
    output = []
    for item in rows:
        box = np.asarray(item["box"], float)
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(item["score"]):
            continue
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        min_size = .5 if item["class_name"] == "ball" else 2
        if box[2] - box[0] < min_size or box[3] - box[1] < min_size:
            continue
        floor = ball_confidence if item["class_name"] == "ball" else person_confidence
        if item["score"] < floor:
            continue
        point = ((box[0] + box[2]) / 2,
                 (box[1] + box[3]) / 2 if item["class_name"] == "ball" else box[3])
        if contour is not None and cv2.pointPolygonTest(contour, point, False) < 0:
            continue
        output.append({**item, "box": box.tolist()})
    return output
