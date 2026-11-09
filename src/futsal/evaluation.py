from pathlib import Path
import time

import numpy as np

from .datasets import reviewed_records
from .models import Detector, class_kind
from .storage import utcnow, write_json
from .video import load_image, save_image


def iou(box, boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return np.zeros(0)
    box = np.asarray(box, float)
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2]))
    intersection = intersection.prod(axis=1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    other = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + other - intersection, 1e-12)


def detection_metrics(truth, predictions, class_name, threshold=0.5):
    """101-point interpolated AP, per-image greedy one-to-one matching by score."""
    gt = {k: [r["box"] for r in rows if r["class"] == class_name] for k, rows in truth.items()}
    total = sum(map(len, gt.values()))
    candidates = sorted(
        [
            (r["score"], k, r["box"])
            for k, rows in predictions.items()
            for r in rows
            if r["class"] == class_name
        ],
        reverse=True,
    )
    matched = {k: set() for k in gt}
    tp = []
    for score, key, box in candidates:
        scores = iou(box, gt.get(key, []))
        available = [i for i in range(len(scores)) if i not in matched.get(key, set())]
        index = max(available, key=lambda i: scores[i]) if available else None
        success = index is not None and scores[index] >= threshold
        if success:
            matched[key].add(index)
        tp.append(int(success))
    cumulative = np.cumsum(tp)
    precision = cumulative / np.arange(1, len(tp) + 1)
    recall = cumulative / max(total, 1)
    ap = (
        float(
            np.mean(
                [np.max(precision[recall >= r]) if np.any(recall >= r) else 0 for r in np.linspace(0, 1, 101)]
            )
        )
        if total
        else None
    )
    return {
        "ap": ap,
        "precision": float(sum(tp) / len(tp)) if tp else 0.0,
        "recall": float(sum(tp) / total) if total else None,
        "ground_truth": total,
        "predictions": len(tp),
        "tp": sum(tp),
        "fp": len(tp) - sum(tp),
        "fn": total - sum(tp),
    }


def evaluate_detector(
    dataset, model, output, device="cpu", imgsz=640, limit=None, exploratory=False, split="val"
):
    manifest, records = reviewed_records(dataset)
    if exploratory:
        records = [r for r in records if not r["issues"]]
    else:
        records = [r for r in records if r["reviewed"] and r["complete"] and r["split"] == split]
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("No eligible reviewed benchmark; use --exploratory only for unverified diagnostics")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    detector = Detector(model, device, imgsz, 0.03)
    truth, predictions = {}, {}
    latency = []
    for record in records:
        image = load_image(record["image"])
        start = time.perf_counter()
        found = detector.predict(image)
        latency.append(time.perf_counter() - start)
        # Role-agnostic person benchmark is intentional for generic and sports checkpoint comparison.
        truth[record["id"]] = [
            {
                "class": class_kind(manifest["names"][b["class_id"]])[0],
                "box": [b[k] for k in ("x1", "y1", "x2", "y2")],
            }
            for b in record["boxes"]
        ]
        predictions[record["id"]] = [
            {"class": d["class_name"], "score": d["score"], "box": d["box"]} for d in found
        ]
        if len(latency) <= 12:
            import cv2

            for d in found:
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
            save_image(output / "overlays" / f"{len(latency):03d}.jpg", image)
    metrics = {}
    for name in ("person", "ball"):
        values = [detection_metrics(truth, predictions, name, t) for t in np.arange(0.5, 0.951, 0.05)]
        metrics[name] = {
            "AP50": values[0]["ap"],
            "AP50_95": float(np.mean([v["ap"] for v in values])) if values[0]["ap"] is not None else None,
            "at_exported_score_floor": values[0],
        }
    result = {
        "created": utcnow(),
        "status": "exploratory_unverified_labels" if exploratory else "evaluated",
        "dataset": str(Path(dataset).resolve()),
        "images": len(records),
        "split": "unverified_import_all_splits" if exploratory else split,
        "source_splits": dict(__import__("collections").Counter(r["split"] for r in records)),
        "scope": "role_agnostic_person_and_ball; excludes invalid labels",
        "model": detector.manifest,
        "imgsz": imgsz,
        "metrics": metrics,
        "warm_latency_s": float(np.mean(latency[1:])) if len(latency) > 1 else None,
        "warning": "Not an acceptance benchmark" if exploratory else None,
    }
    write_json(output / "evaluation.json", result)
    write_json(output / "predictions.json", predictions)
    write_json(output / "ground_truth.json", truth)
    return result


def train_model(export, model, output, device="cpu", epochs=50, imgsz=640):
    import yaml
    from ultralytics import YOLO

    export = Path(export).resolve()
    from .storage import read_json

    provenance = read_json(export / "provenance.json")
    if not provenance or not provenance.get("reviewed_only") or not provenance.get("leakage_checks_passed"):
        raise ValueError("Training requires a reviewed, leakage-checked export")
    splits = {r["split"] for r in provenance["records"]}
    if not {"train", "val"} <= splits:
        raise ValueError("Independent train and validation groups required")
    yaml.safe_load((export / "data.yaml").read_text("utf-8"))
    learner = YOLO(str(model))
    result = learner.train(
        data=str(export / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=2,
        workers=0,
        seed=42,
        deterministic=True,
        patience=10,
        project=str(Path(output).resolve()),
        name="candidate",
        exist_ok=False,
        hsv_h=0.01,
        mosaic=0.3,
        cache=False,
    )
    return result
