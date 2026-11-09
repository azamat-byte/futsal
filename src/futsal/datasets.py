import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from .storage import read_json, sha256, utcnow, write_json
from .video import frames, load_image, save_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def yolo_boxes(path, width, height, n_classes):
    issues = []
    boxes = []
    if not Path(path).exists():
        return [], ["missing_label"]
    for line_number, line in enumerate(Path(path).read_text("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = list(map(float, line.split()))
            if not np.isfinite(parts).all():
                raise ValueError("nonfinite label fields")
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                cls = parts[0]
                vertices = np.asarray(parts[1:]).reshape(-1, 2)
                if np.any(vertices < 0) or np.any(vertices > 1):
                    raise ValueError("polygon outside image")
                lo, hi = vertices.min(0), vertices.max(0)
                cx, cy = (lo + hi) / 2
                w, h = hi - lo
            else:
                raise ValueError("expected detection box or YOLO segmentation polygon")
            if cls != int(cls) or not 0 <= cls < n_classes:
                raise ValueError("invalid class")
            if (
                w <= 0
                or h <= 0
                or min(cx - w / 2, cy - h / 2) < -1e-5
                or max(cx + w / 2, cy + h / 2) > 1.00001
            ):
                raise ValueError("box outside image or nonpositive size")
            boxes.append(
                {
                    "class_id": int(cls),
                    "x1": (cx - w / 2) * width,
                    "y1": (cy - h / 2) * height,
                    "x2": (cx + w / 2) * width,
                    "y2": (cy + h / 2) * height,
                }
            )
        except ValueError as exc:
            issues.append(f"line_{line_number}: {exc}")
    if len({tuple(b.values()) for b in boxes}) != len(boxes):
        issues.append("duplicate_boxes")
    return boxes, issues


def perceptual_hash(image):
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.dct(gray)[:8, :8].ravel()[1:]
    return sum(int(v > np.median(low)) << i for i, v in enumerate(low))


def annotation_warnings(boxes, names, width, height):
    warnings = []
    for i, box in enumerate(boxes):
        name = names[box["class_id"]].lower()
        w = box["x2"] - box["x1"]
        h = box["y2"] - box["y1"]
        if name in {"player", "goalkeeper", "person"}:
            if h < height * 0.035 or w * h < width * height * 0.0002:
                warnings.append(f"box_{i}: small_person_component_check_fragmented_polygon")
            elif w > h * 1.5:
                warnings.append(f"box_{i}: unusually_wide_person_check_occlusion_or_label")
    return warnings


def label_path(image):
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def audit_dataset(source, output, names=None, copy=False):
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output = Path(output)
    if output.exists() and (output / "manifest.json").exists():
        raise FileExistsError("Choose a new immutable dataset import directory")
    output.mkdir(parents=True, exist_ok=True)
    data_file = next(iter(sorted(source.glob("*.yaml"))), None)
    data = yaml.safe_load(data_file.read_text("utf-8")) if data_file else {}
    names = names or data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        raise ValueError("Class names required: provide data.yaml with names")
    records = []
    distribution = Counter()
    hashes = defaultdict(list)
    for image_path in sorted(source.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(source)
        split = next((x for x in relative.parts if x in {"train", "valid", "val", "test"}), "unassigned")
        image = load_image(image_path)
        issues = []
        boxes = []
        phash = None
        width = height = None
        if image is None:
            issues.append("unreadable_image")
        else:
            height, width = image.shape[:2]
            boxes, issues = yolo_boxes(label_path(image_path), width, height, len(names))
            phash = perceptual_hash(image)
            for box in boxes:
                distribution[names[box["class_id"]]] += 1
        digest = sha256(image_path)
        record = {
            "id": str(relative).replace("\\", "/"),
            "image": str(image_path),
            "label": str(label_path(image_path)),
            "sha256": digest,
            "phash": str(phash),
            "split": split,
            "group": None,
            "reviewed": False,
            "complete": False,
            "width": width,
            "height": height,
            "boxes": boxes,
            "issues": issues,
            "warnings": annotation_warnings(boxes, names, width, height) if width and height else [],
        }
        hashes[digest].append(record["id"])
        if copy:
            target = output / "raw" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target)
            record["image"] = str(target.resolve())
            labels = label_path(image_path)
            if labels.exists():
                target_label = output / "raw" / labels.relative_to(source)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(labels, target_label)
                record["label"] = str(target_label.resolve())
        records.append(record)
    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    near = []
    # Intended for curated small datasets; report capped pairs on very large imports.
    candidates = records[:10000]
    for i, a in enumerate(candidates):
        if a["phash"] == "None":
            continue
        for b in candidates[i + 1 :]:
            if a["split"] == b["split"] or b["phash"] == "None" or a["sha256"] == b["sha256"]:
                continue
            if (int(a["phash"]) ^ int(b["phash"])).bit_count() <= 6:
                near.append([a["id"], b["id"]])
                if len(near) >= 1000:
                    break
    manifest = {
        "schema_version": 1,
        "created": utcnow(),
        "source": str(source),
        "source_metadata": data.get("roboflow", {}),
        "source_yaml_sha256": sha256(data_file) if data_file else None,
        "names": names,
        "import_policy": "YOLO polygons converted to enclosing detection boxes; original labels preserved",
        "records": records,
        "class_counts": dict(distribution),
        "exact_duplicates": duplicate_groups,
        "near_duplicate_cross_split_candidates": near,
        "near_duplicate_check_limit": 10000,
        "visual_review": "required_for_all_images",
        "quality_status": "unverified",
        "total_images": len(records),
        "images_with_issues": sum(bool(r["issues"]) for r in records),
        "images_with_warnings": sum(bool(r.get("warnings")) for r in records),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def remove_class_version(dataset, target, class_name):
    """Remove an explicitly unwanted class in a new version, never mutate raw labels."""
    dataset = Path(dataset)
    manifest, records = reviewed_records(dataset)
    if class_name not in manifest["names"]:
        raise ValueError("Class does not exist")
    class_id = manifest["names"].index(class_name)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    affected = []
    for record in records:
        if any(b["class_id"] == class_id for b in record["boxes"]):
            affected.append(record["id"])
            record["complete"] = False
            record["reviewed"] = False
            record["issues"] = [*record["issues"], "removed_class_object_needs_reannotation"]
        record["boxes"] = [
            {**b, "class_id": b["class_id"] - int(b["class_id"] > class_id)}
            for b in record["boxes"]
            if b["class_id"] != class_id
        ]
    names = [n for n in manifest["names"] if n != class_name]
    counts = Counter(names[b["class_id"]] for r in records for b in r["boxes"])
    updated = {
        **manifest,
        "created": utcnow(),
        "records": records,
        "names": names,
        "class_counts": dict(counts),
        "parent_dataset": str(dataset.resolve()),
        "removed_class": class_name,
        "affected_images": affected,
        "images_with_issues": sum(bool(r["issues"]) for r in records),
    }
    write_json(target / "manifest.json", updated)
    return updated


def review_record(dataset, record_id, boxes, group, split, complete):
    dataset = Path(dataset)
    manifest = read_json(dataset / "manifest.json")
    record = next(r for r in manifest["records"] if r["id"] == record_id)
    if not group.strip() or split not in {"train", "val", "test"}:
        raise ValueError("Source match/group and valid split are required")
    for box in boxes:
        values = np.array([box[k] for k in ("x1", "y1", "x2", "y2")], float)
        if (
            not np.isfinite(values).all()
            or box["class_id"] != int(box["class_id"])
            or not 0 <= int(box["class_id"]) < len(manifest["names"])
        ):
            raise ValueError("Invalid class or box")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= record["width"] and 0 <= y1 < y2 <= record["height"]):
            raise ValueError("Box must lie inside image with positive size")
    revisions = read_json(dataset / "reviews.json", {})
    revisions[record_id] = {
        "boxes": boxes,
        "group": group,
        "split": split,
        "complete": bool(complete),
        "reviewed": True,
        "time": utcnow(),
    }
    with (dataset / "review_history.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps({"record_id": record_id, **revisions[record_id]}) + "\n")
    write_json(dataset / "reviews.json", revisions)


def reviewed_records(dataset):
    manifest = read_json(Path(dataset) / "manifest.json")
    review = read_json(Path(dataset) / "reviews.json", {})
    return manifest, [{**r, **review.get(r["id"], {})} for r in manifest["records"]]


def export_yolo(dataset, target):
    manifest, records = reviewed_records(dataset)
    records = [r for r in records if r["reviewed"] and r["complete"]]
    if not records:
        raise ValueError("No fully reviewed complete annotations")
    groups = defaultdict(set)
    seen = {}
    for record in records:
        groups[record["group"]].add(record["split"])
        if record["sha256"] in seen and seen[record["sha256"]] != record["split"]:
            raise ValueError("Exact duplicate across splits")
        seen[record["sha256"]] = record["split"]
    if any(len(s) > 1 for s in groups.values()):
        raise ValueError("Same source match/group assigned to multiple splits")
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            if a["split"] != b["split"] and a["phash"] != "None" and b["phash"] != "None":
                if (int(a["phash"]) ^ int(b["phash"])).bit_count() <= 6:
                    raise ValueError(f"Unresolved near-duplicate leakage: {a['id']} / {b['id']}")
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for record in records:
        filename = sha256(record["image"])[:16] + Path(record["image"]).suffix
        images = target / "images" / record["split"]
        labels = target / "labels" / record["split"]
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image"], images / filename)
        lines = []
        for b in record["boxes"]:
            w, h = record["width"], record["height"]
            lines.append(
                f"{int(b['class_id'])} {(b['x1'] + b['x2']) / 2 / w:.8f} {(b['y1'] + b['y2']) / 2 / h:.8f} "
                f"{(b['x2'] - b['x1']) / w:.8f} {(b['y2'] - b['y1']) / h:.8f}"
            )
        (labels / Path(filename).with_suffix(".txt")).write_text("\n".join(lines), "utf-8")
    (target / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(target),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": manifest["names"],
            },
            sort_keys=False,
        ),
        "utf-8",
    )
    write_json(
        target / "provenance.json",
        {
            "source_dataset": str(Path(dataset).resolve()),
            "created": utcnow(),
            "reviewed_only": True,
            "records": records,
            "leakage_checks_passed": True,
        },
    )
    return target


def working_export(dataset, target):
    """An explicitly unverified snapshot for review/interchange, not a training release."""
    manifest, records = reviewed_records(dataset)
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for i, record in enumerate(records):
        images = target / "train/images"
        labels = target / "train/labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image"], images / f"{i:05d}.jpg")
        w, h = record["width"], record["height"]
        if not w or not h:
            continue
        boxes = []
        seen = set()
        for b in record["boxes"]:
            key = tuple(b.values())
            if key in seen:
                continue
            seen.add(key)
            boxes.append(
                f"{int(b['class_id'])} {(b['x1'] + b['x2']) / 2 / w:.8f} {(b['y1'] + b['y2']) / 2 / h:.8f} "
                f"{(b['x2'] - b['x1']) / w:.8f} {(b['y2'] - b['y1']) / h:.8f}"
            )
        (labels / f"{i:05d}.txt").write_text("\n".join(boxes), "utf-8")
    (target / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(target),
                "train": "train/images",
                "names": manifest["names"],
                "nc": len(manifest["names"]),
            }
        ),
        "utf-8",
    )
    write_json(
        target / "provenance.json",
        {
            "reviewed_only": False,
            "leakage_checks_passed": False,
            "purpose": "annotation review; not an approved training/validation release",
            "parent_dataset": str(Path(dataset).resolve()),
            "records": records,
        },
    )
    return target


def propose_video(source, detector, target, start=0, duration=60, count=30, group=None):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    hashes = []
    records = []
    from .models import class_kind

    names = list(
        dict.fromkeys(
            detector.names[i] for i in detector.allowed if class_kind(detector.names[i])[1] != "referee"
        )
    )
    for frame in frames(source, start, duration, fps=max(count / duration, 0.01)):
        phash = perceptual_hash(frame.image)
        if any((phash ^ h).bit_count() < 5 for h in hashes):
            continue
        found = [d for d in detector.predict(frame.image) if d["model_class"] in names]
        h, w = frame.image.shape[:2]
        image = target / "train" / "images" / f"pts-{frame.index}.jpg"
        save_image(image, frame.image)
        label = label_path(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for d in found:
            x1, y1, x2, y2 = d["box"]
            lines.append(
                f"{names.index(d['model_class'])} {(x1 + x2) / 2 / w} {(y1 + y2) / 2 / h} {(x2 - x1) / w} {(y2 - y1) / h}"
            )
        label.write_text("\n".join(lines), "utf-8")
        records.append(
            {
                "image": str(image),
                "video": str(Path(source).resolve()),
                "time_s": frame.time,
                "pts": frame.index,
                "group": group,
                "status": "proposed",
                "detections": found,
            }
        )
        hashes.append(phash)
        if len(records) >= count:
            break
    (target / "data.yaml").write_text(yaml.safe_dump({"names": names}), "utf-8")
    write_json(
        target / "proposals.json",
        {
            "records": records,
            "model": detector.manifest,
            "warning": "Incomplete predictions; all images need full annotation review before supervised training",
        },
    )
    return records
