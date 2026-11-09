"""First milestone: inspect the supplied checkpoint and export real detection evidence."""

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .models import class_kind
from .storage import environment, sha256, write_json
from .video import VideoWriter, frame_at, frames, probe, save_image


def inspect_model(model, checkpoint):
    import torch
    import ultralytics

    names = {int(k): str(v) for k, v in model.names.items()}
    args = model.ckpt.get("train_args", {})
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "sha256": sha256(checkpoint),
        "task": model.task,
        "architecture": type(model.model).__name__,
        "architecture_yaml": model.model.yaml,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "names": names,
        "role_mapping": {k: class_kind(v) for k, v in names.items()},
        "unsupported_classes": {k: v for k, v in names.items() if class_kind(v)[0] is None},
        "checkpoint_ultralytics": model.ckpt.get("version"),
        "runtime_ultralytics": ultralytics.__version__,
        "training_arguments": {k: args.get(k) for k in ("model", "data", "epochs", "imgsz", "batch")},
        "checkpoint_reported_metrics": model.ckpt.get("train_metrics", {}),
        "metric_status": "checkpoint_reported_only_not_independently_validated",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def detection_check(source, checkpoint, output, start=0, duration=10, fps=15, imgsz=640,
                    device="cpu", confidence=0.15, sample_times=None):
    """Export every actual checkpoint class; unsupported mixed classes remain explicitly named."""
    import torch
    from ultralytics import YOLO

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if start < 0 or duration <= 0 or fps <= 0:
        raise ValueError("Start must be nonnegative; duration and fps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError(f"Expected an object detection checkpoint, received {model.task}")
    metadata = inspect_model(model, checkpoint)
    meta = probe(source)
    cadence = min(fps, meta["average_fps"] or fps)
    use_cuda = str(device) not in {"cpu", "mps"}
    if use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable. Select a GPU runtime or device='cpu'.")
        torch.ones(1, device=f"cuda:{device}" if str(device).isdigit() else device).sum().item()

    def predict(image):
        return model.predict(image, device=device, imgsz=imgsz, conf=confidence, verbose=False)[0]

    times = sample_times if sample_times is not None else np.linspace(start, start + duration * .9, 6)
    samples = []
    tiles = []
    for i, t in enumerate(times):
        image = frame_at(source, float(t))
        result = predict(image)
        annotated = result.plot()
        save_image(output / "frames" / f"sample_{i:02d}.jpg", annotated)
        tile = cv2.resize(annotated, (640, 360))
        cv2.putText(tile, f"Source {float(t):.2f}s", (10, 25), 0, .65, (255, 255, 255), 2)
        tiles.append(tile)
        samples.append({"requested_time_s": float(t), "counts": dict(Counter(
            model.names[int(c)] for c in result.boxes.cls.tolist()))})
    if len(tiles) == 6:
        save_image(output / "representative_frames.jpg", np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:]))))
    counts = Counter()
    inference_seconds = []
    processed = 0
    begin = time.perf_counter()
    writer = None
    try:
        with (output / "detections.jsonl").open("w", encoding="utf-8") as log:
            import json

            for frame in frames(source, start, duration, cadence):
                if use_cuda:
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                result = predict(frame.image)
                if use_cuda:
                    torch.cuda.synchronize()
                inference_seconds.append(time.perf_counter() - tick)
                detections = []
                for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                           result.boxes.conf.tolist()):
                    name = model.names[int(cls)]
                    counts[name] += 1
                    detections.append({"class_id": int(cls), "class_name": name,
                                       "confidence": score, "xyxy": box})
                log.write(json.dumps({"proxy_index": processed, "source_pts": frame.index,
                                      "time_s": frame.time, "detections": detections}) + "\n")
                image = result.plot()
                width = min(1280, image.shape[1])
                height = round(image.shape[0] * width / image.shape[1] / 2) * 2
                image = cv2.resize(image, (width, height))
                if writer is None:
                    writer = VideoWriter(output / "detections.mp4", cadence, (width, height))
                writer.write(image)
                processed += 1
    finally:
        if writer:
            writer.close()
    if not processed:
        raise ValueError("The selected interval contains no video frames")
    report = {
        "model": metadata, "source": meta, "environment": environment(),
        "config": {"start": start, "duration": duration, "fps": cadence, "imgsz": imgsz,
                   "device": str(device), "confidence": confidence},
        "frames": processed, "class_counts": dict(counts), "samples": samples,
        "inference_median_ms": float(np.median(inference_seconds) * 1000),
        "inference_p95_ms": float(np.percentile(inference_seconds, 95) * 1000),
        "processing_fps": processed / (time.perf_counter() - begin),
        "limitations": ["No independent detection accuracy or tracking accuracy measured",
                         "Sampled silent video; JSONL timestamps refer to original stream PTS",
                         "Missing referee class cannot be inferred from the mixed player-ball class",
                         "Raw outputs include mixed classes; tactical pipeline excludes unsupported classes"],
    }
    write_json(output / "report.json", report)
    return report
