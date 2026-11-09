"""Compare jersey LAB/HSV and optional tutorial-style SigLIP -> UMAP -> KMeans."""

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .storage import write_json
from .teams import jersey_feature
from .video import frame_at, save_image


def hsv_feature(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    h = hsv[:, 0] / 180 * 2 * np.pi
    saturation = hsv[:, 1] / 255
    return np.array([np.median(np.cos(h) * saturation), np.median(np.sin(h) * saturation),
                     np.median(saturation), np.median(hsv[:, 2]) / 255])


def siglip_embeddings(crops, device="cpu", batch_size=8):
    """Public appearance encoder only; never replaces the supplied object detector."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, SiglipVisionModel

    model_id = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = SiglipVisionModel.from_pretrained(model_id).eval().to(device)
    embeddings = []
    with torch.inference_mode():
        for offset in range(0, len(crops), batch_size):
            images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                      for c in crops[offset:offset + batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            output = model(**inputs).last_hidden_state.mean(dim=1)
            embeddings.append(torch.nn.functional.normalize(output, dim=1).cpu().numpy())
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.vstack(embeddings)


def compare_teams(source, detection_log, output, include_siglip=False, device="cpu", limit=180):
    """Sample outfield detections; report cluster separation without claiming team accuracy."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    crops, lab, hsv, records = [], [], [], []
    lines = Path(detection_log).read_text("utf-8").splitlines()
    stride = max(1, len(lines) // 24)
    for line in lines[::stride]:
        frame = json.loads(line)
        image = frame_at(source, frame["time_s"])
        for d in frame["detections"]:
            # Logs from the tactical pipeline preserve the model's actual class name.
            if d.get("model_class", d.get("class_name")) != "player":
                continue
            box = d.get("box", d.get("xyxy"))
            feature = jersey_feature(image, box)
            if feature is None:
                continue
            x1, y1, x2, y2 = np.asarray(box, int)
            x1, y1 = max(0, x1), max(0, y1)
            crop = image[y1:min(image.shape[0], y2), x1:min(image.shape[1], x2)]
            if crop.size < 60:
                continue
            ch, cw = crop.shape[:2]
            jersey = crop[int(ch * .18):max(int(ch * .55), int(ch * .18) + 1),
                          int(cw * .2):max(int(cw * .8), int(cw * .2) + 1)]
            crops.append(crop)
            lab.append(feature)
            hsv.append(hsv_feature(jersey))
            records.append({"time_s": frame["time_s"], "box": box})
            save_image(output / "crops" / f"{len(crops) - 1:04d}.jpg", crop)
            if len(crops) >= limit:
                break
        if len(crops) >= limit:
            break
    if len(crops) < 12:
        raise ValueError("Need at least 12 usable outfield crops for a cluster comparison")
    features = {"LAB": np.asarray(lab), "HSV": np.asarray(hsv)}
    if include_siglip:
        import umap

        embeddings = siglip_embeddings(crops, device)
        np.save(output / "siglip_embeddings.npy", embeddings)
        features["SigLIP_UMAP"] = umap.UMAP(n_components=3, n_neighbors=min(15, len(crops) - 1),
                                           random_state=42).fit_transform(embeddings)
    report = {"sample_count": len(crops), "records": records, "methods": {},
              "team_accuracy": None, "status": "unreviewed_cluster_diagnostics",
              "note": "Silhouette measures separation, not team correctness. Numeric clusters need manual mapping."}
    for name, data in features.items():
        model = KMeans(2, n_init=10, random_state=42).fit(data)
        labels = model.labels_
        score = float(silhouette_score(data, labels)) if len(np.unique(labels)) == 2 else None
        report["methods"][name] = {"silhouette": score, "labels": labels.tolist()}
        np.save(output / f"{name}_features.npy", data)
    write_json(output / "comparison.json", report)
    return report
