"""Build a YOLO-pose court-keypoint dataset (26 landmarks) from accepted auto-labels + the user's hand labels.

usage: court_build_dataset.py <out_dir> [per_video=500] [tag ...]   (tags are label rounds, last one wins per frame)

* Frames keep their native 16:9 aspect (no square stretch) so letterbox inference sees the same geometry.
* Per video, up to `per_video` accepted frames are chosen: one best-scoring frame per time bin first, then fill.
* Splits are by 2-minute time blocks per video (no adjacent-frame leakage): ~8% val, ~6% test.
* The user's 52 hand labels (keypoints.v2i.yolov8) are added with their original split; their square
  stretch is undone by resizing to 16:9, which leaves normalised labels valid.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from futsal.court_autolabel import apply_h, load_schema, wireframe  # noqa: E402

FRAMES = ROOT / "data" / "court_frames"
SCHEMA = load_schema(ROOT / "configs" / "court_schema_26.json")
VIDS = ["v1", "v2", "v3", "v4", "v5", "v6"]
out = Path(sys.argv[1])
per_video = int(sys.argv[2]) if len(sys.argv) > 2 else 500
tags = sys.argv[3:] or ["r0"]
WIRE = wireframe()


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, im):
    cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(str(p))


def bbox_of(k, H, w, h):
    pts = []
    for p in WIRE.values():
        xy, ww = apply_h(H, p)
        ok = (ww > 1e-6) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        pts.append(xy[ok])
    pts = np.vstack([p for p in pts if len(p)] + [k[k[:, 2] > 0, :2]])
    x1, y1 = np.clip(pts.min(0) - 4, 0, [w - 1, h - 1])
    x2, y2 = np.clip(pts.max(0) + 4, 0, [w - 1, h - 1])
    return x1, y1, x2, y2


def split_of(vid, block):
    r = int(hashlib.md5(f"{vid}-{block}".encode()).hexdigest(), 16) % 100
    return "test" if r < 6 else "val" if r < 14 else "train"


def write_sample(split, name, im, kpts, box):
    h, w = im.shape[:2]
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    imwrite(out / "images" / split / f"{name}.jpg", im)
    x1, y1, x2, y2 = box
    vals = [0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
    for x, y, v in kpts:
        vals += [x / w, y / h, int(v)] if v > 0 else [0, 0, 0]
    (out / "labels" / split / f"{name}.txt").write_text(
        " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals), "utf-8")


if out.exists():
    raise SystemExit(f"{out} exists; choose a new folder")
stats = {}
for vid in VIDS:
    best = {}
    for tag in tags:
        p = FRAMES / vid / f"labels_{tag}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text("utf-8").splitlines():
            r = json.loads(line)
            if r["accepted"]:
                s = r["score"]
                r["q"] = s["close"] - 0.05 * s["mean_dt"]
                if r["file"] not in best or r["q"] > best[r["file"]]["q"]:
                    best[r["file"]] = r
    recs = sorted(best.values(), key=lambda r: r["file"])
    meta = json.loads((FRAMES / vid / "cand_meta.json").read_text("utf-8"))
    total = meta["count"]
    bins = np.array_split(np.arange(1, total + 1), per_video)
    chosen = []
    by_idx = {int(r["file"][:6]): r for r in recs}
    for b in bins:
        cands = [by_idx[i] for i in b if i in by_idx]
        if cands:
            chosen.append(max(cands, key=lambda r: r["q"]))
    left = [r for r in recs if r not in chosen]
    left.sort(key=lambda r: -r["q"])
    chosen += left[: max(0, per_video - len(chosen))]
    interval = meta["interval_s"]
    n = 0
    for r in chosen:
        idx = int(r["file"][:6])
        im = imread(FRAMES / vid / "cand" / r["file"])
        block = int((idx - 1) * interval // 120)
        split = split_of(vid, block)
        k = np.array(r["kpts"])
        write_sample(split, f"{vid}_{idx:06d}", im, k, bbox_of(k, np.array(r["H"]), im.shape[1], im.shape[0]))
        n += 1
    stats[vid] = {"accepted_pool": len(recs), "exported": n, "candidates": total}
    print(vid, stats[vid], flush=True)

# user's hand labels
user = ROOT / "keypoints.v2i.yolov8"
added = 0
for split_src, split_dst in (("train", "train"), ("valid", "val"), ("test", "test")):
    for lab in sorted((user / split_src / "labels").glob("*.txt")):
        im = imread(user / split_src / "images" / (lab.stem + ".jpg"))
        im = cv2.resize(im, (1024, 576), interpolation=cv2.INTER_CUBIC)
        v = np.array(lab.read_text().split(), float)
        (out / "images" / split_dst).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_dst).mkdir(parents=True, exist_ok=True)
        name = "user_" + lab.stem[:24]
        imwrite(out / "images" / split_dst / f"{name}.jpg", im)
        # remap user's index order -> same order (schema was derived from it); only force-clean invisibles
        k = v[5:].reshape(26, 3)
        k[k[:, 2] == 0, :2] = 0
        (out / "labels" / split_dst / f"{name}.txt").write_text(
            " ".join(f"{x:.6f}" for x in np.r_[v[:5], k.ravel()]).replace(".000000", ""), "utf-8")
        added += 1
stats["user_labels"] = added

(out / "data.yaml").write_text(yaml.safe_dump({
    "path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
    "kpt_shape": [26, 3], "flip_idx": SCHEMA["flip_idx"], "names": {0: "pitch"},
    "kpt_names": {0: SCHEMA["names"]},
}, sort_keys=False, allow_unicode=True), "utf-8")
shutil.copyfile(ROOT / "configs" / "court_schema_26.json", out / "landmark_schema.json")
(out / "build_stats.json").write_text(json.dumps(stats, indent=2), "utf-8")
print(json.dumps(stats, indent=2))
