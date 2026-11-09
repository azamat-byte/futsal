"""Court keypoint auto-labelling driver.

  infer <vid> <model.pt> <tag>   run the keypoint model on every candidate frame (two input modes)
  label <vid> <tag>              homography fit + white-line refinement + acceptance -> labels_<tag>.jsonl
  contact <vid> <tag> [n]        random contact sheet of accepted labels for visual review

Data lives in data/court_frames/<vid>/ (cand/*.jpg from extract_court_frames.py).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from futsal.court_autolabel import (Refiner, apply_h, init_homography, load_schema,  # noqa: E402
                                    project_labels)

FRAMES = ROOT / "data" / "court_frames"
SCHEMA = load_schema(ROOT / "configs" / "court_schema_26.json")
COURT = SCHEMA["court_points"]
L, W = float(SCHEMA["length"]), float(SCHEMA["width"])
# dihedral symmetries of the template as 3x3 court->court maps
SYMS = {
    "id": np.eye(3),
    "mx": np.array([[-1, 0, L], [0, 1, 0], [0, 0, 1.0]]),
    "my": np.array([[1, 0, 0], [0, -1, W], [0, 0, 1.0]]),
    "r180": np.array([[-1, 0, L], [0, -1, W], [0, 0, 1.0]]),
}


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def standard_orientation(H, ref_xy):
    """x increases to the right and y (far->near touchline) increases downwards at ref point."""
    p0 = apply_h(H, [ref_xy])[0][0]
    px = apply_h(H, [ref_xy + np.array([2.0, 0])])[0][0] - p0
    py = apply_h(H, [ref_xy + np.array([0, 2.0])])[0][0] - p0
    return bool(px[0] > 0 and py[1] > 0 and abs(px[0]) > abs(px[1]) * 0.25 and np.linalg.det(np.c_[px, py]) > 0)


def relabel_standard(H, ref_xy):
    for name, S in SYMS.items():
        Hs = H @ S
        # reference point in the *relabelled* frame
        if standard_orientation(Hs, ref_xy):
            return Hs, name
    return None, None


def cmd_infer(vid, model_path, tag, limit=None):
    from ultralytics import YOLO
    model = YOLO(model_path)
    files = sorted((FRAMES / vid / "cand").glob("*.jpg"))[:limit]
    n, k = len(files), len(COURT)
    out = {m: {"xy": np.zeros((n, k, 2), np.float32), "conf": np.zeros((n, k), np.float32),
               "box": np.zeros(n, np.float32)} for m in ("stretch", "letter")}
    sizes = np.zeros((n, 2), np.int32)
    batch = 4  # NOTE: fp16 with batch>=16 silently returned no detections on this GPU/driver; keep fp32 + small batches
    for s in range(0, n, batch):
        imgs = [imread(f) for f in files[s:s + batch]]
        for j, im in enumerate(imgs):
            sizes[s + j] = im.shape[1], im.shape[0]
        for mode in ("stretch", "letter"):
            inp = [cv2.resize(im, (640, 640), interpolation=cv2.INTER_AREA) for im in imgs] if mode == "stretch" else imgs
            res = model.predict(inp, imgsz=640, conf=0.05, verbose=False, device=0, half=False)
            for j, r in enumerate(res):
                if r.keypoints is None or not len(r.boxes):
                    continue
                b = int(r.boxes.conf.argmax())
                xy = r.keypoints.xy[b].cpu().numpy().copy()
                cf = r.keypoints.conf[b].cpu().numpy() if r.keypoints.conf is not None else np.ones(k)
                if mode == "stretch":
                    xy[:, 0] *= imgs[j].shape[1] / 640.0
                    xy[:, 1] *= imgs[j].shape[0] / 640.0
                out[mode]["xy"][s + j], out[mode]["conf"][s + j] = xy, cf
                out[mode]["box"][s + j] = float(r.boxes.conf[b])
        if (s // batch) % 80 == 0:
            print(f"{vid} infer {s}/{n}", flush=True)
    np.savez_compressed(FRAMES / vid / f"kp_{tag}.npz", names=np.array([f.name for f in files]), sizes=sizes,
                        sxy=out["stretch"]["xy"], sconf=out["stretch"]["conf"], sbox=out["stretch"]["box"],
                        lxy=out["letter"]["xy"], lconf=out["letter"]["conf"], lbox=out["letter"]["box"])
    print(f"{vid}: saved kp_{tag}.npz", flush=True)


ACCEPT = dict(close=0.80, mean_dt=1.6, lines=4, inside=200)


def accept(score):
    return (score["close"] >= ACCEPT["close"] and score["mean_dt"] <= ACCEPT["mean_dt"]
            and score["lines"] >= ACCEPT["lines"] and score["inside"] >= ACCEPT["inside"])


def label_one(args):
    vid, name, size, hyps = args
    img = imread(FRAMES / vid / "cand" / name)
    if img is None:
        return {"file": name, "accepted": False, "reason": "unreadable"}
    h, w = img.shape[:2]
    ref = Refiner(img, L, W)
    best = None
    for mode, (kxy, kconf) in hyps.items():
        for thr in (0.5, 0.3):
            H0, pxy, pcourt = init_homography(kxy, kconf, COURT, thr_conf=thr, thr_px=6.0 * w / 960, min_pts=5)
            if H0 is None:
                continue
            out = ref.refine(H0, pxy, pcourt, prior_w=0.2)
            if out is None:
                continue
            H1 = out[0]
            sc = ref.score(H1)
            key = (sc["close"] * min(sc["lines"], 6), sc["inside"])
            if best is None or key > best[0]:
                best = (key, H1, sc, mode, thr)
    if best is None:
        return {"file": name, "accepted": False, "reason": "no_init"}
    _, H, sc, mode, thr = best
    # orientation: reference point = centroid of wireframe samples that fall inside the image
    xy, ok = ref._inside(H)
    ref_pt = ref.samples[ok].mean(0)
    Hs, sym = relabel_standard(H, ref_pt)
    rec = {"file": name, "w": w, "h": h, "score": sc, "mode": mode, "thr": thr}
    if Hs is None:
        rec.update(accepted=False, reason="orientation")
        return rec
    ok_acc = accept(sc)
    k = project_labels(Hs, COURT, w, h)
    rec.update(accepted=bool(ok_acc), reason="ok" if ok_acc else "score", sym=sym, H=Hs.tolist(),
               kpts=k.tolist(), n_vis=int((k[:, 2] > 0).sum()))
    return rec


def cmd_label(vid, tag, limit=None):
    d = np.load(FRAMES / vid / f"kp_{tag}.npz")
    names = d["names"]
    n = len(names) if limit is None else min(limit, len(names))
    jobs = []
    for i in range(n):
        hyps = {"stretch": (d["sxy"][i], d["sconf"][i]), "letter": (d["lxy"][i], d["lconf"][i])}
        jobs.append((vid, str(names[i]), tuple(d["sizes"][i]), hyps))
    out_path = FRAMES / vid / f"labels_{tag}.jsonl"
    done = 0
    acc = 0
    with Pool(max(1, min(6, __import__("os").cpu_count() or 2))) as pool, out_path.open("w", encoding="utf-8") as f:
        for rec in pool.imap(label_one, jobs, chunksize=4):
            f.write(json.dumps(rec) + "\n")
            done += 1
            acc += bool(rec["accepted"])
            if done % 200 == 0:
                print(f"{vid} label {done}/{n} accepted {acc}", flush=True)
    print(f"{vid}: {acc}/{n} accepted -> {out_path.name}", flush=True)


def cmd_contact(vid, tag, n=24, only="accepted"):
    recs = [json.loads(l) for l in (FRAMES / vid / f"labels_{tag}.jsonl").read_text("utf-8").splitlines()]
    pool = [r for r in recs if (r["accepted"] if only == "accepted" else (not r["accepted"] and "kpts" in r))]
    rng = np.random.default_rng(1)
    pick = [pool[i] for i in rng.permutation(len(pool))[:n]]
    from futsal.court_autolabel import wireframe
    wire = wireframe(L, W)
    tiles = []
    for r in pick:
        img = imread(FRAMES / vid / "cand" / r["file"])
        h, w = img.shape[:2]
        H = np.array(r["H"])
        for pts in wire.values():
            xy, ww = apply_h(H, pts)
            xy = np.round(xy).astype(int)
            good = (ww > 1e-6) & (xy[:, 0] > -2000) & (xy[:, 0] < 4000) & (xy[:, 1] > -2000) & (xy[:, 1] < 4000)
            for a, b, g in zip(xy[:-1], xy[1:], good[:-1] & good[1:]):
                if g:
                    cv2.line(img, tuple(a), tuple(b), (60, 255, 60), 1, cv2.LINE_AA)
        for i, (x, y, v) in enumerate(r["kpts"]):
            if v > 0:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(img, str(i), (int(x) + 4, int(y) - 4), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        s = r["score"]
        cv2.putText(img, f"{r['file'][:6]} c={s['close']:.2f} l={s['lines']} {r.get('sym','')}", (6, 18), 0, 0.55, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (480, 270)))
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 270, cols * 480, 3), np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // cols) * 270:(i // cols + 1) * 270, (i % cols) * 480:(i % cols + 1) * 480] = t
    out = ROOT / "data" / "court_frames" / vid / f"contact_{tag}_{only}.jpg"
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out))
    print("saved", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "infer":
        cmd_infer(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else None)
    elif cmd == "label":
        cmd_label(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == "contact":
        cmd_contact(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 24,
                    sys.argv[5] if len(sys.argv) > 5 else "accepted")
