"""Package measured local evidence for an offline, reproducible visual notebook demo."""

import json
import shutil
from pathlib import Path

import cv2

from futsal.storage import load_tracks, sha256
from futsal.video import frame_at

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    target = ROOT / ".cache/walkthrough_demo"
    target.mkdir(parents=True, exist_ok=True)
    optimized = ROOT / "outputs/optimization"
    manifest = json.loads((optimized / "ball_search960/manifest.json").read_text("utf-8"))
    source = manifest["asset"]["path"]
    logs = [json.loads(line) for line in (optimized / "ball_search960/detections.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (optimized / "full_motion/detections.jsonl").read_text().splitlines()]
    by_pts = {r["frame_index"]: r for r in baseline}
    samples = []
    for i, row in enumerate(logs[::3]):
        name = f"frames/{i:03d}.jpg"
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode('.jpg', frame_at(source, row["time_s"]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError('Cannot encode display frame')
        encoded.tofile(str(target / name))
        samples.append({**row, "image": name, "full_frame_detections": by_pts[row["frame_index"]]["detections"]})
    model = json.loads((ROOT / "outputs/01_detection_blue_court/report.json").read_text("utf-8"))["model"]
    assert model["sha256"] == sha256(ROOT / "best.pt")
    evidence = {
        "mode": "recorded_local_evidence", "model": model, "frames": samples,
        "source_sha256": manifest["source_sha256"], "source_description": "Spain–Portugal, source 120–123 seconds",
        "image_note": "JPEG display snapshots; detections were measured on original decoded frames",
        "run_config": manifest["config"],
        "quality": json.loads((ROOT / "reports/optimization/optimized_results.json").read_text()),
        "calibration": json.loads((ROOT / "configs/spain_demo_calibration.json").read_text())[0],
    }
    (target / "evidence.json").write_text(json.dumps(evidence, indent=2), "utf-8")
    for name, relative in (("baseline", "tracker_comparison/baseline"),
                           ("motion", "tracker_comparison/motion_confirmed"), ("ball", "ball_search960")):
        load_tracks(optimized / relative).to_json(target / f"{name}_tracks.json", orient="records")
    for name in ("ball_before_after.mp4", "tracking_before_after.mp4"):
        shutil.copy2(optimized / name, target / name)
    print(f"Prepared {len(samples)} real frames, cached detections and videos: {target}")
    return target


if __name__ == "__main__":
    prepare()
