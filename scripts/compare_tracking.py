"""Same-frame, same-weight tracker comparison; counts are diagnostics, not identity accuracy."""

import argparse
from pathlib import Path

from futsal.config import RunConfig
from futsal.pipeline import run_pipeline
from futsal.quality import summarize_run
from futsal.storage import write_json


def compare(source, output, model="best.pt", start=0, duration=4, fps=20, device="0", appearance=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    common = dict(model=model, device=device, imgsz=1280, confidence=.05, person_confidence=.05,
                  ball_confidence=.05, start=start, duration=duration, sample_fps=fps, radar=False,
                  tracker="bytetrack", track_low_threshold=.05)
    report = {"source": str(Path(source).resolve()), "method": "same_frames_and_cached_detector_outputs",
              "accuracy_status": "No identity/event ground truth. Tracklet counts and coverage are not IDF1/HOTA.",
              "runs": {}}
    variants = [("baseline", {}),
                ("motion", {"tracker": "botsort", "smooth_boxes": True,
                            "track_new_threshold": .35, "track_buffer_seconds": 1.5,
                            "track_min_observations": 2,
                            "ball_confirmation_frames": 2, "ball_start_confidence": .2,
                            "ball_max_gap": .25})]
    if appearance:
        variants.append(("appearance", {**variants[-1][1], "appearance_reid": True}))
    for name, changes in variants:
        if name != "baseline":
            changes["reuse_detections_from"] = str(output / "baseline")
        config = RunConfig(**(common | changes))
        run = output / name
        run_pipeline(source, run, config, progress=lambda n, p: print(name, n, f"{p:.0%}", flush=True))
        report["runs"][name] = summarize_run(run)
        write_json(output / "comparison.json", report)
        print(name, report["runs"][name], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--appearance", action="store_true")
    args = parser.parse_args()
    compare(args.source, args.output, args.model, args.start, args.duration, args.fps, args.device, args.appearance)
