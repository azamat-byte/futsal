"""Execution/coverage diagnostics, never substitutes for labelled accuracy evaluation."""

import json
from collections import Counter
from pathlib import Path

from .storage import load_tracks, read_json, write_json


def empty_observation_message(manifest, has_stored_tracks=False):
    if has_stored_tracks:
        return "No observations in the selected time window. Adjust the Observation window above."
    state = manifest.get("state")
    if state == "running":
        return "Analysis is still running. Refresh progress in Overview to check when observations are available."
    if state in {"failed", "cancelled"}:
        return f"Analysis {state} with no accepted observations. Check the job details in Overview."
    count = manifest.get("detections", 0)
    frames = manifest.get("frames", 0)
    if not count:
        return (f"Analysis finished: processed {frames} frames, but the detector found no supported objects. "
                "No player tracks or ball observations are available for this clip.")
    candidates = "candidate" if count == 1 else "candidates"
    return (f"Analysis finished: processed {frames} frames and found {count} detector {candidates}, "
            "but none passed the tracking or ball filters. Review the video below; "
            "this is a detection/tracking limitation, not a pending result.")


def summarize_run(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    data = load_tracks(run)
    frames = [json.loads(line) for line in (run / "frames.jsonl").read_text("utf-8").splitlines()]
    people = data[data.class_name == "person"]
    counts = people.groupby("tracklet_id").size()
    raw_counts = Counter()
    for line in (run / "detections.jsonl").read_text("utf-8").splitlines():
        raw_counts.update(d["model_class"] for d in json.loads(line)["detections"])
    report = {
        "status": "execution_and_coverage_only",
        "frames": manifest["frames"], "processing_fps": manifest["processing_fps"],
        "processing_mode": "cached_detections_no_inference" if manifest.get("detections_reused_from")
                           else "full_inference",
        "detector_observations_by_class": dict(raw_counts),
        "unique_tracklets": len(counts),
        "median_observations_per_tracklet": float(counts.median()) if len(counts) else None,
        "tracklets_with_at_least_3_observations": int((counts >= 3).sum()),
        "team_assignment_coverage": float(people.team.isin(["team_a", "team_b"]).mean()) if len(people) else 0,
        "accepted_ball_frame_coverage": float(data[data.class_name == "ball"].frame_index.nunique() /
                                               max(len(frames), 1)),
        "calibration_frame_coverage": sum(bool(f["calibration"]["valid"]) for f in frames) / max(len(frames), 1),
        "calibration_reasons": dict(Counter(f["calibration"]["reason"] for f in frames)),
        "id_stability_note": "Tracklet counts do not measure identity correctness or ID switches",
        "accuracy": {key: {"value": None, "reason": "No independent reviewed ground truth provided"}
                     for key in ["detection_precision", "detection_recall", "IDF1", "HOTA",
                                 "team_accuracy", "ball_tracking_accuracy", "position_error_m"]},
    }
    write_json(run / "quality_report.json", report)
    return report
