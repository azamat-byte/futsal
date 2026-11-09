import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import RunConfig
from .geometry import CalibrationSpec, DynamicCalibration, project
from .models import Detector, suppress_duplicate_people
from .rendering import annotate
from .radar import RadarRenderer, inset_radar
from .storage import TrackWriter, environment, read_json, sha256, utcnow, write_json
from .teams import TeamClassifier, jersey_feature
from .tracking import BallTracker, PlayerTracker
from .video import VideoWriter, frames, probe, save_image


class Cancelled(Exception):
    pass


def cut_feature(image):
    small = cv2.resize(image, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def run_pipeline(source, directory, config: RunConfig, progress=None, cancelled=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=directory / "pipeline.log", level=logging.INFO, encoding="utf-8")
    started = time.perf_counter()
    meta = probe(source)
    fps = min(config.sample_fps, meta["average_fps"] or config.sample_fps)
    detector = Detector(config.model, config.device, config.imgsz, config.confidence)
    if hasattr(detector, "configure"):
        detector.configure(config)
    ball_detector = (
        Detector(config.ball_model, config.device, 640, config.confidence) if config.ball_model else None
    )
    if ball_detector:
        ball_detector.configure(config.model_copy(update={"inference_mode": "full"}))
    anchors = (
        [CalibrationSpec.model_validate(a) for a in read_json(config.calibration_file, [])]
        if config.calibration_file
        else []
    )
    for anchor in anchors:
        if anchor.length != config.court_length or anchor.width != config.court_width:
            raise ValueError("Calibration court dimensions differ from run configuration")
    source_digest = sha256(source)
    if getattr(detector, "roi", None) and detector.roi.get("source_sha256") != source_digest:
        raise ValueError("Court ROI must be bound to this video's source_sha256")
    if any(a.source_sha256 and a.source_sha256 != source_digest for a in anchors):
        raise ValueError("Calibration file belongs to a different source video")
    cache = None
    if config.reuse_detections_from:
        prior = Path(config.reuse_detections_from)
        cached = read_json(prior / "manifest.json")
        if not cached or cached.get("state") != "completed":
            raise ValueError("Detection cache must be a completed run")
        if cached["source_sha256"] != source_digest or cached["model"]["sha256"] != detector.manifest["sha256"]:
            raise ValueError("Detection cache source or checkpoint differs from this run")
        cached_config = RunConfig.model_validate(cached["config"])
        for key in ("start", "duration", "sample_fps", "imgsz", "confidence", "ball_model", "tile_ball",
                    "inference_mode", "inference_tile_size", "inference_tile_overlap", "enhance_contrast",
                    "person_confidence", "ball_confidence", "ball_tile_imgsz", "court_roi_file"):
            if cached_config.model_dump()[key] != config.model_dump()[key]:
                raise ValueError(f"Detection cache configuration differs: {key}")
        if getattr(detector, "roi", None) != cached["model"].get("inference", {}).get("court_roi"):
            raise ValueError("Detection cache court ROI contents differ")
        cache = (prior / "detections.jsonl").open(encoding="utf-8")
    calibration = DynamicCalibration(anchors, config.calibration_max_age)
    court_detector = None
    if config.court_model:
        if not config.court_schema:
            raise ValueError("Court keypoint model requires semantic court schema")
        from .court import CourtLandmarkDetector

        court_detector = CourtLandmarkDetector(config.court_model, config.court_schema, config.device)
    cut_times = read_json(config.cuts_file, []) if config.cuts_file else []
    cut_times = sorted(float(t) for t in cut_times if float(t) > config.start)
    next_cut = 0
    team = TeamClassifier()
    overrides = read_json(config.team_overrides_file, {}) if config.team_overrides_file else {}
    if any(label not in {"team_a", "team_b", "unknown"} for label in overrides.values()):
        raise ValueError("Manual team overrides must map tracklet IDs to team_a, team_b, or unknown")
    encoder = None
    if config.appearance_reid:
        from .reid import OSNetEncoder

        device = "cuda:" + config.device if config.device.isdigit() else config.device
        encoder = OSNetEncoder(config.reid_checkpoint, device)

    def new_tracker():
        return PlayerTracker(config.tracker, fps, config, encoder)

    def new_ball_tracker():
        return BallTracker(config.ball_max_gap, config.ball_confirmation_frames, config.ball_start_confidence)

    tracker = new_tracker()
    ball = new_ball_tracker()
    manifest = {
        "schema_version": 1,
        "created": utcnow(),
        "state": "running",
        "asset": meta,
        "source_sha256": source_digest,
        "config": config.model_dump(),
        "effective_sample_fps": fps,
        "model": detector.manifest,
        "ball_model": ball_detector.manifest if ball_detector else None,
        "environment": environment(),
        "detections_reused_from": config.reuse_detections_from,
        "people_postprocessing": "role-agnostic near-duplicate suppression at IoU 0.8; balls separate",
        "tracking": {"mode": config.tracker, "appearance_reid": config.appearance_reid,
                     "reid_sha256": sha256(config.reid_checkpoint) if config.appearance_reid else None,
                     "display_smoothing": config.smooth_boxes, "coordinate_source": "raw_detector_measurements",
                     "missing_positions": "not_interpolated"},
        "warnings": (["Detector person confidence floor exceeds tracking low threshold; low-score recovery is truncated"]
                     if (config.person_confidence if config.person_confidence is not None else config.confidence)
                     > config.track_low_threshold else []),
        "frame_index_semantics": "source_presentation_timestamp_ticks",
        "proxy_timing": "constant sampled cadence; use frames.jsonl to map proxy_index to source PTS",
        "limitations": [
            "team clusters are unverified",
            "segments default to unknown until reviewed",
            "cross-cut IDs are separate",
            "physical accuracy has not been evaluated",
        ],
    }
    write_json(directory / "manifest.json", manifest)
    writer = TrackWriter(directory / "tracks")
    video = None
    radar_video = overlay_video = None
    radar = RadarRenderer(config.court_length, config.court_width, config.team_colors,
                          config.show_ids, config.show_trails) if config.radar else None
    shots = []
    previous_hist = previous_gray = None
    shot = 0
    count = 0
    detections_total = 0
    errors = []
    end = config.start + (
        config.duration if config.duration else max(0, (meta["duration_s"] or 0) - config.start)
    )
    frame_log = (directory / "frames.jsonl").open("w", encoding="utf-8")
    try:
        for frame in frames(source, config.start, config.duration, fps):
            if cancelled and cancelled():
                raise Cancelled()
            image = frame.image
            hist, gray = cut_feature(image)
            cut = False
            if previous_hist is not None:
                hist_change = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                delta = float(np.mean(cv2.absdiff(previous_gray, gray))) / 255
                cut = (hist_change > config.cut_threshold or delta > 0.33
                       or frame.dt > max(.5, config.track_buffer_seconds))
            if next_cut < len(cut_times) and frame.time >= cut_times[next_cut]:
                cut = True
                while next_cut < len(cut_times) and frame.time >= cut_times[next_cut]:
                    next_cut += 1
            if cut or count == 0:
                if shots:
                    shots[-1]["end_s"] = frame.time
                if count:
                    shot += 1
                    tracker = new_tracker()
                    ball = new_ball_tracker()
                shots.append(
                    {
                        "shot_id": shot,
                        "start_s": frame.time,
                        "end_s": frame.time,
                        "label": "unknown",
                        "boundary_status": "automatic_proposal",
                    }
                )
                save_image(directory / "shots" / f"{shot:05d}.jpg", cv2.resize(image, (640, 360)))
            previous_hist, previous_gray = hist, gray
            if cache:
                cached_frame = json.loads(next(cache))
                if cached_frame["frame_index"] != frame.index or abs(cached_frame["time_s"] - frame.time) > 1e-5:
                    raise ValueError("Detection cache timestamp mismatch")
                found = cached_frame["detections"]
            else:
                found = detector.predict(image)
            if ball_detector and not cache:
                found = [d for d in found if d["class_name"] != "ball"] + ball_detector.predict_ball(
                    image, config.tile_ball
                )
            elif config.tile_ball and not cache:
                found = [d for d in found if d["class_name"] != "ball"] + detector.predict_ball(
                    image, tiled=True, base_detections=found)
            found = suppress_duplicate_people(found)
            tracked = tracker.update(found, image, frame.time)
            accepted_ball = ball.update(
                [d for d in found if d["class_name"] == "ball"], frame.time, image.shape
            )
            anchor = court_detector.propose(image, frame.time) if court_detector else None
            matrix, info = calibration.update(image, frame.time, [d["box"] for d in found], cut, anchor)
            info = {**info, "scale_verified": bool(info["scale_verified"] and config.scale_verified)}
            rows = []
            for item in tracked + ([accepted_ball] if accepted_ball else []):
                is_person = item["class_name"] == "person"
                tracklet = f"{directory.parent.name}:{directory.name}:s{shot}:p{item['local_id']}" if is_person else None
                feature = jersey_feature(image, item["box"]) if is_person else None
                label, team_score = (
                    team.predict(feature, tracklet)
                    if (
                        is_person
                        and config.team_mode == "color"
                        and item["role"] not in {"referee", "goalkeeper"}
                    )
                    else ("unknown", None)
                )
                x1, y1, x2, y2 = item["box"]
                label = overrides.get(tracklet, label)
                cx = cy = None
                valid = bool(info["valid"] and is_person and y2 < image.shape[0] - 2
                             and x1 > 1 and x2 < image.shape[1] - 2)
                if valid:
                    try:
                        point = project(matrix, [[(x1 + x2) / 2, y2]])[0]
                        if 0 <= point[0] <= config.court_length and 0 <= point[1] <= config.court_width:
                            cx, cy = map(float, point)
                        else:
                            valid = False
                    except ValueError:
                        valid = False
                rows.append(
                    {
                        "frame_index": frame.index,
                        "time_s": frame.time,
                        "dt_s": frame.dt,
                        "shot_id": shot,
                        "tracklet_id": tracklet,
                        "detection_id": f"{frame.index}:{len(rows)}",
                        "class_name": item["class_name"],
                        "role": item["role"],
                        "score": item["score"],
                        "display_box": item.get("display_box"),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "team": label,
                        "team_score": team_score,
                        "observed": True,
                        "court_x": cx,
                        "court_y": cy,
                        "calibration_valid": valid,
                        "scale_verified": bool(info["scale_verified"] and config.scale_verified),
                        "calibration_error_px": info["error_px"],
                    }
                )
            writer.add(rows)
            # Preserve all detector outputs separately, including low-score/untracked proposals.
            with (directory / "detections.jsonl").open("a", encoding="utf-8") as raw:
                raw.write(
                    json.dumps({"frame_index": frame.index, "time_s": frame.time, "detections": found}) + "\n"
                )
            frame_log.write(
                json.dumps(
                    {
                        "proxy_index": count,
                        "frame_index": frame.index,
                        "time_s": frame.time,
                        "dt_s": frame.dt,
                        "shot_id": shot,
                        "calibration": info,
                        "image_to_court": matrix.tolist() if matrix is not None and info["valid"] else None,
                    }
                )
                + "\n"
            )
            if config.render:
                out = annotate(image, rows, frame.time, shot, info, config.show_boxes,
                               config.show_ids, config.team_colors)
                if config.show_court_lines:
                    from .court import line_candidates

                    for line in line_candidates(image):
                        cv2.line(out, tuple(line[:2]), tuple(line[2:]), (110, 210, 230), 1)
                target_width = min(1280, out.shape[1])
                target_height = round(out.shape[0] * target_width / out.shape[1] / 2) * 2
                out = cv2.resize(out, (target_width, target_height))
                if video is None:
                    video = VideoWriter(directory / "annotated.mp4", fps, (target_width, target_height))
                video.write(out)
            if radar:
                radar_frame = radar.render(rows, frame.time, shot, info)
                if radar_video is None:
                    radar_video = VideoWriter(directory / "radar.mp4", fps,
                                              (radar_frame.shape[1], radar_frame.shape[0]))
                radar_video.write(radar_frame)
                if config.radar_overlay:
                    base = out if config.render else image
                    combined = inset_radar(base, radar_frame)
                    # H264 requires even dimensions, including original sources.
                    ch, cw = combined.shape[:2]
                    combined = cv2.resize(combined, (cw // 2 * 2, ch // 2 * 2))
                    if overlay_video is None:
                        overlay_video = VideoWriter(directory / "tactical.mp4", fps,
                                                    (combined.shape[1], combined.shape[0]))
                    overlay_video.write(combined)
            shots[-1]["end_s"] = min(end, frame.time + 1 / fps)
            count += 1
            detections_total += len(found)
            progress_interval = 1 if config.tile_ball or config.inference_mode != "full" else 20
            if count % progress_interval == 0:
                if progress:
                    progress(count, min(1, (frame.time - config.start) / max(end - config.start, 1)))
                frame_log.flush()
                if count % 20 == 0:
                    writer.flush()
                manifest.update(frames=count, detections=detections_total)
                write_json(directory / "manifest.json", manifest)
                write_json(directory / "shots.json", shots)
        if not count:
            raise ValueError("No decoded frames in the selected time interval")
        manifest["state"] = "completed"
    except Cancelled:
        manifest["state"] = "cancelled"
    except Exception as exc:
        errors.append({"type": type(exc).__name__, "message": str(exc)})
        manifest["state"] = "failed"
        logging.exception("Pipeline failed")
        raise
    finally:
        writer.flush()
        if cache:
            cache.close()
        frame_log.close()
        if video:
            video.close()
        if radar_video:
            radar_video.close()
        if overlay_video:
            overlay_video.close()
        manifest.update(
            frames=count,
            detections=detections_total,
            shots=len(shots),
            wall_time_s=time.perf_counter() - started,
            errors=errors,
            finished=utcnow(),
        )
        manifest["processing_fps"] = count / max(manifest["wall_time_s"], 1e-6)
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "shots.json", shots)
        if team.model is not None:
            write_json(
                directory / "team_prototypes.json",
                {
                    "centers_lab": team.model.cluster_centers_.tolist(),
                    "status": "unverified_match_specific_clusters",
                },
            )
    return manifest
