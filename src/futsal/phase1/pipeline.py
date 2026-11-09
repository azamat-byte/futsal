"""Two passes: bounded color calibration, then sequential full-frame-rate tracking."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import pandas as pd

from .annotations import TrackingAnnotator
from .config import Phase1Config
from .detection import Detector, person_detections
from .teams import JerseyColorExtractor, TeamClassifier, TemporalTeamAssigner, calibration_montage
from .tracking import PersonTracker, SceneCutDetector
from .video import VideoInfo, VideoReader, VideoWriter

LOG = logging.getLogger(__name__)
TRACK_COLUMNS = ["frame_index", "timestamp_s", "segment_id", "track_id", "role", "team",
                 "team_vote_fraction", "detection_confidence", "x1", "y1", "x2", "y2",
                 "foot_x_px", "foot_y_px"]


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class Phase1Pipeline:
    """One run per instance. Detector injection supports offline replay and testing."""

    def __init__(self, detector: Detector, config: Phase1Config) -> None:
        self.detector, self.config = detector, config
        self.extractor = JerseyColorExtractor()
        self.classifier = TeamClassifier(config.teams)

    def calibrate(self, video: Path, output: Path, start_frame: int, max_frames: int | None) -> None:
        features, crops = [], []
        cfg = self.config.teams
        span = cfg.calibration_frames * cfg.calibration_stride
        if max_frames is not None:
            span = min(span, max_frames)
        with VideoReader(video) as reader:
            for _, frame in reader.frames(start_frame, limit=span, stride=cfg.calibration_stride):
                people = person_detections(self.detector.detect(frame), self.config.detector.nms_iou)
                for i in range(len(people)):
                    if people.data["role"][i] != "player" or people.confidence[i] < cfg.calibration_confidence:
                        continue
                    crop = self.extractor.crop(frame, people.xyxy[i])
                    if crop is not None:
                        features.append(self.extractor.from_crop(crop))
                        crops.append(cv2.resize(crop, (48, 64)))
                    if len(features) >= cfg.max_samples:
                        break
                if len(features) >= cfg.max_samples:
                    break
        self.classifier.fit(features)
        write_json(output / "teams.json", self.classifier.report())
        montage = calibration_montage(self.classifier, features, crops)
        ok, jpg = cv2.imencode(".jpg", montage)
        if not ok:
            raise OSError("Could not encode calibration review image")
        # imwrite does not reliably handle Unicode paths in every Windows OpenCV build.
        jpg.tofile(str(output / "calibration.jpg"))
        LOG.info("Calibrated %d jersey crops; review %s", len(features), output / "calibration.jpg")

    def run(self, video: Path, output: Path, start_frame: int = 0, max_frames: int | None = None,
            calibration_start: int | None = None, cut_frames: tuple[int, ...] = ()) -> Path:
        if start_frame < 0 or (max_frames is not None and max_frames < 1):
            raise ValueError("start_frame must be >= 0 and max_frames must be > 0")
        if calibration_start is not None and calibration_start < 0:
            raise ValueError("calibration_start must be >= 0")
        if any(frame < 0 for frame in cut_frames):
            raise ValueError("Cut frame indices must be nonnegative")
        video, output = video.resolve(), output.resolve()
        # Validate the input before creating any output. Existing runs are never overwritten.
        with VideoReader(video) as reader:
            assert reader.info is not None
            info = reader.info
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "running", "source": str(video), "video": asdict(info),
            "start_frame": start_frame, "max_frames": max_frames,
            "calibration_start": start_frame if calibration_start is None else calibration_start,
            "manual_cut_frames": list(cut_frames), "config": self.config.to_dict(),
            "versions": {name: version(name) for name in
                         ("numpy", "opencv-python", "pandas", "scikit-learn", "supervision", "requests")},
            "timestamp_basis": "CFR: absolute source frame_index / fps",
            "audio": False, "team_labels_reviewed": False,
        }
        write_json(output / "run.json", manifest)
        try:
            LOG.info("Calibrating shirt colors")
            self.calibrate(video, output, manifest["calibration_start"],
                           max_frames if calibration_start is None else None)
            frames, observations, segments = self._process(video, output, info, start_frame, max_frames, cut_frames)
            manifest.update(status="completed", frames_processed=frames,
                            observations=observations, segments=segments)
        except BaseException as error:
            # Mark partial files as failed, never masquerade as a completed match.
            manifest.update(status="failed", error_type=type(error).__name__)
            write_json(output / "run.json", manifest)
            raise
        write_json(output / "run.json", manifest)
        return output

    def _process(self, video: Path, output: Path, info: VideoInfo, start: int,
                 limit: int | None, cut_frames: tuple[int, ...]) -> tuple[int, int, int]:
        tracker = PersonTracker(self.config.tracking, info.fps)
        cut_detector = SceneCutDetector(self.config.tracking.cut_threshold)
        assigner = TemporalTeamAssigner(self.config.teams, info.fps)
        annotator = TrackingAnnotator()
        manual_cuts = set(cut_frames)
        pending: list[dict] = []
        total = processed = 0
        csv_path = output / "tracks.csv"
        pd.DataFrame(columns=TRACK_COLUMNS).to_csv(csv_path, index=False)
        with VideoReader(video) as reader, VideoWriter(output / "annotated.mp4", info) as writer:
            for index, frame in reader.frames(start, limit):
                changed = cut_detector.update(frame)
                if processed and (changed or index in manual_cuts):
                    tracker.reset_segment()
                    assigner.reset()
                    LOG.info("Identity boundary at frame %d", index)
                people = person_detections(self.detector.detect(frame), self.config.detector.nms_iou)
                tracks = tracker.update(people)  # Empty frames still advance ByteTrack's clock.
                assigner.prune(index)
                teams: list[str] = []
                for i in range(len(tracks)):
                    track_id = int(tracks.tracker_id[i])
                    role = str(tracks.data["role"][i])
                    candidate = "unknown"
                    if role == "player":
                        feature = self.extractor.extract(frame, tracks.xyxy[i])
                        candidate = self.classifier.predict(feature)
                    team, agreement = assigner.update(track_id, candidate, index)
                    if role != "player":
                        # A goalkeeper role is not a team affiliation.
                        team, agreement = "unknown", 0.0
                        assigner.histories.pop(track_id, None)
                    teams.append(team)
                    x1, y1, x2, y2 = map(float, tracks.xyxy[i])
                    pending.append(dict(frame_index=index, timestamp_s=index / info.fps,
                                        segment_id=tracker.segment_id, track_id=track_id, role=role, team=team,
                                        team_vote_fraction=agreement,
                                        detection_confidence=float(tracks.confidence[i]),
                                        x1=x1, y1=y1, x2=x2, y2=y2,
                                        foot_x_px=(x1 + x2) / 2, foot_y_px=y2))
                writer.write(annotator.annotate(frame, tracks, teams, index, tracker.segment_id))
                total += len(tracks)
                processed += 1
                if len(pending) >= 1000:
                    pd.DataFrame(pending, columns=TRACK_COLUMNS).to_csv(csv_path, mode="a", header=False, index=False)
                    pending.clear()
                if processed % 100 == 0:
                    LOG.info("Processed %d frames; %d observations", processed, total)
        if not processed:
            raise ValueError("No frames were decoded in the selected segment")
        if pending:
            pd.DataFrame(pending, columns=TRACK_COLUMNS).to_csv(csv_path, mode="a", header=False, index=False)
        # Reopen the completed output to detect common codec/disk failures.
        with VideoReader(output / "annotated.mp4") as check:
            assert check.info is not None
            if check.info.frame_count != processed or not next(check.frames(limit=1), None):
                raise OSError("Encoded video verification failed")
        return processed, total, tracker.segment_id + 1
