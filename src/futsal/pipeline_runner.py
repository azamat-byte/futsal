"""End-to-end one-click futsal pipeline runner.

Executes Phases 1 -> 2 -> 3 -> 4 -> 6 -> 5 seamlessly and outputs a single final broadcast video:
`broadcast_tactical.mp4` with 2D radar, Voronoi control, ball possession trails, and jersey IDs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil

import cv2
import pandas as pd

from futsal.phase1.config import Phase1Config
from futsal.phase1.detection import RoboflowDetector
from futsal.phase1.pipeline import Phase1Pipeline
from futsal.phase1.video import VideoReader
from futsal.phase2.pipeline import Phase2Pipeline
from futsal.phase3.pipeline import Phase3Pipeline
from futsal.phase4.pipeline import Phase4Pipeline
from futsal.phase5.pipeline import Phase5Pipeline
from futsal.player_id_ocr import VLMJerseyOCR
from futsal.reid_boundary import InPlayBoundaryFilter, PlayerIdentityManager
from futsal.roboflow_keypoints import RoboflowKeypointModel

LOG = logging.getLogger("futsal.pipeline_runner")


def run_full_pipeline(
    video_path: Path,
    output_dir: Path,
    max_frames: int | None = None,
    stride_keypoints: int = 30,
    api_key: str | None = None,
    clean_intermediates: bool = False,
) -> Path:
    """Run all phases end-to-end and return the path to the final video."""
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY is required for player and court keypoint inference.")

    p1_dir = output_dir / "p1"
    p2_dir = output_dir / "p2"
    p3_dir = output_dir / "p3"
    p4_dir = output_dir / "p4"
    p5_dir = output_dir / "p5"

    with VideoReader(video_path) as reader:
        total_frames = reader.info.total_frames
        fps = reader.info.fps

    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
    LOG.info(f"Starting 1-click pipeline for {frames_to_process} frames ({frames_to_process / fps:.1f}s) @ {fps} fps")

    # --- Phase 1: RF-DETR Detection + ByteTrack + Team Clustering ---
    if not (p1_dir / "positions.csv").is_file():
        LOG.info("=== Phase 1: Player Detection & Tracking ===")
        shutil.rmtree(p1_dir, ignore_errors=True)
        p1_config = Phase1Config()
        detector = RoboflowDetector(
            config=p1_config.detector,
            api_key=api_key,
        )
        Phase1Pipeline(detector, p1_config).run(
            video=video_path,
            output=p1_dir,
            max_frames=frames_to_process,
        )

    # --- Phase 2: Keypoints Calibration + 2D Radar Projection ---
    cal_file = output_dir / "calibration.json"
    if not cal_file.is_file():
        LOG.info("=== Phase 2: Court Auto-Calibration ===")
        kp_model = RoboflowKeypointModel(api_key=api_key)
        calibration = kp_model.auto_calibrate(
            video_path=video_path,
            start_frame=0,
            max_frames=frames_to_process,
            stride=stride_keypoints,
            min_keypoints=4,
            keypoint_threshold=0.35,
        )
        calibration.save(cal_file)

    if not (p2_dir / "positions.csv").is_file():
        LOG.info("=== Phase 2: 2D Pitch Projection ===")
        shutil.rmtree(p2_dir, ignore_errors=True)
        Phase2Pipeline().run(
            phase1_run=p1_dir,
            calibration=cal_file,
            output=p2_dir,
            video_override=video_path,
        )

    # --- Phase 3: Physical (Speed/Distance) & Spatial (Voronoi) ---
    if not (p3_dir / "metrics.csv").is_file():
        LOG.info("=== Phase 3: Speed, Distance & Pitch Control ===")
        shutil.rmtree(p3_dir, ignore_errors=True)
        Phase3Pipeline().run(
            phase2=p2_dir,
            output=p3_dir,
            video_override=video_path,
        )

    # --- Phase 4: Ball Detection, Ankle Pose & Possession Engine ---
    if not (p4_dir / "possession_tracks.csv").is_file():
        LOG.info("=== Phase 4: Ball, Pose & Possession Analysis ===")
        shutil.rmtree(p4_dir, ignore_errors=True)
        # Use Roboflow detector config for ball
        p1_config = Phase1Config()
        ball_det = RoboflowDetector(
            config=p1_config.detector,
            api_key=api_key,
        )
        Phase4Pipeline(detector=ball_det, pose_estimator=None).run(
            phase2=p2_dir,
            phase3=p3_dir,
            output=p4_dir,
            video_override=video_path,
        )

    # --- Phase 6: VLM Jersey Number OCR + Boundary Filter + Stitched IDs ---
    LOG.info("=== Phase 6: VLM Jersey Number OCR & Re-ID ===")
    tracks_csv = p4_dir / "possession_tracks.csv"
    tracks_df = pd.read_csv(tracks_csv)
    if "jersey_number" not in tracks_df.columns or tracks_df["jersey_number"].dropna().empty:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ocr = VLMJerseyOCR(model_id="HuggingFaceTB/SmolVLM-256M-Instruct", device=device)
        mgr = PlayerIdentityManager(ocr=ocr)
        bfilter = InPlayBoundaryFilter()

        by_frame = {int(f): g.to_dict("records") for f, g in tracks_df.groupby("frame_index")}
        enriched_rows = []
        cap = cv2.VideoCapture(str(video_path))
        for i in range(frames_to_process):
            ok, frame = cap.read()
            if not ok:
                break
            rows = by_frame.get(i)
            if not rows:
                continue
            rows = bfilter.filter_frame_rows(rows)
            enriched_rows.extend(mgr.update_frame(frame, rows, frame_index=i))
            if i % 1500 == 0:
                LOG.info(f"Re-ID progress: {i}/{frames_to_process} frames")
        cap.release()
        pd.DataFrame(enriched_rows).to_csv(tracks_csv, index=False)

    # --- Phase 5: Broadcast Compositing + Final Output Video ---
    final_video = output_dir / "broadcast_tactical.mp4"
    LOG.info("=== Phase 5: Rendering Final Broadcast Tactical Video ===")
    shutil.rmtree(p5_dir, ignore_errors=True)
    Phase5Pipeline().run(
        phase2=p2_dir,
        phase3=p3_dir,
        phase4=p4_dir,
        output=p5_dir,
        video_override=video_path,
    )

    # Copy the final video directly to the output directory root
    produced_video = p5_dir / "broadcast_tactical.mp4"
    if produced_video.is_file():
        shutil.copy2(produced_video, final_video)
        LOG.info(f"SUCCESS: Final product video saved to: {final_video}")

    # Copy telemetry.db to the output root as well
    if (p5_dir / "telemetry.db").is_file():
        shutil.copy2(p5_dir / "telemetry.db", output_dir / "telemetry.db")

    if clean_intermediates:
        LOG.info("Cleaning up intermediate phase directories...")
        for d in (p1_dir, p2_dir, p3_dir, p4_dir, p5_dir):
            shutil.rmtree(d, ignore_errors=True)

    return final_video


def main():
    parser = argparse.ArgumentParser(description="One-Click Futsal Match Analysis Pipeline")
    parser.add_argument("--video", type=Path, required=True, help="Input match video (e.g. 9-min clip)")
    parser.add_argument("--output", type=Path, default=Path("runs/match_output"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Process up to N frames")
    parser.add_argument("--stride-keypoints", type=int, default=30, help="Stride for court keypoints")
    parser.add_argument("--clean", action="store_true", help="Remove intermediate files after completion")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run_full_pipeline(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        stride_keypoints=args.stride_keypoints,
        clean_intermediates=args.clean,
    )
    print(f"\nFinal Video Generated: {out}")


if __name__ == "__main__":
    main()
