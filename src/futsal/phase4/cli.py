"""Phase 4 CLI: Ball Possession & Event Engine using Pose Keypoints."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from futsal.match.pose import AnklePoseEstimator, NoPoseEstimator
from futsal.phase1.config import Phase1Config
from futsal.phase1.detection import RoboflowDetector
from .pipeline import Phase4Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4: Ball Possession & Pass/Shot Event Detection using Pose"
    )
    parser.add_argument("--phase2-run", type=Path, required=True, help="Completed Phase 2 run directory")
    parser.add_argument("--phase3-run", type=Path, help="Optional completed Phase 3 run directory for speeds")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for Phase 4")
    parser.add_argument("--config", type=Path, help="Phase 1 JSON config for ball detector")
    parser.add_argument("--pose-weights", type=Path, default=Path("yolo11n-pose.pt"), help="COCO 17-keypoint pose model")
    parser.add_argument("--device", default="cpu", help="Device for local pose inference (e.g. cpu or 0)")
    parser.add_argument("--confirm-seconds", type=float, default=0.12, help="Temporal smoothing dwell time (s)")
    parser.add_argument("--no-pose", action="store_true", help="Disable human pose inference")
    parser.add_argument("--video", type=Path, help="Optional relocated source video")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detector = None
    try:
        cfg = Phase1Config.load(args.config)
        detector = RoboflowDetector(cfg.detector, os.environ.get("ROBOFLOW_API_KEY", ""))
        pose = NoPoseEstimator() if args.no_pose else AnklePoseEstimator(args.pose_weights, device=args.device)
        output = Phase4Pipeline(detector, pose, confirm_s=args.confirm_seconds).run(
            args.phase2_run, args.output, args.phase3_run, args.video
        )
        logging.info("Completed Phase 4: %s", output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
