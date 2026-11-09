"""Run `python -m futsal.phase1 --help` for the standalone Phase 1 interface."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import Phase1Config
from .detection import RoboflowDetector
from .pipeline import Phase1Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1: RF-DETR + ByteTrack + HSV team colors")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing runs are preserved")
    parser.add_argument("--config", type=Path, help="JSON settings; defaults use your RF-DETR model")
    parser.add_argument("--start-frame", type=int, default=0, help="Absolute zero-based source frame")
    parser.add_argument("--max-frames", type=int, help="Process at most this many consecutive source frames")
    parser.add_argument("--calibration-start", type=int, help="Separate calibration segment, absolute source frame")
    parser.add_argument("--cut-frame", action="append", type=int, default=[], help="Manual cut boundary; repeatable")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detector = None
    try:
        config = Phase1Config.load(args.config)
        detector = RoboflowDetector(config.detector, os.environ.get("ROBOFLOW_API_KEY", ""))
        output = Phase1Pipeline(detector, config).run(
            args.video, args.output, args.start_frame, args.max_frames,
            args.calibration_start, tuple(args.cut_frame),
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logging.error("Interrupted; any partially written run is marked failed")
        return 130
    finally:
        if detector is not None:
            detector.close()
    logging.info("Completed: %s", output)
    return 0
