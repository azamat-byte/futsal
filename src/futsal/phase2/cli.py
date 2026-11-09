"""Phase 2 CLI: calibration picker followed by offline radar rendering."""

import argparse
import logging
from pathlib import Path

from .picker import pick
from .pipeline import Phase2Pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2: floor homography and futsal radar")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="Click visible floor landmarks and review the projection")
    calibrate.add_argument("--video", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--frame", type=int, default=0)
    render = commands.add_parser("render", help="Reuse a completed Phase 1 run; no API key needed")
    render.add_argument("--phase1-run", type=Path, required=True)
    render.add_argument("--calibration", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--video", type=Path, help="Relocated source video; must match calibration hash")
    render.add_argument("--max-frames", type=int)
    render.add_argument("--start-frame", type=int, help="Absolute source frame inside the completed Phase 1 range")
    render.add_argument("--max-anchor-age", type=float, default=2.0,
                        help="Maximum seconds of optical-flow propagation from an anchor (default: 2)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "calibrate":
            output = pick(args.video, args.output, args.frame)
            if output is None:
                logging.info("Closed without saving a calibration")
                return 0
        else:
            output = Phase2Pipeline().run(args.phase1_run, args.calibration, args.output,
                                          args.max_anchor_age, args.video, args.max_frames, args.start_frame)
        logging.info("Saved: %s", output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    except KeyboardInterrupt:
        return 130
