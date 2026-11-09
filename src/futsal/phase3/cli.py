"""Phase 3 CLI: run physical & tactical spatial analytics on a completed Phase 2 directory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import Phase3Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3: Speed, Distance, Voronoi Pitch Control, and Team Convex Hulls"
    )
    parser.add_argument("--phase2-run", type=Path, required=True, help="Completed Phase 2 run directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for Phase 3 results")
    parser.add_argument("--video", type=Path, help="Optional relocated source video path")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Rolling average window size (frames)")
    parser.add_argument("--max-gap", type=float, default=0.25, help="Max gap (seconds) before resetting velocity")
    parser.add_argument("--max-speed", type=float, default=12.0, help="Max physical speed gate (m/s)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output = Phase3Pipeline(
            smoothing_window=args.smoothing_window,
            max_gap_s=args.max_gap,
            max_speed_mps=args.max_speed,
        ).run(args.phase2_run, args.output, args.video)
        logging.info("Completed Phase 3: %s", output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
