"""Phase 5 CLI: Render broadcast tactical video and export analytics bundle for the Streamlit dashboard."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import Phase5Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5: Broadcast Video Export (H.264 + Audio) & Analytics Dashboard Bundle"
    )
    parser.add_argument("--phase2-run", type=Path, required=True, help="Completed Phase 2 run directory")
    parser.add_argument("--phase3-run", type=Path, help="Optional completed Phase 3 run directory")
    parser.add_argument("--phase4-run", type=Path, help="Optional completed Phase 4 run directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for Phase 5")
    parser.add_argument("--video", type=Path, help="Optional relocated source video")
    parser.add_argument("--radar-opacity", type=float, default=0.85, help="Radar mini-map blend opacity [0.1..1.0]")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Rolling velocity smoothing window")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        output = Phase5Pipeline(
            radar_opacity=args.radar_opacity,
            smoothing_window=args.smoothing_window,
        ).run(args.phase2_run, args.output, args.phase3_run, args.phase4_run, args.video)
        logging.info("Completed Phase 5: %s", output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
