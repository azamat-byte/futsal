"""Phase 5: Broadcast Video Compositor, Audio-Preserving H.264 Export & Analytics Dashboard."""

from __future__ import annotations

from .compositor import BroadcastCompositor, find_ffmpeg, mux_h264_with_audio
from .pipeline import Phase5Pipeline, build_match_report, compute_heatmaps

__all__ = [
    "BroadcastCompositor",
    "Phase5Pipeline",
    "build_match_report",
    "compute_heatmaps",
    "find_ffmpeg",
    "mux_h264_with_audio",
]
