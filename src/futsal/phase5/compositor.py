"""Broadcast video compositor and FFmpeg H.264 + AAC audio muxer."""

from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import supervision as sv

from futsal.phase4.pipeline import PossessionAnnotator

LOG = logging.getLogger(__name__)


def find_ffmpeg() -> str | None:
    """Resolve system ffmpeg or imageio_ffmpeg binary."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def mux_h264_with_audio(
    raw_mp4: Path,
    source_video: Path,
    output_mp4: Path,
    start_frame: int,
    frames_processed: int,
    fps: float,
) -> bool:
    """Transcode raw MP4V video to browser-compatible H.264 (yuv420p) and mux source audio slice."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copy2(raw_mp4, output_mp4)
        return False

    start_s = start_frame / fps
    duration_s = frames_processed / fps
    staging = output_mp4.with_suffix(".staging.mp4")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_mp4),
        "-ss",
        f"{start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(staging),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
        if res.returncode == 0 and staging.is_file() and staging.stat().st_size > 0:
            staging.replace(output_mp4)
            return True
        LOG.warning("FFmpeg H.264 muxing failed: %s", res.stderr[-500:])
    except OSError as exc:
        LOG.warning("FFmpeg execution error: %s", exc)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)

    shutil.copy2(raw_mp4, output_mp4)
    return False


class BroadcastCompositor:
    """Composites player boxes, IDs, speeds, ankle keypoints, ball trails, HUD, and 2D radar."""

    def __init__(self, radar_opacity: float = 0.85, trail_length: int = 20) -> None:
        if not 0.1 <= radar_opacity <= 1.0:
            raise ValueError("radar_opacity must be in [0.1, 1.0]")
        self.radar_opacity = radar_opacity
        self.annotator = PossessionAnnotator(trail_length=trail_length)

    def reset(self) -> None:
        self.annotator.reset_trail()

    def render_frame(
        self,
        frame: np.ndarray,
        tracks: sv.Detections,
        rows: list[dict],
        feet: dict[int, list[list[float]]],
        ball: dict | None,
        possession: dict,
        emitted_events: list[dict],
        radar_img: np.ndarray,
        frame_index: int,
        segment_id: int,
        team_a_control_pct: float | None = None,
        team_a_possession_pct: float | None = None,
    ) -> np.ndarray:
        scene = self.annotator.annotate_phase4(
            frame=frame,
            tracks=tracks,
            rows=rows,
            feet=feet,
            ball=ball,
            possession=possession,
            emitted_events=emitted_events,
            frame_index=frame_index,
            segment_id=segment_id,
        )

        # Draw top-right broadcast telemetry bar (Pitch Control % & Ball Possession %)
        h, w = scene.shape[:2]
        hud_w = min(w - 10, 410)
        if hud_w > 200:
            x0 = w - hud_w - 10
            cv2.rectangle(scene, (x0, 8), (w - 10, 62), (20, 25, 30), -1)
            cv2.rectangle(scene, (x0, 8), (w - 10, 62), (65, 85, 105), 1)
            ctrl_txt = (
                f"Pitch Control  A: {team_a_control_pct:4.1f}% | B: {100.0 - team_a_control_pct:4.1f}%"
                if team_a_control_pct is not None
                else "Pitch Control  -- / --"
            )
            poss_txt = (
                f"Possession     A: {team_a_possession_pct:4.1f}% | B: {100.0 - team_a_possession_pct:4.1f}%"
                if team_a_possession_pct is not None
                else "Possession     -- / --"
            )
            cv2.putText(scene, ctrl_txt, (x0 + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(scene, poss_txt, (x0 + 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 180), 1, cv2.LINE_AA)

        # Alpha-blend the 2D radar mini-map in the bottom-right corner
        ratio = min(0.34 * w / radar_img.shape[1], 0.36 * h / radar_img.shape[0], 1.0)
        rw = max(1, round(radar_img.shape[1] * ratio))
        rh = max(1, round(radar_img.shape[0] * ratio))
        small_radar = cv2.resize(radar_img, (rw, rh), interpolation=cv2.INTER_AREA)
        margin = min(14, max(0, min(w - rw, h - rh) // 2))
        y1, y2 = h - rh - margin, h - margin
        x1, x2 = w - rw - margin, w - margin
        roi = scene[y1:y2, x1:x2]
        cv2.addWeighted(small_radar, self.radar_opacity, roi, 1.0 - self.radar_opacity, 0, roi)
        cv2.rectangle(scene, (x1, y1), (x2 - 1, y2 - 1), (200, 215, 225), 1)
        return scene
