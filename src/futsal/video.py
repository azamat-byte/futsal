from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import av
import cv2
import imageio_ffmpeg
import numpy as np


@dataclass
class Frame:
    index: int
    time: float
    image: np.ndarray
    dt: float


def probe(path):
    path = Path(path).resolve()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "duration_s": float(container.duration / av.time_base) if container.duration else None,
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "average_fps": float(stream.average_rate) if stream.average_rate else None,
            "base_rate": str(stream.base_rate),
            "time_base": str(stream.time_base),
            "stream_start_pts": stream.start_time,
            "audio": bool(container.streams.audio),
        }


def frames(path, start=0, duration=None, fps=10):
    """Stream presentation timestamps relative to stream start; never use nominal FPS for time."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        if start > 0:
            container.seek(int((start + origin) / stream.time_base), stream=stream, backward=True)
        due = start
        previous = None
        decoded = 0
        for frame in container.decode(stream):
            decoded += 1
            if frame.pts is None:
                raise ValueError("Video frame has no presentation timestamp; create a timed proxy first")
            time = float(frame.pts * stream.time_base) - origin
            if time + 1e-7 < start:
                continue
            if duration is not None and time >= start + duration:
                break
            if time + 1e-7 < due:
                continue
            dt = 1 / fps if previous is None else time - previous
            if dt <= 0:
                continue
            # index is a stable PTS tick, not a fabricated source-frame number after seeking.
            yield Frame(int(frame.pts), time, frame.to_ndarray(format="bgr24"), dt)
            previous = time
            due = start + (int((time - start) * fps + 1e-7) + 1) / fps


def frame_at(path, time):
    try:
        return next(frames(path, start=max(0, time), duration=2, fps=1)).image
    except StopIteration as exc:
        raise ValueError(f"No video frame at {time:.2f} seconds. Choose a timestamp within this video.") from exc


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def load_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


class VideoWriter:
    """H264 analysis proxy. Constant sampled cadence, with source PTS in frame table."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.generator = imageio_ffmpeg.write_frames(
            str(path),
            size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=["-crf", "23", "-preset", "veryfast"],
        )
        self.generator.send(None)

    def write(self, image):
        self.generator.send(np.ascontiguousarray(image[:, :, ::-1]))

    def close(self):
        self.generator.close()


def clip(source, target, start, duration):
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
        capture_output=True,
    )


def ffmpeg_path():
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
