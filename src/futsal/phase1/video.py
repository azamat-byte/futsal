"""Constant-frame-rate OpenCV video I/O with explicit ownership and cleanup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture: cv2.VideoCapture | None = None
        self.info: VideoInfo | None = None

    def __enter__(self) -> VideoReader:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.capture = cv2.VideoCapture(str(self.path))
        try:
            if not self.capture.isOpened():
                raise OSError(f"Cannot open video: {self.path}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or min(width, height) < 2:
                raise ValueError("Video has invalid dimensions/FPS; transcode to constant frame rate")
            self.info = VideoInfo(width, height, fps, count)
            return self
        except BaseException:
            self.capture.release()
            raise

    def frames(self, start: int = 0, limit: int | None = None, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        if self.capture is None or self.info is None:
            raise RuntimeError("VideoReader must be opened in a with block")
        if start < 0 or stride < 1 or (limit is not None and limit < 1):
            raise ValueError("Invalid frame start, limit or stride")
        if self.info.frame_count > 0 and start >= self.info.frame_count:
            raise ValueError("Start frame is beyond the end of the video")
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise OSError("Decoder could not seek to the requested start frame")
        index = start
        while limit is None or index < start + limit:
            ok, frame = self.capture.read()
            if not ok:
                if self.info.frame_count > 0 and index < self.info.frame_count - 1:
                    raise OSError(f"Video decode stopped early at frame {index}")
                break
            if frame.shape[:2] != (self.info.height, self.info.width):
                raise ValueError("Video dimensions changed midstream")
            if (index - start) % stride == 0:
                yield index, frame
            index += 1

    def __exit__(self, *args: object) -> None:
        if self.capture is not None:
            self.capture.release()


class VideoWriter:
    def __init__(self, path: Path, info: VideoInfo) -> None:
        self.path, self.info = path, info
        self.writer: cv2.VideoWriter | None = None
        self.count = 0

    def __enter__(self) -> VideoWriter:
        # Codec-safe even dimensions; source-coordinate geometry remains unchanged.
        self.size = (self.info.width + self.info.width % 2, self.info.height + self.info.height % 2)
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.info.fps, self.size)
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError("Could not open MP4 writer; check OpenCV codec support and output path")
        return self

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            raise RuntimeError("VideoWriter must be opened in a with block")
        if frame.shape[:2] != (self.info.height, self.info.width):
            raise ValueError("Output frame dimensions differ from input")
        if self.size != (self.info.width, self.info.height):
            frame = cv2.copyMakeBorder(frame, 0, self.info.height % 2, 0, self.info.width % 2, cv2.BORDER_CONSTANT)
        self.writer.write(frame)
        self.count += 1

    def __exit__(self, *args: object) -> None:
        if self.writer is not None:
            self.writer.release()
