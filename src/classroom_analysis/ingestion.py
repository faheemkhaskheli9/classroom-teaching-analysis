"""Video ingestion: extract frames from a classroom video at a configured rate.

Frames are yielded one at a time (a generator over ``cv2.VideoCapture.read()``)
so the whole video is never loaded into memory at once, however long it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from classroom_analysis.config import FrameExtractionConfig


class VideoReadError(RuntimeError):
    """Raised when a video file can't be opened or read."""


@dataclass(frozen=True)
class ExtractedFrame:
    index: int  # sequence number among *extracted* frames, not source frames
    timestamp_s: float
    image: np.ndarray  # HxWx3, channel order per config.output_format


def extract_frames(
    video_path: str | Path, config: FrameExtractionConfig
) -> Iterator[ExtractedFrame]:
    """Yield frames from ``video_path`` sampled at ``config.sample_rate_hz``.

    Raises :class:`VideoReadError` if the file can't be opened at all. A video
    that opens but reports zero/unknown FPS falls back to sampling every frame
    -- still correct, just not rate-limited -- rather than crashing or
    silently yielding nothing.
    """
    import cv2

    path = Path(video_path)
    if not path.is_file():
        raise VideoReadError(f"video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"could not open video (unreadable/unsupported format): {path}")

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = config.sample_rate_hz  # sample every frame if FPS is unknown
        frame_stride = max(1, round(source_fps / config.sample_rate_hz))

        source_index = 0
        extracted_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % frame_stride == 0:
                if config.resize_width is not None:
                    frame = cv2.resize(frame, (config.resize_width, config.resize_height))
                if config.output_format == "rgb24":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield ExtractedFrame(
                    index=extracted_index,
                    timestamp_s=source_index / source_fps,
                    image=frame,
                )
                extracted_index += 1
            source_index += 1
    finally:
        capture.release()
