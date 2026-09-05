from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_clip(tmp_path: Path) -> Path:
    """A short synthetic 10 fps, 20-frame, 64x48 clip -- no real footage needed."""
    path = tmp_path / "sample_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for i in range(20):
        frame = np.full((48, 64, 3), (i * 10) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
