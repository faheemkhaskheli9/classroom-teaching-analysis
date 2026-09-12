"""Config-driven settings for the pipeline (never hardcoded, per acceptance criteria)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a config file can't be read or parsed."""


def _load_yaml_section(path: str | Path, section_key: str) -> dict:
    """Read ``path`` as YAML and return the ``section_key`` mapping (or the
    whole document if it has no nested section by that name).

    The path is always caller-supplied, so a missing file is a hard error
    rather than a silent fall-through to defaults.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {p}")
    if not isinstance(raw, dict):
        raise ConfigError(f"expected a mapping at the top level of {p}, got {type(raw).__name__}")

    return raw.get(section_key, raw)


@dataclass(frozen=True)
class FrameExtractionConfig:
    sample_rate_hz: float = 1.0
    resize_width: int | None = None
    resize_height: int | None = None
    output_format: str = "bgr24"  # or "rgb24"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {self.sample_rate_hz}")
        if (self.resize_width is None) != (self.resize_height is None):
            raise ValueError("resize_width and resize_height must both be set, or neither")
        if self.output_format not in ("bgr24", "rgb24"):
            raise ValueError(f"unsupported output_format {self.output_format!r}")


def load_frame_extraction_config(path: str | Path) -> FrameExtractionConfig:
    """Load frame-extraction settings from a YAML file."""
    section = _load_yaml_section(path, "frame_extraction")
    try:
        return FrameExtractionConfig(**section)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid frame_extraction config in {path}: {exc}") from exc


@dataclass(frozen=True)
class DetectionConfig:
    """Config-driven settings for person detection (issue: YOLO-based person detection).

    ``backend`` selects the detector implementation behind the same
    interface: "yolo" loads a real Ultralytics YOLO model from ``model``
    (a path or a known model name such as ``yolov8n.pt``); "hog" uses
    OpenCV's built-in, fully offline HOG+SVM pedestrian detector and ignores
    ``model``/``device`` -- it exists as a CPU-only, no-network-download
    stand-in for environments/demos where fetching real YOLO weights isn't
    available.
    """

    backend: str = "yolo"  # or "hog"
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.45
    person_class_id: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.backend not in ("yolo", "hog"):
            raise ValueError(f"unsupported detection backend {self.backend!r}")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be within [0, 1], got {self.confidence_threshold}"
            )
        if not (0.0 <= self.nms_iou_threshold <= 1.0):
            raise ValueError(f"nms_iou_threshold must be within [0, 1], got {self.nms_iou_threshold}")


def load_detection_config(path: str | Path) -> DetectionConfig:
    """Load person-detection settings from a YAML file."""
    section = _load_yaml_section(path, "detection")
    try:
        return DetectionConfig(**section)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid detection config in {path}: {exc}") from exc
