"""Config-driven frame-extraction settings (never hardcoded, per acceptance criteria)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a frame-extraction config file can't be read or parsed."""


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
    """Load frame-extraction settings from a YAML file.

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

    section = raw.get("frame_extraction", raw)
    try:
        return FrameExtractionConfig(**section)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid frame_extraction config in {p}: {exc}") from exc
