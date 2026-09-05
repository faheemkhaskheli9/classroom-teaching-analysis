"""CLI entry point.

    python -m classroom_analysis.cli extract-frames examples/sample_clip.mp4 \
        --config configs/frame_extraction.yaml

Extracts frames from a video at the configured sample rate and reports how
many were pulled (add --save-dir to also write them as .png files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classroom_analysis.config import ConfigError, load_frame_extraction_config
from classroom_analysis.ingestion import VideoReadError, extract_frames


def _run_extract_frames(video_path: str, config_path: str, save_dir: str | None) -> int:
    try:
        config = load_frame_extraction_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    try:
        count = 0
        for frame in extract_frames(video_path, config):
            count += 1
            if save_dir:
                import cv2

                cv2.imwrite(str(Path(save_dir) / f"frame_{frame.index:05d}.png"), frame.image)
    except VideoReadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"extracted {count} frame(s) from {video_path} at {config.sample_rate_hz} Hz")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classroom_analysis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-frames", help="extract frames from a classroom video")
    p.add_argument("video_path")
    p.add_argument("--config", default="configs/frame_extraction.yaml")
    p.add_argument("--save-dir", default=None, help="directory to write extracted frames as PNGs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract-frames":
        return _run_extract_frames(args.video_path, args.config, args.save_dir)
    return 2


if __name__ == "__main__":
    sys.exit(main())
