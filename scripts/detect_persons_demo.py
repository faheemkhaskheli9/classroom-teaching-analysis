"""Demo: run person detection on a classroom clip and save annotated frames.

    PYTHONPATH=src python scripts/detect_persons_demo.py \\
        examples/sample_clip.mp4 \\
        --frame-config configs/frame_extraction.yaml \\
        --detection-config configs/detection_demo.yaml \\
        --save-dir examples/detections

NOTE (mock): the default detection config (configs/detection_demo.yaml) uses
the offline HOG+SVM backend instead of a real YOLO model, so this script
runs fully offline on CPU with no network access and no paid API calls --
see classroom_analysis.detection.HogBackend. Pass
--detection-config configs/detection.yaml (with `pip install -e .[yolo]`)
to run real YOLO inference instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classroom_analysis.config import ConfigError, load_detection_config, load_frame_extraction_config
from classroom_analysis.detection import DetectionError, PersonDetector, draw_boxes
from classroom_analysis.ingestion import VideoReadError, extract_frames


def run(video_path: str, frame_config_path: str, detection_config_path: str, save_dir: str, max_frames: int | None) -> int:
    try:
        frame_config = load_frame_extraction_config(frame_config_path)
        detection_config = load_detection_config(detection_config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        detector = PersonDetector(detection_config)
    except DetectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total_persons = 0
    frames_seen = 0
    try:
        for frame in extract_frames(video_path, frame_config):
            if max_frames is not None and frames_seen >= max_frames:
                break
            boxes = detector.detect(frame.image)
            annotated = draw_boxes(frame.image, boxes)

            import cv2

            cv2.imwrite(str(out_dir / f"frame_{frame.index:05d}.png"), annotated)

            total_persons += len(boxes)
            frames_seen += 1
            print(f"frame {frame.index}: {len(boxes)} person(s)")
    except (VideoReadError, DetectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"processed {frames_seen} frame(s), {total_persons} person detection(s) total -> {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path")
    parser.add_argument("--frame-config", default="configs/frame_extraction.yaml")
    parser.add_argument("--detection-config", default="configs/detection_demo.yaml")
    parser.add_argument("--save-dir", default="examples/detections")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args.video_path, args.frame_config, args.detection_config, args.save_dir, args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
