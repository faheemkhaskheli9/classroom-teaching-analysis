"""Person detection on classroom frames.

Detection is split into a swappable *backend* (the thing that produces raw
boxes + scores for a frame) and shared, config-driven post-processing
(confidence filtering + non-max suppression), so backends stay simple and
the filtering logic is tested once regardless of which model produced the
boxes.

Backends:

- ``YoloBackend`` -- wraps an Ultralytics YOLO model (pretrained weights,
  restricted to the person class per ``DetectionConfig.person_class_id``).
  The ``ultralytics`` package is imported lazily, only when a detection is
  actually requested, so constructing a detector or running the test suite
  never requires it to be installed.
- ``HogBackend`` -- OpenCV's built-in HOG+SVM pedestrian detector. It ships
  with opencv-python (no extra install, no weight download, no network
  access) and runs on CPU, so it is used as an explicit mock/offline
  stand-in for YOLO in the sample demo script and anywhere real YOLO
  weights aren't available.

Both backends implement the same ``raw_detect(frame) -> list[(x1, y1, x2,
y2, confidence)]`` interface, which lets tests inject a fake backend (or a
fake underlying model) instead of depending on real weights or hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from classroom_analysis.config import DetectionConfig


class DetectionError(RuntimeError):
    """Raised when a detection backend can't be loaded or run."""


@dataclass(frozen=True)
class PersonBox:
    """A single detected person: a box in pixel coordinates + confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"degenerate box: ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )
        if self.confidence < 0.0:
            raise ValueError(f"confidence must be >= 0, got {self.confidence}")


RawDetection = tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence


class DetectionBackend(Protocol):
    def raw_detect(self, frame: np.ndarray) -> list[RawDetection]: ...


class YoloBackend:
    """Ultralytics YOLO backend, restricted to the person class.

    The model is loaded lazily (on first ``raw_detect``) so constructing
    this class never touches the filesystem/network. Pass ``model`` to
    inject an already-loaded model (or a fake with a compatible
    ``.predict()``) for testing -- this skips the lazy import entirely.
    """

    def __init__(self, config: DetectionConfig, model=None) -> None:
        self._config = config
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = self._load_model(self._config)
        return self._model

    @staticmethod
    def _load_model(config: DetectionConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectionError(
                "ultralytics is required for the 'yolo' detection backend "
                "(pip install ultralytics, or install the 'yolo' extra); "
                "use backend: hog for a fully offline alternative"
            ) from exc
        try:
            return YOLO(config.model)
        except Exception as exc:  # pragma: no cover - depends on external weights/IO
            raise DetectionError(f"failed to load YOLO model {config.model!r}: {exc}") from exc

    def raw_detect(self, frame: np.ndarray) -> list[RawDetection]:
        model = self._get_model()
        # conf/iou are wide open here: our own confidence filter + NMS
        # (shared with every backend) does the real thresholding below, so
        # behavior is identical regardless of which backend produced the
        # raw boxes.
        results = model.predict(
            frame,
            conf=0.001,
            iou=1.0,
            classes=[self._config.person_class_id],
            device=self._config.device,
            verbose=False,
        )
        raw: list[RawDetection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in np.asarray(box.xyxy[0]).tolist())
                confidence = float(np.asarray(box.conf[0]))
                raw.append((x1, y1, x2, y2, confidence))
        return raw


class HogBackend:
    """OpenCV HOG+SVM pedestrian detector -- offline, CPU-only, no download.

    Used as an explicit mock/placeholder for YOLO where real pretrained
    weights or network access aren't available (see module docstring).
    """

    def __init__(self, config: DetectionConfig, hog=None) -> None:
        self._config = config
        self._hog = hog

    def _get_hog(self):
        if self._hog is None:
            import cv2

            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
        return self._hog

    def raw_detect(self, frame: np.ndarray) -> list[RawDetection]:
        hog = self._get_hog()
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        raw: list[RawDetection] = []
        for (x, y, w, h), weight in zip(boxes, weights):
            # HOG's SVM decision-function weight is an unbounded score
            # (commonly ~0-3 for real detections), not a probability. Squash
            # it into an approximate [0, 1] confidence so it's comparable to
            # the yolo backend's scores and to DetectionConfig.confidence_threshold.
            confidence = max(0.0, min(1.0, float(weight) / 3.0))
            raw.append((float(x), float(y), float(x + w), float(y + h), confidence))
        return raw


def build_backend(config: DetectionConfig) -> DetectionBackend:
    if config.backend == "yolo":
        return YoloBackend(config)
    if config.backend == "hog":
        return HogBackend(config)
    raise DetectionError(f"unknown detection backend {config.backend!r}")  # pragma: no cover


def apply_confidence_and_nms(
    raw_boxes: Sequence[RawDetection], config: DetectionConfig
) -> list[PersonBox]:
    """Filter ``raw_boxes`` by confidence and run non-max suppression.

    Shared across every backend so "confidence threshold and NMS settings
    are configurable" means the same thing regardless of which model
    produced the raw detections.
    """
    import cv2

    kept = [box for box in raw_boxes if box[4] >= config.confidence_threshold]
    if not kept:
        return []

    rects = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2, _) in kept]
    scores = [confidence for (*_, confidence) in kept]
    indices = cv2.dnn.NMSBoxes(
        rects, scores, score_threshold=config.confidence_threshold, nms_threshold=config.nms_iou_threshold
    )
    if len(indices) == 0:
        return []
    return [PersonBox(*kept[int(i)]) for i in np.asarray(indices).reshape(-1)]


class PersonDetector:
    """Config-driven person detector: picks a backend, applies shared post-processing."""

    def __init__(self, config: DetectionConfig, backend: DetectionBackend | None = None) -> None:
        self._config = config
        self._backend = backend if backend is not None else build_backend(config)

    def detect(self, frame: np.ndarray) -> list[PersonBox]:
        raw = self._backend.raw_detect(frame)
        return apply_confidence_and_nms(raw, self._config)


def draw_boxes(frame: np.ndarray, boxes: Sequence[PersonBox]) -> np.ndarray:
    """Return a copy of ``frame`` annotated with ``boxes`` for visual sanity-checking."""
    import cv2

    annotated = frame.copy()
    for box in boxes:
        p1 = (int(round(box.x1)), int(round(box.y1)))
        p2 = (int(round(box.x2)), int(round(box.y2)))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        label = f"person {box.confidence:.2f}"
        cv2.putText(
            annotated, label, (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    return annotated
