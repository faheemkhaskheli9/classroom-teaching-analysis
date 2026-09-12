import sys

import numpy as np
import pytest

from classroom_analysis.config import ConfigError, DetectionConfig, load_detection_config
from classroom_analysis.detection import (
    DetectionError,
    HogBackend,
    PersonBox,
    PersonDetector,
    YoloBackend,
    apply_confidence_and_nms,
    build_backend,
    draw_boxes,
)


class FakeBackend:
    def __init__(self, raw):
        self._raw = raw

    def raw_detect(self, frame):
        return self._raw


# -- DetectionConfig / load_detection_config -----------------------------------


def test_loads_nested_detection_section(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "detection:\n"
        "  backend: hog\n"
        "  confidence_threshold: 0.4\n"
        "  nms_iou_threshold: 0.3\n"
    )
    config = load_detection_config(path)
    assert config.backend == "hog"
    assert config.confidence_threshold == 0.4
    assert config.nms_iou_threshold == 0.3


def test_missing_detection_config_file_is_a_hard_error():
    with pytest.raises(ConfigError):
        load_detection_config("configs/does_not_exist.yaml")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        DetectionConfig(backend="not-a-real-backend")


@pytest.mark.parametrize("field", ["confidence_threshold", "nms_iou_threshold"])
def test_thresholds_out_of_range_are_rejected(field):
    with pytest.raises(ValueError):
        DetectionConfig(**{field: 1.5})
    with pytest.raises(ValueError):
        DetectionConfig(**{field: -0.1})


def test_unknown_field_is_a_hard_error(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("detection:\n  not_a_real_field: 1\n")
    with pytest.raises(ConfigError):
        load_detection_config(path)


# -- PersonBox ------------------------------------------------------------------


def test_person_box_rejects_degenerate_coordinates():
    with pytest.raises(ValueError):
        PersonBox(x1=10, y1=10, x2=10, y2=20, confidence=0.9)


def test_person_box_rejects_negative_confidence():
    with pytest.raises(ValueError):
        PersonBox(x1=0, y1=0, x2=10, y2=10, confidence=-0.1)


# -- apply_confidence_and_nms ----------------------------------------------------


def test_confidence_threshold_filters_low_scores():
    config = DetectionConfig(backend="hog", confidence_threshold=0.5, nms_iou_threshold=0.9)
    raw = [(0, 0, 10, 10, 0.9), (20, 20, 30, 30, 0.2)]
    boxes = apply_confidence_and_nms(raw, config)
    assert len(boxes) == 1
    assert boxes[0].confidence == 0.9


def test_nms_keeps_only_highest_confidence_of_overlapping_boxes():
    config = DetectionConfig(backend="hog", confidence_threshold=0.1, nms_iou_threshold=0.3)
    raw = [
        (0, 0, 10, 10, 0.9),
        (1, 1, 11, 11, 0.6),  # heavily overlaps the box above -> suppressed
        (50, 50, 60, 60, 0.7),  # far away -> kept independently
    ]
    boxes = apply_confidence_and_nms(raw, config)
    confidences = sorted(b.confidence for b in boxes)
    assert confidences == [0.7, 0.9]


def test_no_detections_above_threshold_returns_empty_list():
    config = DetectionConfig(backend="hog", confidence_threshold=0.9)
    raw = [(0, 0, 10, 10, 0.1)]
    assert apply_confidence_and_nms(raw, config) == []


# -- PersonDetector / build_backend ----------------------------------------------


def test_build_backend_selects_yolo_and_hog():
    assert isinstance(build_backend(DetectionConfig(backend="yolo")), YoloBackend)
    assert isinstance(build_backend(DetectionConfig(backend="hog")), HogBackend)


def test_person_detector_uses_injected_backend_end_to_end():
    config = DetectionConfig(backend="hog", confidence_threshold=0.5, nms_iou_threshold=0.9)
    detector = PersonDetector(config, backend=FakeBackend([(0, 0, 10, 10, 0.8)]))
    boxes = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))
    assert boxes == [PersonBox(x1=0, y1=0, x2=10, y2=10, confidence=0.8)]


# -- YoloBackend ------------------------------------------------------------------


class _FakeBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.array(conf, dtype=np.float32)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYoloModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.predict_calls = []

    def predict(self, frame, **kwargs):
        self.predict_calls.append(kwargs)
        return [_FakeResult(self._boxes)]


def test_yolo_backend_extracts_boxes_from_injected_model():
    model = _FakeYoloModel([_FakeBox([1.0, 2.0, 30.0, 40.0], 0.77)])
    backend = YoloBackend(DetectionConfig(backend="yolo"), model=model)
    raw = backend.raw_detect(np.zeros((50, 50, 3), dtype=np.uint8))
    assert raw == [(1.0, 2.0, 30.0, 40.0, pytest.approx(0.77, abs=1e-5))]
    # confidence/NMS filtering is done by apply_confidence_and_nms, not the
    # backend itself, so the backend must ask the model for a wide-open set.
    assert model.predict_calls[0]["conf"] < 0.01
    assert model.predict_calls[0]["iou"] == 1.0
    assert model.predict_calls[0]["classes"] == [0]


def test_yolo_backend_raises_detection_error_when_ultralytics_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    backend = YoloBackend(DetectionConfig(backend="yolo"))
    with pytest.raises(DetectionError, match="ultralytics"):
        backend.raw_detect(np.zeros((10, 10, 3), dtype=np.uint8))


# -- HogBackend / draw_boxes ------------------------------------------------------


class _FakeHog:
    def detectMultiScale(self, frame, **kwargs):
        boxes = np.array([[5, 5, 10, 20]])
        weights = np.array([1.5])
        return boxes, weights


def test_hog_backend_normalizes_weight_to_confidence_range():
    backend = HogBackend(DetectionConfig(backend="hog"), hog=_FakeHog())
    raw = backend.raw_detect(np.zeros((30, 30, 3), dtype=np.uint8))
    assert raw == [(5.0, 5.0, 15.0, 25.0, 0.5)]


def test_hog_backend_with_no_detections_returns_empty_list():
    class _EmptyHog:
        def detectMultiScale(self, frame, **kwargs):
            return np.empty((0, 4)), np.empty((0,))

    backend = HogBackend(DetectionConfig(backend="hog"), hog=_EmptyHog())
    assert backend.raw_detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_draw_boxes_does_not_mutate_input_and_returns_same_shape():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    boxes = [PersonBox(x1=1, y1=1, x2=10, y2=10, confidence=0.9)]
    annotated = draw_boxes(frame, boxes)
    assert annotated.shape == frame.shape
    assert (frame == 0).all()  # original untouched
    assert (annotated != 0).any()  # something was drawn
