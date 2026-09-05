from pathlib import Path

import pytest

from classroom_analysis.config import ConfigError, FrameExtractionConfig, load_frame_extraction_config


def test_loads_nested_frame_extraction_section(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "frame_extraction:\n"
        "  sample_rate_hz: 5\n"
        "  resize_width: 100\n"
        "  resize_height: 80\n"
        "  output_format: rgb24\n"
    )
    config = load_frame_extraction_config(path)
    assert config.sample_rate_hz == 5
    assert (config.resize_width, config.resize_height) == (100, 80)
    assert config.output_format == "rgb24"


def test_loads_bare_mapping_without_nested_key(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("sample_rate_hz: 3\n")
    config = load_frame_extraction_config(path)
    assert config.sample_rate_hz == 3


def test_missing_config_file_is_a_hard_error():
    with pytest.raises(ConfigError):
        load_frame_extraction_config("configs/does_not_exist.yaml")


def test_empty_config_file_is_a_hard_error(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ConfigError):
        load_frame_extraction_config(path)


def test_non_mapping_config_is_a_hard_error(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_frame_extraction_config(path)


def test_zero_sample_rate_is_rejected():
    with pytest.raises(ValueError):
        FrameExtractionConfig(sample_rate_hz=0)


def test_partial_resize_dims_are_rejected():
    with pytest.raises(ValueError):
        FrameExtractionConfig(resize_width=100, resize_height=None)


def test_unknown_field_is_a_hard_error(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("frame_extraction:\n  not_a_real_field: 1\n")
    with pytest.raises(ConfigError):
        load_frame_extraction_config(path)
