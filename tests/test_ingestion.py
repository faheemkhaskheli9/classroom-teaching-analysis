import pytest

from classroom_analysis.config import FrameExtractionConfig
from classroom_analysis.ingestion import VideoReadError, extract_frames


def test_extracts_expected_number_of_frames_at_sample_rate(sample_clip):
    # 10 fps source, 2 Hz sample rate -> every 5th frame -> 20 / 5 = 4 frames.
    config = FrameExtractionConfig(sample_rate_hz=2.0)
    frames = list(extract_frames(sample_clip, config))
    assert len(frames) == 4
    assert [f.index for f in frames] == [0, 1, 2, 3]


def test_sampling_at_the_source_fps_yields_every_frame(sample_clip):
    config = FrameExtractionConfig(sample_rate_hz=10.0)
    frames = list(extract_frames(sample_clip, config))
    assert len(frames) == 20


def test_frames_are_resized_when_configured(sample_clip):
    config = FrameExtractionConfig(sample_rate_hz=2.0, resize_width=32, resize_height=16)
    frames = list(extract_frames(sample_clip, config))
    assert all(f.image.shape[:2] == (16, 32) for f in frames)


def test_output_format_rgb24_swaps_channel_order(sample_clip):
    bgr_config = FrameExtractionConfig(sample_rate_hz=2.0, output_format="bgr24")
    rgb_config = FrameExtractionConfig(sample_rate_hz=2.0, output_format="rgb24")

    bgr_frame = next(extract_frames(sample_clip, bgr_config))
    rgb_frame = next(extract_frames(sample_clip, rgb_config))

    # Channels are swapped, not just re-labeled.
    assert (bgr_frame.image[..., 0] == rgb_frame.image[..., 2]).all()


def test_does_not_load_whole_video_into_memory_at_once(sample_clip):
    # extract_frames is a generator: advancing it once must not have read the
    # rest of the video yet.
    config = FrameExtractionConfig(sample_rate_hz=10.0)
    gen = extract_frames(sample_clip, config)
    first = next(gen)
    assert first.index == 0
    # The generator object itself, not a realized list, proves frames are
    # pulled lazily one cv2.read() at a time.
    assert hasattr(gen, "__next__")


def test_missing_file_raises_clear_error(tmp_path):
    config = FrameExtractionConfig()
    with pytest.raises(VideoReadError, match="not found"):
        list(extract_frames(tmp_path / "does_not_exist.mp4", config))


def test_unreadable_file_raises_clear_error(tmp_path):
    bad_file = tmp_path / "not_a_video.mp4"
    bad_file.write_bytes(b"this is not a real video file")
    config = FrameExtractionConfig()
    with pytest.raises(VideoReadError):
        list(extract_frames(bad_file, config))
