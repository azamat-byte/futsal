import numpy as np
import pytest
from futsal.video import VideoWriter, probe, frames, save_image, load_image, frame_at


def test_unicode_video_roundtrip(tmp_path):
    path = tmp_path / "матч space.mp4"
    writer = VideoWriter(path, 10, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.close()
    assert probe(path)["width"] == 64
    decoded = list(frames(path, fps=5))
    assert len(decoded) == 5
    assert decoded[1].time == pytest.approx(0.2, abs=0.01)
    assert decoded[1].dt == pytest.approx(0.2, abs=0.01)
    with pytest.raises(ValueError, match="within this video"):
        frame_at(path, 600)


def test_unicode_image(tmp_path):
    path = tmp_path / "кадр.jpg"
    save_image(path, np.zeros((20, 30, 3), np.uint8))
    assert load_image(path).shape == (20, 30, 3)
