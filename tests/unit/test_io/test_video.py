"""Tests for atomic RGB video encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from jarl.io.video import encode_rgb_video


class TestEncodeRgbVideo:
    """Frame validation, scaling, and atomic destination handling."""

    def test_encodes_scaled_frames_atomically(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        clip = mocker.Mock()

        def write_video(path: str, *, logger: object) -> None:
            assert logger is None
            Path(path).write_bytes(b"mp4")

        clip.write_videofile.side_effect = write_video
        clip_cls = mocker.patch(
            "jarl.io.video.ImageSequenceClip",
            return_value=clip,
        )
        frames = np.zeros((3, 2, 4, 3), dtype=np.uint8)
        destination = tmp_path / "nested" / "rollout.mp4"

        result = encode_rgb_video(
            frames,
            destination,
            fps=12,
            scale=2,
        )

        assert result == destination
        assert destination.read_bytes() == b"mp4"
        assert not destination.with_suffix(".tmp.mp4").exists()
        encoded_frames = np.asarray(clip_cls.call_args.args[0])
        assert encoded_frames.shape == (3, 4, 8, 3)
        assert clip_cls.call_args.kwargs["fps"] == 12
        clip.close.assert_called_once_with()

    @pytest.mark.parametrize(
        ("frames", "fps", "scale", "message"),
        [
            pytest.param(
                np.zeros((2, 4, 3), dtype=np.uint8),
                10,
                1,
                "shape",
                id="rank",
            ),
            pytest.param(
                np.zeros((0, 2, 4, 3), dtype=np.uint8),
                10,
                1,
                "at least one",
                id="empty",
            ),
            pytest.param(
                np.zeros((1, 2, 4, 3), dtype=np.uint8),
                0,
                1,
                "fps",
                id="fps",
            ),
            pytest.param(
                np.zeros((1, 2, 4, 3), dtype=np.uint8),
                10,
                0,
                "scale",
                id="scale",
            ),
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        tmp_path: Path,
        frames: np.ndarray,
        fps: int,
        scale: int,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            encode_rgb_video(
                frames,
                tmp_path / "rollout.mp4",
                fps=fps,
                scale=scale,
            )
