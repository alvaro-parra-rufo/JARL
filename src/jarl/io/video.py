"""Atomic RGB video encoding helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

_RGB_FRAME_RANK = 4
"""Expected rank for a single ``(time, height, width, channels)`` video."""

_RGB_CHANNEL_COUNT = 3
"""Expected channel count for RGB frames."""

__all__ = ["encode_rgb_video"]


def encode_rgb_video(
    frames: np.ndarray,
    destination: str | Path,
    *,
    fps: int,
    scale: int = 1,
) -> Path:
    """Encode one RGB frame sequence to an MP4 file atomically.

    Args:
        frames: RGB frames shaped ``(time, height, width, 3)``.
        destination: Final MP4 path.
        fps: Encoded frames per second.
        scale: Positive nearest-neighbour integer scale.

    Returns:
        Final MP4 path.

    Raises:
        ValueError: If frames, `fps`, or `scale` are invalid.
    """
    array = np.asarray(frames)
    if array.ndim != _RGB_FRAME_RANK or array.shape[-1] != _RGB_CHANNEL_COUNT:
        msg = f"RGB video frames must have shape (time, height, width, 3), got {array.shape}."
        raise ValueError(msg)
    if array.shape[0] == 0:
        msg = "RGB video requires at least one frame."
        raise ValueError(msg)
    if fps <= 0:
        msg = f"fps must be > 0, got {fps}."
        raise ValueError(msg)
    if scale <= 0:
        msg = f"scale must be > 0, got {scale}."
        raise ValueError(msg)

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    encoded_frames = _scale_frames_nearest(
        np.asarray(array, dtype=np.uint8),
        scale,
    )
    clip = ImageSequenceClip(list(encoded_frames), fps=fps)
    try:
        clip.write_videofile(str(temporary), logger=None)
        os.replace(temporary, output)
    finally:
        clip.close()
        temporary.unlink(missing_ok=True)
    return output


def _scale_frames_nearest(frames: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return frames
    return np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)
