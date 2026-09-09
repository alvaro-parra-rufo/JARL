"""Asynchronous MP4 encoding for Navix policy videos."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from jarl.envs.navix.custom.rgb_overlay import apply_empty_variant_floor_tint
from jarl.io.video import encode_rgb_video

LOGGER = logging.getLogger(__name__)

_RGB_FRAME_RANK = 5
_RGB_CHANNEL_COUNT = 3

__all__ = [
    "AsyncNavixVideoRecorder",
    "NavixVideoJob",
]


@dataclass(frozen=True)
class NavixVideoJob:
    """CPU video frames and metadata for one asynchronous encode job."""

    frames: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    global_step: int
    name_prefix: str
    fps: int
    scale: int
    rollout_seconds: float
    transfer_seconds: float
    is_final: bool = False
    env_id: str | None = None
    view_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AsyncNavixVideoRecorder:
    """Encode Navix MP4 videos in one background worker."""

    def __init__(
        self,
        video_dir: str | Path,
        metrics_jsonl_path: str | Path,
        *,
        max_pending_videos: int = 1,
    ) -> None:
        """Create an asynchronous video recorder."""
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl_path = Path(metrics_jsonl_path)
        self.max_pending_videos = max(1, int(max_pending_videos))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarl-navix-video")
        self._pending: Future[None] | None = None
        self._closed = False
        self._lock = Lock()

    def can_accept_intermediate(self) -> bool:
        """Return whether an intermediate video can be scheduled without blocking."""
        with self._lock:
            self._collect_finished_locked()
            return self._pending is None

    def submit_intermediate(self, job: NavixVideoJob) -> bool:
        """Submit an intermediate video or skip it when the worker is busy."""
        with self._lock:
            self._collect_finished_locked()
            if self._closed:
                self._write_metric_locked(job, status="skipped", reason="recorder_closed")
                return False
            if self._pending is not None:
                self._write_metric_locked(job, status="skipped", reason="encoder_busy")
                return False
            self._pending = self._executor.submit(self._encode_job, job)
            return True

    def submit_final(self, job: NavixVideoJob) -> None:
        """Submit a final video and wait for it to finish."""
        self.wait()
        with self._lock:
            if self._closed:
                self._write_metric_locked(job, status="skipped", reason="recorder_closed")
                return
            self._pending = self._executor.submit(self._encode_job, job)
        self.wait()

    def wait(self) -> None:
        """Wait for the current pending video, if any."""
        with self._lock:
            future = self._pending
        if future is None:
            return
        try:
            future.result()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Navix video worker failed: %s", exc)
        finally:
            with self._lock:
                if self._pending is future:
                    self._pending = None

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background worker."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if wait:
            self.wait()
        self._executor.shutdown(wait=wait)

    def _collect_finished_locked(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        try:
            self._pending.result()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Navix video worker failed: %s", exc)
        finally:
            self._pending = None

    def _encode_job(self, job: NavixVideoJob) -> None:
        if job.frames.ndim != _RGB_FRAME_RANK or job.frames.shape[-1] != _RGB_CHANNEL_COUNT:
            self._write_metric_locked(job, status="failed", reason=f"invalid_frame_shape:{job.frames.shape}")
            return

        for episode_index in range(job.frames.shape[0]):
            start_time = time.perf_counter()
            final_path = self.video_dir / f"{job.name_prefix}_episode_{episode_index + 1:02d}.mp4"
            temp_path = self.video_dir / f"{job.name_prefix}_episode_{episode_index + 1:02d}.tmp.mp4"
            frame_count = self._frame_count(job, episode_index)
            try:
                episode_frames = apply_empty_variant_floor_tint(
                    np.asarray(job.frames[episode_index, :frame_count], dtype=np.uint8),
                    env_id=job.env_id or "",
                    view_mode=job.view_mode,
                )
                encode_rgb_video(
                    episode_frames,
                    final_path,
                    fps=job.fps,
                    scale=job.scale,
                )
                encode_seconds = time.perf_counter() - start_time
                file_size_bytes = final_path.stat().st_size
                self._write_metric_locked(
                    job,
                    status="completed",
                    episode_index=episode_index,
                    video_file=final_path,
                    frame_count=frame_count,
                    encode_seconds=encode_seconds,
                    file_size_bytes=file_size_bytes,
                )
            except Exception as exc:  # pragma: no cover
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                self._write_metric_locked(
                    job,
                    status="failed",
                    reason=str(exc),
                    episode_index=episode_index,
                    video_file=final_path,
                    frame_count=frame_count,
                    encode_seconds=time.perf_counter() - start_time,
                )

    def _frame_count(self, job: NavixVideoJob, episode_index: int) -> int:
        available_frames = int(job.frames.shape[1])
        episode_length = int(np.asarray(job.episode_lengths[episode_index]))
        return max(1, min(available_frames, episode_length + 1))

    def _write_metric_locked(
        self,
        job: NavixVideoJob,
        status: str,
        reason: str | None = None,
        episode_index: int | None = None,
        video_file: Path | None = None,
        frame_count: int | None = None,
        encode_seconds: float | None = None,
        file_size_bytes: int | None = None,
    ) -> None:
        metric = {
            "status": status,
            "reason": reason,
            "global_step": int(job.global_step),
            "video_file": str(video_file) if video_file is not None else None,
            "episode_index": episode_index,
            "episode_return": float(np.asarray(job.episode_returns[episode_index]))
            if episode_index is not None
            else None,
            "episode_length": int(np.asarray(job.episode_lengths[episode_index]))
            if episode_index is not None
            else None,
            "frame_count": frame_count,
            "video_scale": int(job.scale),
            "encoded_frame_shape": _encoded_frame_shape(job.frames, job.scale),
            "rollout_seconds": float(job.rollout_seconds),
            "transfer_seconds": float(job.transfer_seconds),
            "encode_seconds": encode_seconds,
            "file_size_bytes": file_size_bytes,
            "is_final": bool(job.is_final),
        }
        if job.metadata:
            metric.update(job.metadata)
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(metric) + "\n")


def _encoded_frame_shape(frames: np.ndarray, scale: int) -> list[int] | None:
    if frames.ndim != _RGB_FRAME_RANK:
        return None
    scale = max(1, int(scale))
    return [int(frames.shape[2]) * scale, int(frames.shape[3]) * scale]
