"""Capture thread that reads frames pushed by the desktop app (no OBS camera)."""

from __future__ import annotations

import time
from collections import deque
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Deque, Optional, Union

import cv2
import numpy as np

from killfeed.capture import BufferedFrame

FrameBuffer = Union[Queue, object]


class RemoteCaptureThread:
    """Ring buffer fed by desktop ``frame_buffer`` (JPEG bytes or BGR arrays)."""

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        fps: int = 30,
        buffer_seconds: float = 3.0,
    ):
        self.frame_buffer = frame_buffer
        self.fps = max(1, int(fps))
        self.buffer_seconds = float(buffer_seconds)
        self._ring: Deque[BufferedFrame] = deque(
            maxlen=max(int(self.fps * self.buffer_seconds), 30)
        )
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self.frame_num = 0
        self._latest: Optional[BufferedFrame] = None

    def _decode_item(self, item) -> Optional[np.ndarray]:
        if item is None:
            return None
        if isinstance(item, np.ndarray):
            return item if item.size > 0 else None
        if isinstance(item, (bytes, bytearray)):
            arr = np.frombuffer(item, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame if frame is not None and frame.size > 0 else None
        if isinstance(item, dict):
            data = item.get("frame") or item.get("image") or item.get("data")
            return self._decode_item(data)
        return None

    def _poll_frame(self) -> Optional[np.ndarray]:
        buf = self.frame_buffer
        if buf is None:
            return None
        try:
            if hasattr(buf, "get_nowait"):
                while True:
                    try:
                        item = buf.get_nowait()
                    except Empty:
                        break
                    frame = self._decode_item(item)
                    if frame is not None:
                        return frame
            elif hasattr(buf, "get"):
                item = buf.get(timeout=0.05)
                return self._decode_item(item)
        except Empty:
            return None
        except Exception:
            return None
        return None

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        while not self._stop.is_set():
            loop_start = time.time()
            frame = self._poll_frame()
            if frame is not None and frame.size > 0:
                self.frame_num += 1
                bf = BufferedFrame(
                    frame_num=self.frame_num,
                    timestamp=time.time(),
                    frame=frame.copy(),
                )
                with self._lock:
                    self._ring.append(bf)
                    self._latest = bf
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._loop, daemon=True, name="RemoteCapture")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def get_latest(self) -> Optional[BufferedFrame]:
        with self._lock:
            return self._latest

    def get_ring_snapshot(self) -> list[BufferedFrame]:
        with self._lock:
            return list(self._ring)
