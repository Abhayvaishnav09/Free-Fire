"""Continuous capture thread with ring buffer — never blocks on OCR."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Deque, Optional, Union

import cv2
import numpy as np

from killfeed.camera_util import open_video_capture

CameraSource = Union[int, str]


@dataclass
class BufferedFrame:
    frame_num: int
    timestamp: float
    frame: np.ndarray


class CaptureThread:
    """Grab frames at target FPS; drain camera buffer for freshest frame."""

    def __init__(
        self,
        camera_source: CameraSource,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        buffer_seconds: float = 3.0,
        open_capture: Optional[Callable[[CameraSource], cv2.VideoCapture]] = None,
    ):
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self._open_capture = open_capture or open_video_capture
        self._cap: Optional[cv2.VideoCapture] = None
        self._ring: Deque[BufferedFrame] = deque(
            maxlen=max(int(fps * buffer_seconds), 30)
        )
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self.frame_num = 0
        self._latest: Optional[BufferedFrame] = None
        self._last_init_fail_log = 0.0

    def _init_camera(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cap = self._open_capture(self.camera_source)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(8):
            cap.grab()
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cap.release()
            return False
        self._cap = cap
        return True

    def _loop(self) -> None:
        interval = 1.0 / max(self.fps, 1)
        while not self._stop.is_set():
            loop_start = time.time()
            if self._cap is None or not self._cap.isOpened():
                if not self._init_camera():
                    now = time.time()
                    if now - self._last_init_fail_log >= 5.0:
                        src = self.camera_source
                        print(
                            f"⏳ OBS Virtual Camera ({src}) reconnecting… "
                            "check OBS → Start Virtual Camera"
                        )
                        self._last_init_fail_log = now
                    time.sleep(1.5)
                    continue
            for _ in range(3):
                self._cap.grab()
            ret, frame = self._cap.retrieve()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.05)
                continue
            self.frame_num += 1
            if self.frame_num == 1:
                h, w = frame.shape[:2]
                print(f"📷 First frame captured ({w}x{h}) from {self.camera_source}")
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
        self._thread = Thread(target=self._loop, daemon=True, name="KillfeedCapture")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_latest(self) -> Optional[BufferedFrame]:
        with self._lock:
            return self._latest

    def get_ring_snapshot(self) -> list[BufferedFrame]:
        with self._lock:
            return list(self._ring)
