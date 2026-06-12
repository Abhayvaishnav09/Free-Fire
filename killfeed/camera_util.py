"""V4L2 / OBS Virtual Camera discovery and OpenCV capture helpers."""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional, Union

import cv2

CameraSource = Union[int, str]


def list_v4l2_devices() -> list[dict]:
    """List /dev/video* devices with human-readable names from sysfs."""
    devices: list[dict] = []
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return devices
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("video"):
            continue
        try:
            index = int(entry.replace("video", ""))
        except ValueError:
            continue
        label = ""
        name_path = os.path.join(base, entry, "name")
        try:
            with open(name_path, encoding="utf-8") as f:
                label = f.read().strip()
        except OSError:
            pass
        devices.append(
            {
                "index": index,
                "path": f"/dev/{entry}",
                "name": label,
            }
        )
    return devices


def find_device_by_name(name_substring: str) -> Optional[dict]:
    needle = (name_substring or "").lower()
    if not needle:
        return None
    for dev in list_v4l2_devices():
        if needle in dev["name"].lower():
            return dev
    return None


def _video_index_from_path(path: str) -> Optional[int]:
    try:
        return int(path.replace("/dev/video", ""))
    except ValueError:
        return None


def _configure_loopback_cap(cap: cv2.VideoCapture, width: int, height: int) -> None:
    """Best-effort settings for OBS v4l2loopback devices."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    except Exception:
        pass


def open_video_capture(source: CameraSource, width: int = 1920, height: int = 1080) -> cv2.VideoCapture:
    """
    Open OBS v4l2loopback or regular webcam.
    Prefer device PATH (/dev/videoN) over numeric index on Linux.
    """
    candidates: list[CameraSource] = []
    if isinstance(source, str):
        candidates.append(source)
        idx = _video_index_from_path(source)
        if idx is not None:
            candidates.append(idx)
    else:
        index = int(source)
        candidates.append(f"/dev/video{index}")
        candidates.append(index)

    if sys.platform == "win32":
        for src in candidates:
            cap = cv2.VideoCapture(str(src) if isinstance(src, str) else int(src))
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(int(candidates[-1]) if candidates else 0)

    for src in candidates:
        if isinstance(src, str):
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
        if cap.isOpened():
            _configure_loopback_cap(cap, width, height)
            return cap
        cap.release()

    return cv2.VideoCapture(candidates[0] if candidates else 0, cv2.CAP_V4L2)


def _probe_camera_once(
    source: CameraSource,
    min_width: int,
    min_height: int,
) -> Optional[tuple[int, int]]:
    cap = open_video_capture(source)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        for _ in range(8):
            cap.grab()
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    if w < min_width or h < min_height:
        return None
    return w, h


def probe_camera(
    source: CameraSource,
    min_width: int = 640,
    min_height: int = 480,
    timeout_sec: float = 3.0,
) -> Optional[tuple[int, int]]:
    """Return (width, height) if the source delivers a frame within timeout."""
    result: list[Optional[tuple[int, int]]] = [None]

    def _worker() -> None:
        try:
            result[0] = _probe_camera_once(source, min_width, min_height)
        except Exception:
            result[0] = None

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    return result[0]


def resolve_obs_camera(
    preferred_index: Optional[int] = None,
    obs_device_name: str = "OBS Virtual Camera",
    prefer_obs_only: bool = True,
    min_width: int = 640,
    min_height: int = 480,
) -> tuple[Optional[CameraSource], Optional[tuple[int, int]], Optional[str]]:
    """
    Resolve OBS Virtual Camera ONLY when it delivers real frames.
    Never returns a fake size — probe must succeed.
    """
    obs = find_device_by_name(obs_device_name)
    if obs is not None:
        for src in (obs["path"], obs["index"]):
            size = probe_camera(src, min_width=min_width, min_height=min_height, timeout_sec=3.0)
            if size is not None:
                return obs["path"], size, obs["name"]

    if preferred_index is not None and not prefer_obs_only:
        for src in (f"/dev/video{preferred_index}", preferred_index):
            size = probe_camera(src, min_width=min_width, min_height=min_height, timeout_sec=2.0)
            if size is not None:
                return src, size, f"index {preferred_index}"

    return None, None, None
