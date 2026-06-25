"""When ffkillblock runs inside 16score Desktop, send logs to Free-Fire files only."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import IO, Optional, TextIO, Tuple

_state: Optional[Tuple[TextIO, TextIO, TextIO, str]] = None


class _FileOnlyStream:
    """Replace stdout/stderr — writes to log file, not Desktop console."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self._file.write(
            f"\n--- detection log started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
        self._file.flush()

    def write(self, data) -> int:
        if data:
            self._file.write(data)
            self._file.flush()
        return len(data) if data else 0

    def flush(self) -> None:
        self._file.flush()

    def fileno(self) -> int:
        return self._file.fileno()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass


def install_desktop_detection_log(
    log_path: str,
    *,
    notify_stream: Optional[IO[str]] = None,
) -> str:
    """Redirect stdout/stderr to *log_path*; optional one-line hint on Desktop."""
    global _state
    if _state is not None:
        _close_current()

    abs_path = os.path.abspath(log_path)
    writer = _FileOnlyStream(abs_path)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = writer  # type: ignore[assignment]
    sys.stderr = writer  # type: ignore[assignment]
    _state = (writer, orig_out, orig_err, abs_path)
    os.environ["FF_DETECTION_LOG"] = abs_path

    if notify_stream is not None:
        try:
            notify_stream.write(f"📝 Free-Fire killfeed logs → {abs_path}\n")
            notify_stream.write(f"   tail -f {abs_path}\n")
            notify_stream.flush()
        except OSError:
            pass
    return abs_path


def switch_desktop_log(log_path: str, *, notify_stream: Optional[IO[str]] = None) -> str:
    """Move active log to session folder (after session_dir is created)."""
    abs_path = os.path.abspath(log_path)
    install_desktop_detection_log(abs_path)
    print(f"📁 Session detection log: {abs_path}")
    if notify_stream is not None:
        try:
            notify_stream.write(f"📝 Session log → {abs_path}\n")
            notify_stream.flush()
        except OSError:
            pass
    return abs_path


def restore_desktop_log() -> None:
    global _state
    _close_current()


def _close_current() -> None:
    global _state
    if _state is None:
        return
    writer, orig_out, orig_err, _ = _state
    try:
        writer.close()
    except Exception:
        pass
    sys.stdout = orig_out
    sys.stderr = orig_err
    os.environ.pop("FF_DETECTION_LOG", None)
    _state = None
