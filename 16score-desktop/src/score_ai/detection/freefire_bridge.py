"""
Bridge 16score-desktop → Free Fire AI server.

Desktop ONLY captures OBS frames and streams them to the AI machine.
All YOLO / OCR / killfeed detection runs on the server (serve_ff_desktop.py).

On localhost, the AI server is auto-started if not already running.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import cv2
import grpc
import numpy as np

_SERVER_PROC = None


def _desktop_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _grpc_proto_path() -> str:
    return os.path.join(_desktop_root(), "app")


def _import_grpc_stubs():
    app_dir = _grpc_proto_path()
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    import killfeed_detection_pb2
    import killfeed_detection_pb2_grpc

    return killfeed_detection_pb2, killfeed_detection_pb2_grpc


def _resolve_freefire_dir() -> str:
    desktop_root = _desktop_root()
    parent_root = os.path.abspath(os.path.join(desktop_root, ".."))
    for path in (
        os.path.join(parent_root, "Free-Fire"),
        os.path.join(parent_root, "Free Fire"),
    ):
        if os.path.isfile(os.path.join(path, "ffkillblock.py")):
            return os.path.normpath(path)
    raise FileNotFoundError("Free-Fire/ffkillblock.py not found next to 16score-desktop")


def sync_desktop_auth(match_id=None, access_token=None, user_email=None):
    """Write Desktop login/match into Free-Fire/.desktop_session.json for the AI server."""
    try:
        freefire_dir = _resolve_freefire_dir()
        if freefire_dir not in sys.path:
            sys.path.insert(0, freefire_dir)
        from killfeed.desktop_session import save_desktop_match, save_desktop_token

        if access_token:
            save_desktop_token(access_token, user_email, base_dir=freefire_dir)
        if match_id:
            save_desktop_match(str(match_id), access_token, user_email, base_dir=freefire_dir)
    except Exception as exc:
        print(f"[Free Fire] session sync skipped: {exc}", flush=True)


def _format_connect_error(exc: BaseException) -> str:
    if isinstance(exc, grpc.FutureTimeoutError):
        return "connection timed out (is serve_ff_desktop.py running?)"
    if isinstance(exc, grpc.RpcError):
        return f"gRPC {exc.code()}: {exc.details() or exc}"
    text = str(exc).strip()
    return text or repr(exc)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_local_ai_server(port: int, model_path: str) -> subprocess.Popen | None:
    global _SERVER_PROC
    if _SERVER_PROC is not None and _SERVER_PROC.poll() is None:
        return _SERVER_PROC

    freefire_dir = _resolve_freefire_dir()
    script = os.path.join(freefire_dir, "serve_ff_desktop.py")
    if not os.path.isfile(script):
        print(f"[Free Fire] ❌ Missing {script}", flush=True)
        return None

    log_dir = os.path.join(_desktop_root(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ai_server.log")

    print(f"[Free Fire] 🚀 Starting local AI server (port {port})…", flush=True)
    print(f"[Free Fire]    Server log: {log_path}", flush=True)

    log_file = open(log_path, "a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        script,
        "--ingest-port",
        str(port),
        "--model",
        model_path,
    ]
    _SERVER_PROC = subprocess.Popen(
        cmd,
        cwd=freefire_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return _SERVER_PROC


def _ensure_ai_server(host: str, port: int, model_path: str, auto_start: bool) -> bool:
    """Return True when frame ingest port accepts connections."""
    if _port_open(host, port):
        return True

    is_local = host in ("127.0.0.1", "localhost", "::1")
    if not is_local or not auto_start:
        return False

    proc = _start_local_ai_server(port, model_path)
    if proc is None:
        return False

    deadline = time.time() + 45.0
    while time.time() < deadline:
        if proc.poll() is not None:
            print(
                f"[Free Fire] ❌ AI server exited early (code {proc.returncode}). "
                "See logs/ai_server.log",
                flush=True,
            )
            return False
        if _port_open(host, port):
            print("[Free Fire] ✅ AI server is listening", flush=True)
            return True
        time.sleep(0.5)

    print("[Free Fire] ❌ AI server did not start within 45s. See logs/ai_server.log", flush=True)
    return False


def _connect_ingest(target: str, pb2_grpc, options, timeout: float = 8.0):
    channel = grpc.insecure_channel(target, options=options)
    grpc.channel_ready_future(channel).result(timeout=timeout)
    return channel, pb2_grpc.FrameIngestServiceStub(channel)


def _test_ingest_connection(stub, pb2, timeout: float = 8.0) -> bool:
    test = np.ones((8, 8, 3), dtype=np.uint8) * 255
    ok, buf = cv2.imencode(".jpg", test)
    if not ok:
        return False
    req = pb2.ProcessFrameRequest(
        frame_image=buf.tobytes(),
        frame_width=8,
        frame_height=8,
    )
    ack = stub.SendFrame(req, timeout=timeout)
    return bool(ack.success)


def _open_camera(camera_index: int, width: int = 1920, height: int = 1080):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def obs_frame_capture(match_id=1, access_token=None, camera_index=1, stop_flag=None):
    """
    Capture OBS virtual camera and stream JPEG frames to the Free Fire AI server.
    No local YOLO/OCR — detection runs in serve_ff_desktop.py / ffkillblock.py.
    """
    from score_ai.core.config_manager import config

    pb2, pb2_grpc = _import_grpc_stubs()

    ai_host = config.get("freefire.ai_host") or config.get("grpc.ai_host", "127.0.0.1")
    ingest_port = int(config.get("freefire.ingest_port", 50052))
    model_path = config.get("freefire.model_path", "best (1).pt")
    auto_start = bool(config.get("freefire.auto_start_server", True))
    fps = float(config.get("freefire.stream_fps", config.get("detection.capture_fps", 30)))
    fps = max(5.0, min(60.0, fps))
    quality = int(config.get("freefire.jpeg_quality", 85))
    quality = max(70, min(95, quality))
    target = f"{ai_host}:{ingest_port}"

    print(
        f"[Free Fire] Desktop → AI server {target} | match={match_id} | camera={camera_index}",
        flush=True,
    )
    print("[Free Fire] Desktop role: stream frames only (no local detection)", flush=True)

    sync_desktop_auth(match_id=match_id, access_token=access_token)

    if not _ensure_ai_server(ai_host, ingest_port, model_path, auto_start):
        print(f"[Free Fire] ❌ AI server not reachable at {target}", flush=True)
        if ai_host in ("127.0.0.1", "localhost"):
            print(
                "[Free Fire] Start manually:\n"
                "  cd Free-Fire && python3 serve_ff_desktop.py",
                flush=True,
            )
        else:
            print(
                f"[Free Fire] Start serve_ff_desktop.py on {ai_host} "
                f"and open port {ingest_port}",
                flush=True,
            )
        return

    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]

    print(f"[Free Fire] Connecting to AI server at {target}…", flush=True)
    channel = None
    stub = None
    last_err = None
    for attempt in range(1, 6):
        try:
            channel, stub = _connect_ingest(target, pb2_grpc, options, timeout=10.0)
            if _test_ingest_connection(stub, pb2):
                break
            raise RuntimeError("AI server rejected test frame")
        except Exception as exc:
            last_err = exc
            if channel is not None:
                channel.close()
                channel = None
                stub = None
            if attempt < 5:
                print(
                    f"[Free Fire] Retry {attempt}/5: {_format_connect_error(exc)}",
                    flush=True,
                )
                time.sleep(2.0)
    else:
        print(
            f"[Free Fire] ❌ Cannot connect to AI server at {target}: "
            f"{_format_connect_error(last_err)}",
            flush=True,
        )
        return

    print(f"[Free Fire] ✅ Connected — streaming @ {fps:.0f} fps", flush=True)

    cap = _open_camera(
        camera_index,
        width=int(config.get("camera.default_width", 1920)),
        height=int(config.get("camera.default_height", 1080)),
    )
    if cap is None:
        print(f"[Free Fire] ❌ Could not open camera index {camera_index}", flush=True)
        channel.close()
        return

    interval = 1.0 / fps
    sent = 0
    errors = 0

    try:
        while stop_flag is None or not stop_flag.is_set():
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.05)
                continue

            enc_ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not enc_ok:
                continue

            try:
                req = pb2.ProcessFrameRequest(
                    frame_image=buf.tobytes(),
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                )
                ack = stub.SendFrame(req, timeout=3.0)
                if ack.success:
                    sent += 1
                    if sent == 1 or sent % 100 == 0:
                        print(f"[Free Fire] 📤 Frames streamed to AI: {sent}", flush=True)
                else:
                    errors += 1
                    if errors <= 3:
                        print(f"[Free Fire] ⚠️ AI rejected frame: {ack.error_message}", flush=True)
            except grpc.RpcError as exc:
                errors += 1
                if errors <= 5 or errors % 20 == 0:
                    print(f"[Free Fire] ⚠️ gRPC error: {exc.code()} — retrying…", flush=True)
                time.sleep(0.5)

            elapsed = time.time() - loop_start
            time.sleep(max(0.001, interval - elapsed))
    finally:
        cap.release()
        channel.close()
        print(f"[Free Fire] Stream stopped ({sent} frames sent)", flush=True)


def get_obs_frame_capture():
    return obs_frame_capture
