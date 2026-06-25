#!/usr/bin/env python3
"""
16score-desktop — screen capture ONLY. Sends frames to Free-Fire AI machine.

Usage (Windows):
    python remote_screen_capture.py --ai-host 192.168.1.100

Linux (full-screen only, no window picker):
    pip install -r requirements-remote.txt
    python3 remote_screen_capture.py --ai-host 192.168.1.100

Optional:
    --ai-port 50052          Frame ingest port on AI machine
    --fps 10                 Capture rate (default 10)
    --window "BlueStacks"    Capture a specific window title (partial match)
"""
import argparse
import os
import sys
import time

import cv2
import grpc
import numpy as np

# gRPC stubs live next to this script (copied from Free-Fire)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    import killfeed_detection_pb2
    import killfeed_detection_pb2_grpc
except ImportError:
    print("❌ Missing gRPC files. Copy from Free-Fire/:")
    print("   killfeed_detection.proto")
    print("   killfeed_detection_pb2.py")
    print("   killfeed_detection_pb2_grpc.py")
    print("   Then: pip install grpcio grpcio-tools && python -m grpc_tools.protoc ...")
    sys.exit(1)

# Platform-specific capture backends
HAS_WIN32 = False
HAS_MSS = False

if sys.platform == "win32":
    try:
        import win32gui
        import win32ui
        from ctypes import windll
        from PIL import Image
        HAS_WIN32 = True
    except ImportError:
        pass

try:
    import mss
    HAS_MSS = True
except ImportError:
    pass

if not HAS_MSS and sys.platform != "win32":
    try:
        import pyautogui
    except Exception:
        print("❌ Install: pip install mss")
        sys.exit(1)


def _screenshot_mss():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        img = sct.grab(monitor)
        frame = np.array(img, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def _screenshot_pyautogui():
    import pyautogui
    shot = pyautogui.screenshot()
    frame = np.array(shot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def find_window(partial_title):
    if not HAS_WIN32 or not partial_title:
        return None
    matches = []

    def handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and partial_title.lower() in title.lower():
                matches.append((hwnd, title))

    win32gui.EnumWindows(handler, None)
    return matches[0] if matches else None


def capture_window_win32(hwnd):
    from PIL import Image
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)
    windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
    info = bitmap.GetInfo()
    bits = bitmap.GetBitmapBits(True)
    image = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]), bits, "raw", "BGRX", 0, 1)
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    return np.array(image)


def capture_frame(window_title=None):
    # Windows: try specific window first, then full screen
    if window_title and HAS_WIN32:
        match = find_window(window_title)
        if match:
            frame_rgb = capture_window_win32(match[0])
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        print(f"⚠️  Window '{window_title}' not found — using full screen")

    # Linux/Windows fallback: mss (fast, no tkinter needed)
    if HAS_MSS:
        return _screenshot_mss()

    # Last resort: pyautogui (Windows only, needs tkinter on Linux)
    return _screenshot_pyautogui()


def main():
    parser = argparse.ArgumentParser(description="16score-desktop screen capture → Free-Fire AI")
    parser.add_argument("--ai-host", required=True, help="IP of the Free-Fire AI machine")
    parser.add_argument("--ai-port", type=int, default=50052, help="Frame ingest port (default 50052)")
    parser.add_argument("--fps", type=float, default=10.0, help="Capture FPS (default 10)")
    parser.add_argument("--window", default=None, help="Window title substring (e.g. BlueStacks, OBS)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100")
    args = parser.parse_args()

    target = f"{args.ai_host}:{args.ai_port}"
    print("=" * 60)
    print("  16score-desktop — Screen Capture Only")
    print("=" * 60)
    print(f"  Sending frames to: {target}")
    print(f"  Capture rate     : {args.fps} FPS")
    if args.window:
        print(f"  Window filter    : {args.window}")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(target, options=options)
    stub = killfeed_detection_pb2_grpc.FrameIngestServiceStub(channel)

    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        print(f"❌ Cannot reach AI at {target}")
        print("   Make sure run_connected_pipeline.py is running on the AI machine")
        sys.exit(1)

    print(f"✅ Connected to AI at {target}")

    interval = 1.0 / max(args.fps, 0.5)
    sent = 0
    errors = 0

    while True:
        loop_start = time.time()
        try:
            frame = capture_frame(args.window)
            if frame is None or frame.size == 0:
                time.sleep(0.1)
                continue

            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue

            request = killfeed_detection_pb2.ProcessFrameRequest(
                frame_image=encoded.tobytes(),
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
            ack = stub.SendFrame(request, timeout=5)
            if ack.success:
                sent += 1
                if sent == 1 or sent % 50 == 0:
                    print(f"📤 Frames sent: {sent}")
            else:
                errors += 1
                if errors <= 3:
                    print(f"⚠️  AI rejected frame: {ack.error_message}")
        except grpc.RpcError as exc:
            errors += 1
            if errors <= 5 or errors % 20 == 0:
                print(f"⚠️  gRPC error: {exc.code()} — retrying…")
            time.sleep(1.0)
        except KeyboardInterrupt:
            print(f"\n⏹️ Stopped. Sent {sent} frames.")
            break

        elapsed = time.time() - loop_start
        time.sleep(max(0.001, interval - elapsed))

    channel.close()


if __name__ == "__main__":
    main()
