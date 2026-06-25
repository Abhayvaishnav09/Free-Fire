#!/usr/bin/env python3
"""
Run Free-Fire AI pipeline (YOLO + OCR + TMS) receiving frames from 16score-desktop.

Terminal 1 (AI machine — this script):
    python run_connected_pipeline.py --match-id 1 --token YOUR_JWT

Terminal 2 (Windows desktop — screen capture only):
    python remote_screen_capture.py --ai-host <AI_MACHINE_IP>
"""
import argparse
import subprocess
import sys
import threading
import time

from frame_ingest import FrameIngestServer, frames_received, get_latest_frame


def _wait_for_grpc(port, timeout=30):
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        finally:
            sock.close()
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Free-Fire AI pipeline (remote desktop capture)")
    parser.add_argument("--match-id", default="1", help="TMS match ID")
    parser.add_argument("--token", default=None, help="TMS access token (JWT)")
    parser.add_argument("--ingest-port", type=int, default=50052, help="Port for desktop frame ingest")
    parser.add_argument("--grpc-port", type=int, default=50051, help="Port for YOLO gRPC server")
    parser.add_argument("--model", default="best (1).pt", help="YOLO model path")
    parser.add_argument("--no-api", action="store_true", help="Disable TMS API push")
    args = parser.parse_args()

    print("=" * 60)
    print("  Free-Fire AI Pipeline (16score-desktop → AI)")
    print("=" * 60)
    print(f"  YOLO server port : {args.grpc_port}")
    print(f"  Frame ingest port: {args.ingest_port}")
    print(f"  Match ID         : {args.match_id}")
    print()
    print("  On Windows desktop, run:")
    print(f"    python remote_screen_capture.py --ai-host <THIS_MACHINE_IP>")
    print("=" * 60)
    print()

    grpc_proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "grpc_block.py",
            "--serve",
            "--model",
            args.model,
            "--port",
            str(args.grpc_port),
        ],
        cwd=__import__("os").path.dirname(__file__) or ".",
    )

    if not _wait_for_grpc(args.grpc_port):
        print("❌ YOLO gRPC server failed to start on port", args.grpc_port)
        grpc_proc.terminate()
        sys.exit(1)
    print(f"✅ YOLO server ready on port {args.grpc_port}")

    ingest = FrameIngestServer(port=args.ingest_port)
    ingest.start()
    print(f"✅ Frame ingest ready on port {args.ingest_port} — waiting for desktop…")

    print("⏳ Waiting for first frame from 16score-desktop…")
    while get_latest_frame(timeout=2.0) is None:
        print("   …still waiting (start remote_screen_capture.py on Windows)")
    print(f"✅ Receiving frames ({frames_received()} so far)")

    from ff import start_match

    stop_flag = threading.Event()

    def frame_callback():
        return get_latest_frame(timeout=0.5)

    try:
        start_match(
            match_id=args.match_id,
            access_token=args.token,
            camera_index=None,
            stop_flag=stop_flag,
            frame_capture_callback=frame_callback,
        )
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
        stop_flag.set()
    finally:
        ingest.stop()
        grpc_proc.terminate()
        grpc_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
