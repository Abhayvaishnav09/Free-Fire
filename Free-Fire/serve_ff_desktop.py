#!/usr/bin/env python3
"""
Free Fire AI server — receives OBS frames from 16score-desktop, runs ffkillblock.

Terminal 1 (AI machine):
    python3 serve_ff_desktop.py

Terminal 2 (desktop — Start Match in UI):
    Streams frames only; detection + TMS happen here on the server.
"""
from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Free Fire AI server for 16score-desktop")
    parser.add_argument("--ingest-port", type=int, default=50052, help="Desktop frame ingest port")
    parser.add_argument("--model", default="best (1).pt", help="YOLO model path")
    args = parser.parse_args()

    # Start frame ingest immediately so Desktop can connect while YOLO loads.
    from frame_ingest import FrameIngestServer, frames_received, get_latest_frame

    frame_queue: queue.Queue = queue.Queue(maxsize=90)
    ingest = FrameIngestServer(port=args.ingest_port, frame_queue=frame_queue)
    ingest.start()

    print("=" * 60, flush=True)
    print("  Free Fire AI Server (desktop → frames → ffkillblock)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Frame ingest : 0.0.0.0:{args.ingest_port}", flush=True)
    print(f"  Model        : {args.model}", flush=True)
    print(flush=True)
    print("  On desktop: login → Start Match (sends OBS frames here)", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    print("✅ Frame ingest ready — Desktop can stream now", flush=True)
    print("⏳ Waiting for first frame from 16score-desktop…", flush=True)

    while get_latest_frame(timeout=2.0) is None:
        print("   …still waiting (start match in Desktop app)", flush=True)
    print(f"✅ Receiving frames ({frames_received()} so far)", flush=True)
    print("📦 Loading YOLO + OCR (this may take ~1 min on first run)…", flush=True)

    from ffkillblock import KillblockDetector, _resolve_tms_credentials, _load_app_config
    from killfeed.remote_capture import RemoteCaptureThread

    match_id, access_token = _resolve_tms_credentials(None, None)
    if not access_token:
        print("⚠️  No TMS token in .desktop_session.json — login in Desktop first", flush=True)
    print(f"⚡ Match={match_id} | TMS={'on' if access_token else 'off'}", flush=True)

    stop_event = threading.Event()

    def _handle_sig(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    detector = KillblockDetector(
        match_id=str(match_id),
        access_token=access_token,
        api_enabled=bool(access_token),
        use_local_model=True,
        model_path=args.model,
        camera_index=-1,
        stop_flag=stop_event,
        auto_resolve_auth=False,
    )

    if detector.local_model is None:
        print("❌ Failed to load YOLO model", flush=True)
        ingest.stop()
        sys.exit(1)

    detection_cfg = _load_app_config().get("detection", {})
    remote_cap = RemoteCaptureThread(
        frame_queue,
        fps=int(detection_cfg.get("capture_fps", 30)),
        buffer_seconds=float(detection_cfg.get("frame_buffer_seconds", 3)),
    )

    try:
        detector._run_fifo_pipeline_loop(capture_override=remote_cap)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        detector.stop_event.set()
        ingest.stop()
        print("✅ AI server stopped", flush=True)


if __name__ == "__main__":
    main()
