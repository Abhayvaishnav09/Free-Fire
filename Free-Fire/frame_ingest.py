"""gRPC frame ingest server — receives screen frames from 16score-desktop."""
import logging
import threading
import time
from concurrent import futures

import cv2
import grpc
import numpy as np

try:
    import killfeed_detection_pb2
    import killfeed_detection_pb2_grpc
except ImportError:
    killfeed_detection_pb2 = None
    killfeed_detection_pb2_grpc = None

logger = logging.getLogger(__name__)

_latest_frame = None
_frame_lock = threading.Lock()
_frame_event = threading.Event()
_frames_received = 0


def get_latest_frame(timeout=1.0):
    """Return the most recent BGR frame from desktop capture, or None."""
    if not _frame_event.wait(timeout=timeout):
        return None
    with _frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()


def frames_received():
    with _frame_lock:
        return _frames_received


class FrameIngestServicer(killfeed_detection_pb2_grpc.FrameIngestServiceServicer):
    def __init__(self, frame_queue=None):
        self._frame_queue = frame_queue

    def SendFrame(self, request, context):
        global _latest_frame, _frames_received
        try:
            frame_array = np.frombuffer(request.frame_image, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return killfeed_detection_pb2.FrameAck(
                    success=False, error_message="Failed to decode frame"
                )
            with _frame_lock:
                _latest_frame = frame
                _frames_received += 1
            _frame_event.set()
            if self._frame_queue is not None:
                try:
                    self._frame_queue.put_nowait(frame)
                except Exception:
                    try:
                        self._frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except Exception:
                        pass
            return killfeed_detection_pb2.FrameAck(success=True)
        except Exception as exc:
            logger.error("Frame ingest error: %s", exc)
            return killfeed_detection_pb2.FrameAck(success=False, error_message=str(exc))


class FrameIngestServer:
    def __init__(self, port=50052, max_workers=4, frame_queue=None):
        self.port = port
        self.max_workers = max_workers
        self.frame_queue = frame_queue
        self._server = None
        self._thread = None

    def start(self):
        if killfeed_detection_pb2 is None or killfeed_detection_pb2_grpc is None:
            raise RuntimeError(
                "gRPC code missing. Run: python generate_grpc_code.py"
            )
        options = [
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=options,
        )
        killfeed_detection_pb2_grpc.add_FrameIngestServiceServicer_to_server(
            FrameIngestServicer(frame_queue=self.frame_queue), self._server
        )
        self._server.add_insecure_port(f"0.0.0.0:{self.port}")
        self._server.start()
        logger.info("Frame ingest server listening on 0.0.0.0:%s", self.port)

    def stop(self):
        if self._server:
            self._server.stop(grace=2)
            self._server = None
