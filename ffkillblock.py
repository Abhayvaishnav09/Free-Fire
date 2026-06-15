"""Free Fire Killblock Detector - Optimized client for killfeed detection and API integration."""
import cv2
import os

# Reduce Paddle/OpenMP thread contention (helps prevent OCR segfaults on Linux)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

# Reduce OpenCV spam when OBS Virtual Camera is not running yet
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
try:
    cv2.setLogLevel(0)  # LOG_LEVEL_SILENT
except Exception:
    pass

import numpy as np
import warnings
import time
import tempfile
import re
import heapq
import hashlib
from collections import deque
from text import FreeFireTextDetector
import json
import base64
import requests
import subprocess
import sys
import socket
import grpc
from queue import Queue, Empty
from threading import Thread, Lock, Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Try to import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️ RapidFuzz not available. Install with: pip install rapidfuzz")
    print("   Fuzzy matching will use basic Levenshtein distance instead.")

# Try to import YOLO for local processing
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO not available. Install with: pip install ultralytics")

# ---------------------------------------------------------------------------
# Current-match roster loader
# Sirf match_roster.json update karo — yeh file kabhi mat chhedo.
# Naya match = match_roster.json update karo, program restart karo. Done.
# ---------------------------------------------------------------------------
def _load_match_roster(json_path: str = "match_roster.json") -> list:
    """Load player names from match_roster.json (same folder as this script).

    File structure:
      { "teams": [ { "players": [...], "sub": "..." }, ... ] }

    Returns empty list if file is missing — program still works via TMS API roster.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, json_path)
        with open(full_path, encoding="utf-8") as f:
            data = json.load(f)
        names: list = []
        for team in data.get("teams", []):
            for p in team.get("players", []):
                if p and p.strip():
                    namsaes.append(p.strip().upper())
            sub = team.get("sub", "")
            if sub and sub.strip():
                names.append(sub.strip().upper())
        if names:
            print(f"✅ match_roster.json loaded: {len(names)} players from {len(data.get('teams', []))} teams")
        else:
            print("⚠️  match_roster.json: no players found — roster snap disabled, TMS API will be used")
        return names
    except FileNotFoundError:
        print("⚠️  match_roster.json not found — match se pehle file banao (roster snap disabled)")
        return []
    except Exception as e:
        print(f"⚠️  match_roster.json load error ({type(e).__name__}: {e}) — roster snap disabled")
        return []


MATCH_PLAYERS = _load_match_roster()

# Import generated gRPC code
try:
    import killfeed_detection_pb2
    import killfeed_detection_pb2_grpc
except ImportError:
    print("⚠️ gRPC code not generated. Run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. killfeed_detection.proto")
    killfeed_detection_pb2 = None
    killfeed_detection_pb2_grpc = None

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings("ignore")

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
_LOCAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_config.json")


def _load_app_config():
    """Load config.json if present."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _load_local_config():
    """Load local_config.json (gitignored) for match_id + token defaults."""
    try:
        with open(_LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _is_valid_jwt(token):
    """Return True if token looks like a decodable JWT (catches corrupted local_config saves)."""
    if not token or token.count(".") != 2:
        return False
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        base64.urlsafe_b64decode(payload_b64)
        return True
    except Exception:
        return False


def _backend_url_from_config():
    """Resolve TMS backend URL from local_config.json then config.json."""
    local_cfg = _load_local_config()
    app_cfg = _load_app_config()
    backend = local_cfg.get("backend_url") or app_cfg.get("api", {}).get(
        "backend_url", "http://192.168.29.47:5005"
    )
    return backend.rstrip("/")


from killfeed.camera_util import open_video_capture as _open_video_capture_util
from killfeed.camera_util import resolve_obs_camera


def _open_video_capture(source):
    """Open camera by index (int) or device path (/dev/videoN)."""
    return _open_video_capture_util(source)


def _resolve_camera_source(preferred_index=None, quiet=False):
    """
    Resolve OBS Virtual Camera. Returns (source, (w,h), label).
    source is int or /dev/videoN path. Does NOT fall back to laptop webcam.
    """
    config = _load_app_config().get("camera", {})
    if preferred_index is None:
        preferred_index = config.get("default_index", 2)
    obs_name = config.get("obs_device_name", "OBS Virtual Camera")
    prefer_obs_only = config.get("prefer_obs_only", True)
    min_w = int(config.get("default_width", 1920)) // 2
    min_h = int(config.get("default_height", 1080)) // 2

    source, size, label = resolve_obs_camera(
        preferred_index=preferred_index,
        obs_device_name=obs_name,
        prefer_obs_only=prefer_obs_only,
        min_width=min_w,
        min_height=min_h,
    )
    if source is not None and size is not None:
        if not quiet:
            src_label = source if isinstance(source, str) else f"index {source}"
            print(f"📷 {label}: {src_label} ({size[0]}x{size[1]})")
        return source, size, label

    if not quiet:
        print(
            "⚠️  OBS Virtual Camera not ready. In OBS: Start Virtual Camera "
            f"(looking for '{obs_name}' on /dev/video{preferred_index})"
        )
    return None, None, None


class _KillfeedOCRAdapter:
    """Wraps KillblockDetector OCR for strip row parsing."""

    def __init__(self, detector, subprocess_ocr=None):
        self.detector = detector
        self.subprocess_ocr = subprocess_ocr

    def extract_row_names(self, row_crop):
        if self.subprocess_ocr is not None:
            killer_raw, victim_raw, confidence = self.subprocess_ocr.extract_row_names(row_crop)
            return self.detector._finalize_fifo_names(killer_raw, victim_raw, confidence)
        with self.detector.ocr_lock:
            killer, victim, confidence = self.detector._extract_names(row_crop)
        return killer or "", victim or "", float(confidence or 0.0)


class KillblockDetector:
    """Optimized killblock detector with three core responsibilities:
    1. Capture frames from OBS
    2. Process frames locally OR via gRPC server (configurable)
    3. Receive results, perform OCR, and send to TMS API
    """
    
    GRPC_SERVER_PORT = 50051
    GRPC_SERVER_HOST = "localhost"
    
    def __init__(self, match_id="1", access_token=None, api_enabled=True, grpc_port=50051, 
                 use_local_model=True, model_path="best.pt", camera_index=None):
        """Initialize detector with local or gRPC processing, OCR, and API configuration.
        
        Args:
            match_id: Match ID for TMS API
            access_token: Access token for TMS API
            api_enabled: Enable API integration
            grpc_port: Port for gRPC server (if using server mode)
            use_local_model: If True, use local YOLO model. If False, use gRPC server
            model_path: Path to YOLO model file (for local mode)
            camera_index: OpenCV camera index (None = read config.json / auto-detect)
        """
        print("🚀 Initializing Killblock Detector...")
        
        # Processing mode
        self.use_local_model = use_local_model
        self.model_path = model_path
        self.local_model = None
        
        # gRPC configuration (only if using server mode)
        self.grpc_port = grpc_port
        self.grpc_channel = None
        self.grpc_stub = None
        self.server_process = None
        
        # Initialize processing mode (local or server)
        if self.use_local_model:
            print("🔧 Using LOCAL YOLO model for processing...")
            if not YOLO_AVAILABLE:
                print("⚠️  YOLO not available. Falling back to server mode...")
                self.use_local_model = False
            else:
                self.local_model = self._load_local_model()
                if self.local_model is None:
                    print("⚠️  Failed to load local model. Falling back to server mode...")
                    self.use_local_model = False
                else:
                    # Local model loaded successfully - skip gRPC entirely
                    print("✅ Local model loaded - skipping gRPC server setup")
        
        # If not using local model, use gRPC server
        if not self.use_local_model:
            print(f"📡 Using gRPC server mode on port {grpc_port}...")
            # Start/connect to gRPC server
            if self._check_server_running():
                print(f"✅ gRPC server already running on port {self.grpc_port}")
            else:
                # Check if port is in use by another process
                if not self._check_port_available():
                    print(f"⚠️  Port {self.grpc_port} is already in use!")
                    print(f"   This usually means:")
                    print(f"   1. Another instance of grpc_block.py is running")
                    print(f"   2. Another application is using port {self.grpc_port}")
                    print(f"")
                    print(f"   Solutions:")
                    print(f"   - Close the other instance and try again")
                    print(f"   - Or let this program try to kill the process on the port")
                    
                    # Ask user or auto-kill on Windows
                    if sys.platform == 'win32':
                        print(f"\n   Attempting to free port {self.grpc_port}...")
                        if self._kill_process_on_port():
                            time.sleep(2)  # Wait for port to be released
                            if self._check_port_available():
                                print(f"✅ Port {self.grpc_port} is now available")
                            else:
                                print(f"⚠️  Port still in use. Please close the other process manually.")
                                raise RuntimeError(f"Port {self.grpc_port} is in use. Please close the other process and try again.")
                        else:
                            print(f"⚠️  Could not free port. Please close the process manually.")
                            raise RuntimeError(f"Port {self.grpc_port} is in use. Please close the other process and try again.")
                    else:
                        raise RuntimeError(f"Port {self.grpc_port} is in use. Please close the other process and try again.")
                
                print("🚀 Starting gRPC server...")
                if not self._start_grpc_server():
                    raise RuntimeError("Failed to start gRPC server")
            
            # Connect to server (only if using gRPC mode)
            if not self._connect_to_server():
                raise RuntimeError("Failed to connect to gRPC server")
        else:
            # Local model mode - no gRPC server needed
            print("✅ Local model mode - no gRPC server needed")
            self.grpc_stub = None  # Ensure it's None for local mode
        
        # Camera
        self.cap = None
        camera_cfg = _load_app_config().get("camera", {})
        self.camera_index = camera_index if camera_index is not None else camera_cfg.get("default_index", 2)
        self.camera_source = self.camera_index  # int or /dev/videoN path once resolved
        self.camera_width = camera_cfg.get("default_width", 1920)
        self.camera_height = camera_cfg.get("default_height", 1080)
        self.camera_auto_detect = camera_cfg.get("auto_detect", True)
        
        # API configuration
        app_config = _load_app_config()
        local_cfg = _load_local_config()
        api_cfg = app_config.get("api", {})
        backend_url = local_cfg.get("backend_url") or api_cfg.get("backend_url", "http://192.168.29.47:5005")
        backend_url = backend_url.rstrip("/")
        endpoints_cfg = app_config.get("endpoints", {})
        killfeed_endpoint = endpoints_cfg.get(
            "killfeed",
            "api/LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed",
        )
        self.match_id = match_id
        self.access_token = access_token
        self.api_enabled = api_enabled
        self.api_push_enabled = bool(api_enabled and access_token)
        self.api_timeout = int(api_cfg.get("timeout", 10))
        self.api_max_retries = int(api_cfg.get("max_retries", 2))
        self.api_fast_push = bool(api_cfg.get("fast_push", True))
        self.api_url = f"{backend_url}/{killfeed_endpoint}?matchId={match_id}"
        self.session_dir = None

        # HTTP session must exist before TMS verify / players list load
        self.api_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0,
            pool_block=False,
        )
        self.api_session.mount("http://", adapter)
        self.api_session.mount("https://", adapter)
        
        if self.api_enabled and not self.access_token:
            print("⚠️  WARNING: No access token provided.")
            print("   Local crops will still save, but TMS killfeed will NOT update.")
            print("   Re-run and paste your access token to push killfeed to the panel.")
        elif self.api_push_enabled:
            print(f"🌐 TMS killfeed API enabled: {self.api_url}")
            self._verify_tms_api()
            self._verify_match_active()
        
        # Detection tracking
        self.detection_sequence = 0
        self.recent_detections = set()  # Name-based duplicate prevention (API/logging)
        self.name_cooldown = {}  # name key -> last seen timestamp (3s window for API)
        self.name_cooldown_window = 8.0
        self.match_output_base = self._init_match_output_dirs()
        
        detection_cfg = app_config.get("detection", {})
        processing_cfg = app_config.get("processing", {})
        ocr_queue_cfg = app_config.get("ocr_queue", {})
        queue_cfg = app_config.get("queue", {})
        logging_cfg = app_config.get("logging", {})
        self.use_fifo_pipeline = detection_cfg.get("use_fifo_pipeline", True)
        self.capture_fps = int(detection_cfg.get("capture_fps", 30))
        self.frame_buffer_seconds = float(detection_cfg.get("frame_buffer_seconds", 3))
        self.roi_change_threshold = float(detection_cfg.get("roi_change_threshold", 1.5))
        self.dhash_max_distance = int(detection_cfg.get("dhash_max_distance", 4))
        self.yolo_heartbeat_frames = int(detection_cfg.get("yolo_heartbeat_frames", 5))
        self.ocr_queue_max = int(ocr_queue_cfg.get("max_size", 8))
        self.fast_row0_emit = bool(ocr_queue_cfg.get("fast_row0_emit", True))
        self.max_visible_slots = int(queue_cfg.get("max_visible_slots", 4))
        self.cache_ttl_seconds = float(queue_cfg.get("cache_ttl_seconds", 45))
        self.pair_cooldown_seconds = float(queue_cfg.get("pair_cooldown_seconds", 10))
        self.killfeed_log_path = logging_cfg.get("killfeed_log", "killfeed.log")
        self.killfeed_jsonl_path = logging_cfg.get("killfeed_jsonl", "killfeed_events.jsonl")
        self.fifo_pipeline = None
        self.yolo_confidence = float(detection_cfg.get("confidence_threshold", 0.20))
        self.revive_confidence = float(detection_cfg.get("revive_confidence", 0.28))
        self.save_unverified_crops = processing_cfg.get("save_unverified_crops", True)

        ocr_cfg = app_config.get("ocr", {})
        self.ocr_subprocess = None
        if self.use_fifo_pipeline and ocr_cfg.get("use_subprocess", True):
            print("🔍 OCR will run in isolated subprocess (main process skips PaddleOCR)")
            self.text_detector = None
        else:
            self.text_detector = self._init_ocr()
        
        # Ordered output: min-heap by detection number (assigned at queue time)
        self.output_heap = []
        self.heap_lock = Lock()
        self.next_expected_frame = 0
        self.next_detection_number = 0
        
        # Image hash deduplication (before OCR)
        self.saved_hashes = set()
        self.hash_lock = Lock()
        
        # Temporal smoothing for OCR results
        self.name_history = deque(maxlen=5)  # Keep last 5 detections for voting
        self.confidence_threshold = 0.6  # Minimum confidence to accept
        self.min_name_length = 2  # Minimum name length
        self.max_name_length = 20  # Maximum name length
        self.ocr_retry_attempts = 1  # One full OCR retry when validation fails (burst fights)
        
        # Known players dictionary (can be loaded from file or API)
        self.known_players = set()  # Will be populated from successful detections
        
        # TMS players list for fuzzy matching (loaded from API)
        self.tms_players_list = []  # List of canonical player names from TMS
        self.tms_players_loaded = False
        Thread(target=self._load_tms_players_list, daemon=True, name="TMSPlayersLoad").start()

        # Match-specific roster for high-accuracy OCR snapping (set via set_match_roster)
        self._match_roster_names: list = []
        self._match_roster_suffixes: dict = {}  # suffix_after_dot → full_name
        
        # Async processing pipeline (work queue in; min-heap orders saves out)
        self.detection_queue = Queue(maxsize=500)  # Large buffer for burst killfeeds
        self.api_queue = Queue(maxsize=200)  # Queue for API calls
        self.processing_lock = Lock()  # Lock for sequence number updates
        self.stop_event = Event()  # Event to signal shutdown
        
        # Thread pool for parallel processing
        # Reduced OCR workers to 1 because PaddleOCR is not fully thread-safe
        self.ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OCR")
        self.api_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="API")
        
        # OCR lock for thread safety (PaddleOCR is not thread-safe)
        self.ocr_lock = Lock()
        self.yolo_lock = Lock()
        
        # Statistics
        self.stats = {
            'detections_processed': 0,
            'api_success': 0,
            'api_failed': 0,
            'api_retries': 0,
            'queue_drops': 0,
            'ocr_errors': 0,
            'crops_saved': 0,
            'raw_crops_saved': 0,
            'crops_skipped_unverified': 0,
        }
        self.stats_lock = Lock()
        
        # Worker health monitoring
        self.worker_restart_lock = Lock()
        self.last_worker_check = time.time()
        
        # Start background workers (legacy per-crop pipeline only)
        if not self.use_fifo_pipeline:
            self._start_workers()
            self.worker_monitor = Thread(
                target=self._monitor_workers, daemon=True, name="WorkerMonitor"
            )
            self.worker_monitor.start()
        else:
            self.detection_worker = None
            self.worker_monitor = None
            # API worker still needed for TMS push
            self.api_worker = Thread(target=self._api_worker, daemon=True, name="APIWorker")
            self.api_worker.start()
            print("📋 FIFO killfeed pipeline enabled (emit once per real event)")
            if self.api_fast_push:
                print("⚡ Fast TMS push: in-memory JPEG + row-0-first OCR")
        
        print(f"🎯 YOLO confidence: killblock>={self.yolo_confidence}, revive>={self.revive_confidence}")
        if self.use_fifo_pipeline:
            print(f"📹 Capture FPS: {self.capture_fps} | OCR queue max: {self.ocr_queue_max}")
        print(f"💾 Session output: {self.match_output_base}/")
    
    def _check_server_running(self):
        """Check if gRPC server is running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex((self.GRPC_SERVER_HOST, self.grpc_port))
            sock.close()
            return result == 0
        except:
            return False
    
    def _check_port_available(self):
        """Check if port is available (not in use)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.bind((self.GRPC_SERVER_HOST, self.grpc_port))
            sock.close()
            return True
        except OSError:
            # Port is in use
            return False
        except:
            return False
    
    def _kill_process_on_port(self):
        """Kill any process using the gRPC port (Windows only)."""
        if sys.platform != 'win32':
            return False
        
        try:
            # Use netstat to find process using the port
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Find line with our port
            for line in result.stdout.split('\n'):
                if f':{self.grpc_port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            # Kill the process
                            subprocess.run(['taskkill', '/F', '/PID', pid], 
                                         capture_output=True, timeout=5)
                            print(f"🔄 Killed process {pid} using port {self.grpc_port}")
                            time.sleep(1)  # Wait for port to be released
                            return True
                        except:
                            pass
        except:
            pass
        
            return False
    
    def _start_grpc_server(self):
        """Start gRPC server in background."""
        try:
            grpc_block_path = os.path.join(os.path.dirname(__file__), 'grpc_block.py')
            if not os.path.exists(grpc_block_path):
                print(f"❌ grpc_block.py not found at {grpc_block_path}")
                return False
            
            # Double-check port is available before starting
            if not self._check_port_available():
                print(f"❌ Port {self.grpc_port} is not available. Cannot start server.")
                return False
            
            self.server_process = subprocess.Popen(
                [sys.executable, '-u', grpc_block_path, '--serve', '--model', 'best.pt', '--port', str(self.grpc_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            # Wait for server to start (check both process status and port)
            for i in range(30):  # Increased wait time to 15 seconds
                time.sleep(0.5)
                
                # Check if process died
                if self.server_process.poll() is not None:
                    # Process exited - read error output
                    try:
                        output, _ = self.server_process.communicate(timeout=1)
                        error_msg = output.decode('utf-8', errors='ignore') if output else "Unknown error"
                        if "Failed to bind" in error_msg or "port" in error_msg.lower():
                            print(f"❌ Server failed to start: Port binding error")
                            print(f"   Error: {error_msg[:200]}")
                        else:
                            print(f"❌ Server process exited unexpectedly")
                            print(f"   Output: {error_msg[:200]}")
                    except:
                        print(f"❌ Server process exited unexpectedly")
                    return False
                
                # Check if server is now running
                if self._check_server_running():
                    print(f"✅ gRPC server started on port {self.grpc_port}")
                    return True
            
            # Timeout - check one more time
            if self._check_server_running():
                print(f"✅ gRPC server started on port {self.grpc_port}")
                return True
            else:
                print(f"❌ Server start timeout - port {self.grpc_port} not responding")
                # Try to kill the process
                try:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=2)
                except:
                    try:
                        self.server_process.kill()
                    except:
                        pass
                    return False
            
        except Exception as e:
            print(f"❌ Failed to start gRPC server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _connect_to_server(self):
        """Connect to gRPC server."""
        try:
            if killfeed_detection_pb2 is None or killfeed_detection_pb2_grpc is None:
                print("❌ gRPC code not generated")
                return False
            
            options = [
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
            self.grpc_channel = grpc.insecure_channel(f'{self.GRPC_SERVER_HOST}:{self.grpc_port}', options=options)
            
            try:
                grpc.channel_ready_future(self.grpc_channel).result(timeout=5)
            except grpc.FutureTimeoutError:
                return False
            
            self.grpc_stub = killfeed_detection_pb2_grpc.KillfeedDetectionServiceStub(self.grpc_channel)
            print(f"✅ Connected to gRPC server on port {self.grpc_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def _load_local_model(self):
        """Load YOLO model for local processing."""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model file not found: {self.model_path}")
                return None
            
            print(f"📦 Loading YOLO model from {self.model_path}...")
            model = YOLO(self.model_path)
            print("✅ YOLO model loaded successfully")
            
            if hasattr(model, 'names'):
                class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
                print(f"📋 Available classes: {class_names}")
            
            return model
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _init_ocr(self):
        """Initialize OCR text detector with simplified error handling."""
        print("🔍 Initializing OCR...")
        try:
            import paddleocr
            detector = FreeFireTextDetector()
            print("✅ OCR initialized successfully")
            return detector
        except ImportError:
            print("⚠️ PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
        except (OSError, Exception) as e:
            error_msg = str(e)
            if "shm.dll" in error_msg or "WinError 127" in error_msg:
                print("❌ Visual C++ Redistributables required")
                print("   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            else:
                print(f"⚠️ OCR initialization error: {e}")
        
        print("⚠️ OCR disabled - names will be empty")
        return None
    
    def _process_frame_local(self, frame):
        """Process frame locally using YOLO model."""
        if self.local_model is None or frame is None or frame.size == 0:
            return []
        
        try:
            # Detect killblocks and revives
            detections = self._detect_killblocks_and_revives_local(frame)
            if not detections:
                return []
            
            results = []
            for detection in detections:
                try:
                    x1, y1, x2, y2 = detection['bbox']
                    cropped_image = frame[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        continue
                    
                    # Determine status with proper detection
                    status = self._determine_status_local(cropped_image, detection.get('class_name'))
                    
                    results.append({
                        'cropped_image': cropped_image,
                        'status': status,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence']
                    })
                except Exception as e:
                    continue
            
            return results
        except Exception as e:
            error_type = type(e).__name__
            print(f"⚠️ Local processing error: {error_type}")
            return []
    
    def _detect_killblocks_and_revives_local(self, image):
        """Detect killblocks and revives using local YOLO model."""
        try:
            processed_image = cv2.resize(image, (1280, 720))
            names_map = getattr(self.local_model, 'names', None)
            
            with self.yolo_lock:
                results = self.local_model.predict(
                    processed_image,
                    conf=min(self.yolo_confidence, self.revive_confidence),
                    verbose=False,
                    device='cpu',
                    imgsz=640,
                    half=False
                )
            
            if not results or len(results) == 0:
                return []
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []
            
            detections = []
            orig_size = image.shape[:2]
            
            for box in result.boxes:
                try:
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = names_map.get(class_id, f"class_{class_id}") if isinstance(names_map, dict) else f"class_{class_id}"
                    class_name_lower = class_name.lower()
                    min_conf = self.revive_confidence if class_name_lower == "revive" else self.yolo_confidence
                    
                    # Detect both killblock and revive classes
                    if (class_name_lower == "killblock" or class_name_lower == "revive") and confidence >= min_conf:
                        bbox_resized = box.xyxy[0].cpu().numpy()
                        bbox = self._convert_bbox_coords(bbox_resized, orig_size, (1280, 720))
                        detections.append({
                            'bbox': bbox, 
                            'confidence': confidence,
                            'class_name': class_name_lower
                        })
                except:
                    continue
            
            return detections
        except Exception as e:
            return []
    
    def _convert_bbox_coords(self, bbox_resized, orig_size, resize_size):
        """Convert bbox coordinates from resized to original image size."""
        x1_r, y1_r, x2_r, y2_r = bbox_resized
        orig_h, orig_w = orig_size
        resize_w, resize_h = resize_size
        
        x1 = max(0, min(int(x1_r * orig_w / resize_w), orig_w - 1))
        y1 = max(0, min(int(y1_r * orig_h / resize_h), orig_h - 1))
        x2 = max(x1 + 1, min(int(x2_r * orig_w / resize_w), orig_w))
        y2 = max(y1 + 1, min(int(y2_r * orig_h / resize_h), orig_h))
        return [x1, y1, x2, y2]
    
    def _determine_status_local(self, cropped_image, detected_class=None):
        """
        Determine status from cropped image with proper detection logic:
        1. First check for revive class from YOLO model
        2. Then check victim text color (red = kill, white = gun knockout)
        3. Use green color as fallback for revive
        
        Args:
            cropped_image: Cropped killfeed image
            detected_class: Class detected by YOLO ('killblock' or 'revive')
        
        Returns:
            str: 'kill', 'gun knockout', or 'revive'
        """
        # Priority 1: Check if YOLO detected revive class
        if detected_class == 'revive':
            # Validate with green color check as fallback
            has_green = self._check_green_color_in_image(cropped_image)
            if has_green:
                print(f"   🟢 Status: revive (YOLO + green color detected)")
                return 'revive'
            # If no green, still trust YOLO revive detection
            print(f"   🟢 Status: revive (YOLO model detected)")
            return 'revive'
        
        # Priority 2: Analyze victim text color
        try:
            if self.text_detector:
                # Extract victim text region and analyze color
                victim_color = self._analyze_victim_text_color(cropped_image)
                
                if victim_color == 'red':
                    print(f"   🔴 Status: kill (victim text color: red)")
                    return 'kill'
                elif victim_color == 'white':
                    print(f"   ⚪ Status: gun knockout (victim text color: white)")
                    return 'gun knockout'
                elif victim_color == 'green':
                    # Green text indicates revive (fallback)
                    print(f"   🟢 Status: revive (victim text color: green - fallback)")
                    return 'revive'
                else:
                    # If color detection failed, check for green as revive fallback
                    if self._check_green_color_in_image(cropped_image):
                        print(f"   🟢 Status: revive (green color detected - fallback)")
                        return 'revive'
        except Exception as e:
            pass
        
        # Default: gun knockout
        print(f"   ⚪ Status: gun knockout (default)")
        return 'gun knockout'
    
    def _analyze_victim_text_color(self, cropped_image):
        """
        Analyze the color of victim name text in cropped killfeed image.
        
        Returns:
            str: 'red', 'white', 'green', or 'unknown'
        """
        try:
            from killfeed import vision
            return vision.analyze_victim_text_color(cropped_image)
        except Exception:
            return 'unknown'
    
    def _check_green_color_in_image(self, cropped_image):
        """
        Check if green color is present in the image (for revive detection fallback).
        
        Returns:
            bool: True if green color detected
        """
        try:
            if cropped_image is None or cropped_image.size == 0:
                return False
            
            hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
            
            # Green color range
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
            
            green_pixels = cv2.countNonZero(mask_green)
            total_pixels = cropped_image.shape[0] * cropped_image.shape[1]
            
            if total_pixels == 0:
                return False
            
            green_ratio = green_pixels / total_pixels
            # Threshold: at least 0.5% green pixels
            return green_ratio >= 0.005
        except:
            return False
    
    def _process_frame_via_grpc(self, frame):
        """Send frame to gRPC server and receive detection results."""
        if self.grpc_stub is None or frame is None or frame.size == 0:
            return []
        
        try:
            # Encode and send frame as JPEG (much smaller than PNG, reduces memory usage)
            # Use quality 85 for good balance between size and quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, frame_encoded = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = frame_encoded.tobytes()
            
            request = killfeed_detection_pb2.ProcessFrameRequest(
                frame_image=frame_bytes,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0]
            )
            
            # Explicitly delete encoded frame to free memory
            del frame_encoded, frame_bytes
            
            # Get response with increased timeout to handle slow processing
            # Increased from 30s to 60s to prevent timeouts during heavy load
            response = self.grpc_stub.ProcessFrame(request, timeout=60)
            
            # Delete request to free memory
            del request
            
            if not response.success:
                return []
            
            # Convert response to list
            results = []
            for detection in response.detections:
                try:
                    # Decode JPEG image (server now sends JPEG instead of PNG)
                    cropped_array = np.frombuffer(detection.cropped_image, dtype=np.uint8)
                    cropped_image = cv2.imdecode(cropped_array, cv2.IMREAD_COLOR)
                    
                    # Delete array immediately after decoding
                    del cropped_array
                    
                    if cropped_image is None or cropped_image.size == 0:
                        continue
                    
                    results.append({
                        'cropped_image': cropped_image,
                        'status': detection.status,
                        'bbox': [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2],
                        'confidence': detection.confidence
                    })
                except Exception as e:
                    # Skip invalid detections
                    continue
            
            return results
        except grpc.RpcError as e:
            error_code = e.code()
            if error_code == grpc.StatusCode.UNAVAILABLE:
                print("⚠️ gRPC server unavailable, attempting to reconnect...")
                try:
                    self._connect_to_server()
                except:
                    pass  # Will retry on next frame
            elif error_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                print("⚠️ gRPC request timeout (60s exceeded), will retry on next frame...")
                # Try to reconnect if connection might be stale
                try:
                    self._connect_to_server()
                except:
                    pass
            elif error_code == grpc.StatusCode.UNAVAILABLE:
                print("⚠️ gRPC server unavailable, attempting to reconnect...")
                try:
                    self._connect_to_server()
                except:
                    pass
            else:
                print(f"⚠️ gRPC error ({error_code}): {str(e)[:100]}")
            return []
        except Exception as e:
            # Log but don't stop - return empty results and continue
            error_type = type(e).__name__
            print(f"⚠️ gRPC processing error: {error_type}")
            return []
    
    def _validate_image_for_ocr(self, image):
        """Validate and normalize image format for PaddleOCR."""
        if image is None or image.size == 0:
            return None
        
        try:
            # Ensure it's a numpy array
            if not isinstance(image, np.ndarray):
                return None
            
            # Handle different image formats
            if len(image.shape) == 2:
                # Grayscale - convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    # RGBA - convert to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.shape[2] != 3:
                    return None
            else:
                return None
            
            # Check dimensions
            height, width = image.shape[:2]
            if height < 10 or width < 10:
                return None
            
            # PaddleOCR has size limits - resize if too large
            max_dimension = 2000  # PaddleOCR works best under 2000px
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Ensure uint8 type
            if image.dtype != np.uint8:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    # Normalize to 0-255 range
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Ensure values are in valid range
            image = np.clip(image, 0, 255)
            
            return image
        except Exception as e:
            print(f"⚠️ Image validation error: {e}")
            return None
    
    def _preprocess_image_for_ocr(self, image, strategy='default', target_dpi=300):
        """
        Robust preprocessing with multiple strategies for maximum OCR accuracy.
        Returns validated image in correct format for PaddleOCR.
        """
        # First validate the input image
        image = self._validate_image_for_ocr(image)
        if image is None:
            return None
        
        try:
            height, width = image.shape[:2]
            processed = image.copy()
            
            # Strategy 1: Moderate upscaling (don't go too large - PaddleOCR has limits)
            if target_dpi > 0:
                # Limit upscaling to avoid memory issues
                scale_factor = min(2.5, max(1.5, target_dpi / 96.0))  # 1.5x to 2.5x max
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                # Ensure we don't exceed max dimensions
                if new_width <= 2000 and new_height <= 2000:
                    processed = cv2.resize(processed, (new_width, new_height), 
                                         interpolation=cv2.INTER_CUBIC)
            
            # Strategy 2: High contrast enhancement
            if strategy in ['default', 'high_contrast']:
                try:
                    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    l_enhanced = clahe.apply(l)
                    lab_enhanced = cv2.merge([l_enhanced, a, b])
                    processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                except:
                    pass  # Fallback to original if CLAHE fails
            
            # Strategy 3: Black and white conversion
            if strategy == 'bw':
                try:
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    # Adaptive thresholding for better text extraction
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                except:
                    pass
            
            # Strategy 4: Sharpening
            if strategy in ['default', 'sharp']:
                try:
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                    processed = cv2.filter2D(processed, -1, kernel * 0.12)
                except:
                    pass
            
            # Strategy 5: Denoising
            if strategy == 'denoise':
                try:
                    processed = cv2.fastNlMeansDenoisingColored(processed, None, 8, 8, 7, 21)
                except:
                    pass
            
            # Final validation before returning
            processed = self._validate_image_for_ocr(processed)
            return processed
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            # Return validated original if preprocessing fails
            return self._validate_image_for_ocr(image)
    
    def _normalize_name(self, name):
        """Normalize name using regex rules and known players dictionary."""
        if not name or len(name) < self.min_name_length:
            return ""
        
        # Remove common OCR artifacts
        name = re.sub(r'[^\w\d._-]', '', name)  # Keep alphanumeric, dots, underscores, hyphens
        name = re.sub(r'_{2,}', '_', name)  # Replace multiple underscores with single
        name = name.strip('._-')  # Remove leading/trailing special chars
        
        # Fix common OCR mistakes
        replacements = {
            '0': 'O',  # In names, 0 is usually O
            '1': 'I',  # In names, 1 is usually I
            '5': 'S',  # Sometimes 5 is S
            '8': 'B',  # Sometimes 8 is B
        }
        # Only apply if it makes sense (not at start/end of numbers)
        for wrong, correct in replacements.items():
            # Replace standalone wrong chars (not part of numbers)
            name = re.sub(rf'(?<!\d){wrong}(?!\d)', correct, name)
        
        # Check against known players (fuzzy matching)
        if self.known_players:
            name_upper = name.upper()
            for known in self.known_players:
                known_upper = known.upper()
                # Exact match
                if name_upper == known_upper:
                    return known
                # Close match (1-2 char difference)
                if len(name_upper) >= 3 and len(known_upper) >= 3:
                    if self._levenshtein_distance(name_upper, known_upper) <= 2:
                        return known
        
        # Validate length
        if len(name) < self.min_name_length or len(name) > self.max_name_length:
            return ""
        
        return name
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance for fuzzy matching."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _verify_tms_api(self):
        """Ping TMS with token before detection starts."""
        if not self.access_token:
            return
        if not _is_valid_jwt(self.access_token):
            print("❌ TMS token in local_config.json is corrupted (invalid JWT)")
            print("   Login at http://192.168.29.47:5173 → DevTools → copy Bearer token")
            print("   Paste into local_config.json → access_token → restart")
            self.api_push_enabled = False
            return
        try:
            app_config = _load_app_config()
            token_ep = app_config.get("endpoints", {}).get(
                "user_from_token", "api/User/GetUserFromToken"
            )
            backend = _backend_url_from_config()
            url = f"{backend}/{token_ep.lstrip('/')}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.api_session.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                print("✅ TMS API token verified")
            elif response.status_code == 401:
                print("❌ TMS token rejected (401) — login at http://192.168.29.47:5173")
                print("   Copy fresh token from browser → update local_config.json → restart")
                self.api_push_enabled = False
            else:
                print(f"⚠️ TMS API check: HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            print("❌ TMS API unreachable (timeout) — check backend_url in local_config.json")
            self.api_push_enabled = False
        except Exception as e:
            print(f"⚠️ TMS API check failed: {type(e).__name__}")

    def _parse_api_error_message(self, response) -> str:
        """Extract human-readable error from TMS JSON response."""
        try:
            body = response.json()
            if isinstance(body, dict):
                msg = body.get("message") or body.get("error", {}).get("detail")
                if msg:
                    return str(msg)
        except Exception:
            pass
        return (getattr(response, "text", None) or "")[:200]

    def _verify_match_active(self):
        """Warn if TMS match is completed — killfeed POST will be rejected with HTTP 400."""
        if not self.api_push_enabled or not self.match_id:
            return
        try:
            app_config = _load_app_config()
            backend = _backend_url_from_config()
            ep = app_config.get("endpoints", {}).get(
                "match_states", "api/LeagueMatch/getMatchStates/details"
            )
            url = f"{backend}/{ep.lstrip('/')}?matchId={self.match_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.api_session.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                print(f"⚠️ Could not verify match status (HTTP {response.status_code})")
                return
            data = response.json()
            details = (data.get("data") or {}).get("matchDetails") or {}
            status = (details.get("status") or "").strip()
            name = details.get("name") or self.match_id
            if status.lower() in ("completed", "complete", "finished", "ended"):
                print(f"❌ TMS match is '{status}' — killfeed API will reject all posts (HTTP 400)")
                print(f"   Match: {name} ({self.match_id})")
                print("   Fix: In TMS panel set match to Live/In Progress, OR update match_id in local_config.json")
                self.api_push_enabled = False
            elif status:
                print(f"✅ TMS match status: {status} ({name})")
        except requests.exceptions.Timeout:
            print("⚠️ Match status check timed out — continuing anyway")
        except Exception as e:
            print(f"⚠️ Match status check failed: {type(e).__name__}")

    def _load_tms_players_list(self):
        """Load active TMS players list from API for fuzzy matching."""
        if not self.api_enabled or not self.access_token or not self.match_id:
            print("⚠️  TMS players list not loaded: API disabled or missing credentials")
            return

        try:
            app_config = _load_app_config()
            backend_url = _backend_url_from_config()
            team_ep = app_config.get("endpoints", {}).get(
                "team_players",
                "api/LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players",
            )
            api_url = f"{backend_url}/{team_ep.lstrip('/')}?matchId={self.match_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            response = self.api_session.get(api_url, headers=headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                
                # Extract player names from response
                players = set()
                if isinstance(data, dict) and 'data' in data:
                    for team in data.get('data', []):
                        # Extract all player fields (player1, player2, etc.)
                        for key, value in team.items():
                            if key.startswith('player') and value and isinstance(value, str):
                                players.add(value.strip())
                
                self.tms_players_list = list(players)
                self.tms_players_loaded = True
                print(f"✅ Loaded {len(self.tms_players_list)} TMS players for fuzzy matching")
                if len(self.tms_players_list) > 0:
                    print(f"   Sample players: {', '.join(list(self.tms_players_list)[:5])}")
            else:
                print(f"⚠️  Failed to load TMS players list: HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️  Error loading TMS players list: {type(e).__name__}")
            # Continue without TMS players list - will use raw OCR names
    
    # ------------------------------------------------------------------
    # Match-roster snapping  (zero-latency — ~60 rapidfuzz comparisons)
    # ------------------------------------------------------------------

    def set_match_roster(self, player_names):
        """Load canonical player names for the current match.

        Call once per match with the full list (all teams + subs).
        Enables high-accuracy OCR correction without adding latency.

        Args:
            player_names: iterable of strings like ['K9.HUNNYSUNY', 'RNTX.ARSH17', ...]
        """
        names = [n.strip().upper() for n in player_names if n and n.strip()]
        self._match_roster_names = names
        # Pre-build suffix → full_name map for prefix-mangled OCR output
        self._match_roster_suffixes = {}
        for name in names:
            if '.' in name:
                suffix = name.split('.', 1)[1]
                if len(suffix) >= 3:
                    self._match_roster_suffixes[suffix] = name
        # Also merge into tms_players_list so existing _fuzzy_match_player_name also benefits
        merged = set(self.tms_players_list) | set(names)
        self.tms_players_list = list(merged)
        self.tms_players_loaded = True
        print(f"✅ Match roster loaded: {len(names)} players")

    def _snap_to_roster(self, name: str) -> str:
        """Correct OCR noise by snapping name to the closest known match player.

        Three-stage strategy (fastest first):
          1. Exact match        — O(1)
          2. WRatio full-name   — handles char substitutions, missing dot, extra chars
          3. Suffix-only match  — handles mangled team prefix (e.g. 'K8.' → 'K9.')

        Returns canonical player name on a confident hit, else the original string.
        """
        if not name or not RAPIDFUZZ_AVAILABLE:
            return name

        roster = self._match_roster_names
        if not roster:
            # Fall back to TMS API list if match roster not set yet
            if self.tms_players_list:
                roster = [n.upper() for n in self.tms_players_list]
            else:
                return name

        name_upper = name.strip().upper()

        # Stage 1: exact hit — free
        if name_upper in roster:
            return name_upper

        # Stage 2: WRatio full-name (handles substitution + partial alignment)
        result = process.extractOne(name_upper, roster, scorer=fuzz.WRatio)
        if result and result[1] >= 75:
            return result[0]

        # Stage 3: suffix-only match (catches "K8.HUNNYSUNY" → "K9.HUNNYSUNY")
        suffixes = self._match_roster_suffixes
        if suffixes:
            if '.' in name_upper:
                ocr_suffix = name_upper.split('.', 1)[1]
            else:
                ocr_suffix = name_upper  # no dot at all — try raw suffix match
            if len(ocr_suffix) >= 3:
                suf_result = process.extractOne(
                    ocr_suffix, list(suffixes.keys()), scorer=fuzz.WRatio
                )
                if suf_result and suf_result[1] >= 82:
                    return suffixes[suf_result[0]]

        return name  # no confident match — keep raw OCR

    def _fuzzy_match_player_name(self, ocr_name: str, min_similarity: float = 0.65) -> tuple:
        """
        Fuzzy match OCR-extracted name against TMS players list.
        
        Args:
            ocr_name: Name extracted from OCR
            min_similarity: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            tuple: (canonical_name, similarity_score, matched)
                - canonical_name: Matched TMS player name or original OCR name
                - similarity_score: Similarity score (0.0-1.0)
                - matched: True if reliable match found, False otherwise
        """
        if not ocr_name or len(ocr_name) < 2:
            return ocr_name, 0.0, False
        
        if not self.tms_players_loaded or len(self.tms_players_list) == 0:
            # No TMS players list available - return raw OCR name
            return ocr_name, 0.0, False
        
        ocr_name_lower = ocr_name.lower().strip()
        
        # Use RapidFuzz if available (faster and more accurate)
        if RAPIDFUZZ_AVAILABLE:
            try:
                # Find best match using rapidfuzz
                best_match = process.extractOne(
                    ocr_name_lower,
                    [p.lower() for p in self.tms_players_list],
                    scorer=fuzz.WRatio
                )
                
                if best_match:
                    matched_name_lower, score, idx = best_match
                    similarity = score / 100.0  # Convert to 0.0-1.0
                    canonical_name = self.tms_players_list[idx]
                    
                    if similarity >= min_similarity:
                        return canonical_name, similarity, True
                    else:
                        # Low confidence match - log and return raw OCR
                        print(f"⚠️  Low-confidence match: '{ocr_name}' → '{canonical_name}' (similarity: {similarity:.2f} < {min_similarity})")
                        return ocr_name, similarity, False
            except Exception as e:
                print(f"⚠️  RapidFuzz matching error: {type(e).__name__}")
        
        # Fallback: Use Levenshtein distance
        best_match = None
        best_similarity = 0.0
        
        for tms_name in self.tms_players_list:
            tms_name_lower = tms_name.lower().strip()
            
            # Calculate similarity using Levenshtein distance
            max_len = max(len(ocr_name_lower), len(tms_name_lower))
            if max_len == 0:
                continue
            
            distance = self._levenshtein_distance(ocr_name_lower, tms_name_lower)
            similarity = 1.0 - (distance / max_len)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = tms_name
        
        if best_match and best_similarity >= min_similarity:
            return best_match, best_similarity, True
        else:
            if best_match:
                print(f"⚠️  Low-confidence match: '{ocr_name}' → '{best_match}' (similarity: {best_similarity:.2f} < {min_similarity})")
            return ocr_name, best_similarity, False
    
    def _extract_names_with_ocr(self, cropped_image, preprocessing_strategy='default'):
        """Extract names using PaddleOCR with specified preprocessing and robust error handling."""
        if cropped_image is None or cropped_image.size == 0 or self.text_detector is None:
            return None, None, 0.0
        
        # Use lock for thread-safe OCR (PaddleOCR is not thread-safe)
        with self.ocr_lock:
            return self._extract_names_with_ocr_unsafe(cropped_image, preprocessing_strategy)
    
    def _extract_names_with_ocr_unsafe(self, cropped_image, preprocessing_strategy='default'):
        """Internal OCR extraction (must be called with ocr_lock held)."""
        try:
            # Preprocess with specified strategy
            preprocessed = self._preprocess_image_for_ocr(cropped_image, strategy=preprocessing_strategy)
            
            if preprocessed is None:
                return None, None, 0.0
            
            # Validate preprocessed image
            if preprocessed.size == 0:
                return None, None, 0.0
            
            # Save to temp file with error handling
            temp_path = None
            try:
                temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                
                # Ensure image is valid before writing
                if not isinstance(preprocessed, np.ndarray):
                    return None, None, 0.0
                
                # Write with validation
                success = cv2.imwrite(temp_path, preprocessed, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                if not success or not os.path.exists(temp_path):
                    return None, None, 0.0
                
                # Run OCR with error handling
                results = self.text_detector.process_image(temp_path)
                
                if not results or not isinstance(results, dict):
                    return None, None, 0.0
                
                # Extract names and confidences
                # NOTE: This method uses classify_text_positions which may not preserve strict left-to-right
                # We should avoid using this fallback method, but if we do, we'll extract based on position
                killer_dict = results.get("killer", {})
                victim_dict = results.get("victim", {})
                
                killer = ""
                killer_conf = 0.0
                if isinstance(killer_dict, dict):
                    killer = killer_dict.get("text", "").strip()
                    killer_conf = float(killer_dict.get("confidence", 0.0))
                
                victim = ""
                victim_conf = 0.0
                if isinstance(victim_dict, dict):
                    victim = victim_dict.get("text", "").strip()
                    victim_conf = float(victim_dict.get("confidence", 0.0))
                
                # Calculate average confidence
                avg_conf = (killer_conf + victim_conf) / 2.0 if (killer or victim) else 0.0
                
                # Return in order: killer (left) → victim (right)
                # NOTE: This fallback method is deprecated - use extract_killfeed_sequence instead
                return killer, victim, avg_conf
                
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            
        except RuntimeError as e:
            # PaddleOCR specific runtime errors - try fallback
            error_msg = str(e).lower()
            if "unknown exception" in error_msg or "runtime" in error_msg:
                # Track error but don't spam console
                with self.stats_lock:
                    self.stats['ocr_errors'] += 1
                
                # Try with original image (no preprocessing) as fallback
                try:
                    original_validated = self._validate_image_for_ocr(cropped_image)
                    if original_validated is not None:
                        temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                        os.close(temp_fd)
                        cv2.imwrite(temp_path, original_validated, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        results = self.text_detector.process_image(temp_path)
                        os.remove(temp_path)
                        
                        if results and isinstance(results, dict):
                            killer_dict = results.get("killer", {})
                            victim_dict = results.get("victim", {})
                            killer = killer_dict.get("text", "").strip() if isinstance(killer_dict, dict) else ""
                            victim = victim_dict.get("text", "").strip() if isinstance(victim_dict, dict) else ""
                            killer_conf = float(killer_dict.get("confidence", 0.0)) if isinstance(killer_dict, dict) else 0.0
                            victim_conf = float(victim_dict.get("confidence", 0.0)) if isinstance(victim_dict, dict) else 0.0
                            avg_conf = (killer_conf + victim_conf) / 2.0 if (killer or victim) else 0.0
                            return killer, victim, avg_conf
                except:
                    pass
            
            return None, None, 0.0
            
        except Exception as e:
            # Track error but suppress verbose output
            with self.stats_lock:
                self.stats['ocr_errors'] += 1
            
            # Only print error for default strategy to avoid spam
            if preprocessing_strategy == 'default':
                error_type = type(e).__name__
                # Suppress "Unknown exception" spam - it's a known PaddleOCR issue
                if "unknown" not in str(e).lower():
                    print(f"⚠️ OCR error: {error_type}")
            return None, None, 0.0
    
    def _extract_names(self, cropped_image):
        """
        Robust name extraction with STRICT left-to-right ordering (killer → victim).
        Uses sequential text reading to ensure correct sequence and prevents swapping.
        Applies fuzzy matching against TMS players list for canonical name replacement.
        """
        if cropped_image is None or cropped_image.size == 0:
            return "", "", 0.0
        
        if self.text_detector is None:
            return "", "", 0.0
        
        # Use left-to-right extraction — pass numpy array directly (no temp file I/O)
        try:
            result = self.text_detector.extract_killfeed_sequence(cropped_image)
            killer_raw = result.get('killer', '')
            victim_raw = result.get('victim', '')
            confidence = result.get('confidence', 0.0)

            if not killer_raw or not victim_raw:
                if not killer_raw and not victim_raw:
                    print(f"⚠️  Left-to-right extraction failed - No text regions detected (OCR found 0 names)")
                elif not killer_raw:
                    print(f"⚠️  Left-to-right extraction failed - Killer name missing (victim: '{victim_raw}')")
                else:
                    print(f"⚠️  Left-to-right extraction failed - Victim name missing (killer: '{killer_raw}')")
            else:
                killer_raw = self._normalize_name(killer_raw)
                victim_raw = self._normalize_name(victim_raw)

                if killer_raw.upper() == victim_raw.upper():
                    print(f"⚠️  Left-to-right extraction failed - Names are identical: '{killer_raw}' == '{victim_raw}'")
                elif len(killer_raw) < self.min_name_length:
                    print(f"⚠️  Left-to-right extraction failed - Killer name too short: '{killer_raw}' (min: {self.min_name_length})")
                elif len(victim_raw) < self.min_name_length:
                    print(f"⚠️  Left-to-right extraction failed - Victim name too short: '{victim_raw}' (min: {self.min_name_length})")
                else:
                    killer_canonical, killer_sim, killer_matched = self._fuzzy_match_player_name(killer_raw)
                    victim_canonical, victim_sim, victim_matched = self._fuzzy_match_player_name(victim_raw)

                    if killer_matched and killer_canonical != killer_raw:
                        print(f"   ✅ Killer matched: '{killer_raw}' → '{killer_canonical}' (similarity: {killer_sim:.2f})")
                    if victim_matched and victim_canonical != victim_raw:
                        print(f"   ✅ Victim matched: '{victim_raw}' → '{victim_canonical}' (similarity: {victim_sim:.2f})")

                    killer = killer_canonical
                    victim = victim_canonical

                    smoothed_killer, smoothed_victim, smoothed_conf = self._apply_temporal_smoothing(
                        killer, victim, confidence
                    )

                    if smoothed_killer and smoothed_victim and smoothed_killer.upper() != smoothed_victim.upper():
                        return smoothed_killer, smoothed_victim, smoothed_conf
                    print(f"⚠️  Left-to-right extraction failed - Temporal smoothing returned invalid result (killer: '{smoothed_killer}', victim: '{smoothed_victim}')")
        except Exception as e:
            print(f"⚠️  Error in left-to-right extraction: {type(e).__name__}: {str(e)}")
            if hasattr(self, 'debug_mode') and self.debug_mode:
                import traceback
                traceback.print_exc()

        return "", "", 0.0
    
    def _finalize_fifo_names(self, killer_raw, victim_raw, confidence):
        """Normalize + fuzzy-match raw OCR from subprocess (no Paddle in main thread)."""
        killer_raw = (killer_raw or "").strip()
        victim_raw = (victim_raw or "").strip()
        confidence = float(confidence or 0.0)
        if not killer_raw or not victim_raw:
            return "", "", 0.0
        killer_raw = self._normalize_name(killer_raw)
        victim_raw = self._normalize_name(victim_raw)
        if (
            len(killer_raw) < self.min_name_length
            or len(victim_raw) < self.min_name_length
            or killer_raw.upper() == victim_raw.upper()
        ):
            return "", "", 0.0
        killer, killer_sim, killer_matched = self._fuzzy_match_player_name(killer_raw)
        victim, victim_sim, victim_matched = self._fuzzy_match_player_name(victim_raw)
        # Second-pass roster snap: catches OCR noise not fixed by _fuzzy_match_player_name
        killer_snapped = self._snap_to_roster(killer)
        victim_snapped = self._snap_to_roster(victim)
        if killer_snapped != killer:
            print(f"   ✅ Killer snapped: '{killer}' → '{killer_snapped}'")
            killer = killer_snapped
        elif killer_matched and killer != killer_raw:
            print(f"   ✅ Killer matched: '{killer_raw}' → '{killer}' (similarity: {killer_sim:.2f})")
        if victim_snapped != victim:
            print(f"   ✅ Victim snapped: '{victim}' → '{victim_snapped}'")
            victim = victim_snapped
        elif victim_matched and victim != victim_raw:
            print(f"   ✅ Victim matched: '{victim_raw}' → '{victim}' (similarity: {victim_sim:.2f})")
        return killer, victim, confidence
    
    def _apply_temporal_smoothing(self, killer, victim, confidence):
        """
        Apply temporal smoothing using voting across recent frames.
        CRITICAL: Preserves strict left-to-right order (killer, victim) - NEVER swaps.
        High-confidence detections are carried forward.
        """
        # Add current detection to history (preserving order)
        detection = {
            'killer': killer,  # Leftmost (first)
            'victim': victim,  # Rightmost (second)
            'confidence': confidence,
            'timestamp': time.time()
        }
        self.name_history.append(detection)
        
        # If confidence is high, use it directly (preserving order)
        if confidence >= self.confidence_threshold:
            # Update known players
            self.known_players.add(killer)
            self.known_players.add(victim)
            # Return in strict order: killer (left) → victim (right)
            return killer, victim, confidence
        
        # If we have history, vote on most common names (preserving order)
        if len(self.name_history) >= 2:
            # Count occurrences of each name in their respective positions
            # CRITICAL: Vote separately for killer position and victim position
            # This preserves left-to-right order and prevents swapping
            killer_votes = {}  # Votes for leftmost position
            victim_votes = {}  # Votes for rightmost position
            
            for hist in self.name_history:
                k = hist['killer']  # Leftmost name
                v = hist['victim']  # Rightmost name
                conf = hist['confidence']
                
                # Weight by confidence
                weight = conf
                
                # Vote for names in their respective positions (NO SWAPPING)
                killer_votes[k] = killer_votes.get(k, 0) + weight  # Left position
                victim_votes[v] = victim_votes.get(v, 0) + weight  # Right position
            
            # Get most voted names in their respective positions
            if killer_votes and victim_votes:
                best_killer = max(killer_votes.items(), key=lambda x: x[1])[0]  # Leftmost
                best_victim = max(victim_votes.items(), key=lambda x: x[1])[0]  # Rightmost
                
                # Use voted names if they're valid (preserving order)
                if (best_killer and best_victim and 
                    len(best_killer) >= self.min_name_length and 
                    len(best_victim) >= self.min_name_length):
                    
                    # Update known players
                    self.known_players.add(best_killer)
                    self.known_players.add(best_victim)
                    
                    # Return with boosted confidence (preserving strict order)
                    return best_killer, best_victim, min(1.0, confidence + 0.1)
        
        # Fallback: return current detection if valid (preserving order)
        if killer and victim:
            return killer, victim, confidence
        
        return "", "", 0.0
    
    def _init_match_output_dirs(self):
        """Create a fresh session folder for each run — new game = new folder."""
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join("sessions", f"match_{self.match_id}_{ts}")
        self.session_dir = base

        for folder in ("KILL", "KNOCK", "REVIVE", "ELIMINATE", "CROPS"):
            os.makedirs(os.path.join(base, folder), exist_ok=True)

        self.cropkillblock_dir = os.path.join(base, "cropkillblock")
        os.makedirs(self.cropkillblock_dir, exist_ok=True)

        print(f"📁 New session folder: {base}/")
        print(f"   ├── KILL/ KNOCK/ REVIVE/ ELIMINATE/ CROPS/")
        print(f"   └── cropkillblock/")
        return base
    
    def _status_to_save_folder(self, status):
        """Map classified status to folder name and filename label."""
        normalized = (status or "").lower().strip()
        if normalized == "revive" or "revive" in normalized:
            return "REVIVE", "REVIVE"
        if normalized in ("gun knockout", "knockout", "knock", "knocked"):
            return "KNOCK", "KNOCK"
        if "knockout" in normalized or "-knockout" in normalized:
            return "KNOCK", "KNOCK"
        if normalized == "kill":
            return "KILL", "KILL"
        if "kill" in normalized or "elimination" in normalized:
            return "KILL", "KILL"
        if normalized in ("eliminate", "eliminated", "elimination"):
            return "ELIMINATE", "ELIMINATE"
        return None, None
    
    def _is_verified_killfeed(self, result):
        """True only when OCR names and status classification are both complete."""
        if not result.get('names_validated'):
            return False
        killer = result.get('killer_name') or ""
        victim = result.get('victim_name') or ""
        if not killer or not victim or killer == "UNKNOWN" or victim == "UNKNOWN":
            return False
        folder, _ = self._status_to_save_folder(result.get('status'))
        return folder is not None
    
    def _mirror_crop_to_legacy_folder(self, image, filename):
        """Also save crop to cropkillblock/ for backward compatibility."""
        try:
            filepath = os.path.join(self.cropkillblock_dir, filename)
            cv2.imwrite(filepath, image)
            return filepath
        except Exception:
            return None

    def _save_raw_crop(self, cropped_image, detection_number, status):
        """Save YOLO crop immediately so killfeeds appear even before OCR finishes."""
        if cropped_image is None or cropped_image.size == 0:
            return None
        try:
            output_dir = os.path.join(self.match_output_base, "CROPS")
            os.makedirs(output_dir, exist_ok=True)
            _, status_label = self._status_to_save_folder(status)
            status_label = status_label or "UNKNOWN"
            filename = f"{detection_number + 1:03d}_{status_label}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cropped_image)
            self._mirror_crop_to_legacy_folder(cropped_image, filename)
            with self.stats_lock:
                self.stats['raw_crops_saved'] = self.stats.get('raw_crops_saved', 0) + 1
            print(
                f"📸 Raw crop saved: {filepath} "
                f"(also in {self.cropkillblock_dir}/{filename})"
            )
            return filepath
        except Exception as e:
            print(f"⚠️ Raw crop save error: {e}")
            return None
    
    def _save_image(self, image, filename, status_folder):
        """Save verified crop under match_x/<STATUS>/filename."""
        try:
            output_dir = os.path.join(self.match_output_base, status_folder)
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, image)
            return filepath
        except Exception as e:
            print(f"⚠️ Save error: {e}")
            return None
    
    def _is_duplicate_frame(self, crop_image):
        """Layer 2 duplicate check: identical crop pixels already queued/processed."""
        if crop_image is None or crop_image.size == 0:
            return True
        img_hash = hashlib.md5(crop_image.tobytes()).hexdigest()
        with self.hash_lock:
            if img_hash in self.saved_hashes:
                return True
            self.saved_hashes.add(img_hash)
            return False
    
    def _fallback_result_payload(self, result):
        """Build save payload when OCR/worker path fails; image still saved with UNKNOWN names."""
        return {
            'cropped_image': result.get('cropped_image'),
            'status': result.get('status', 'UNKNOWN'),
            'detection_number': result.get('detection_number'),
            'killer_name': 'UNKNOWN',
            'victim_name': 'UNKNOWN',
            'confidence': 0.0,
            'suspect_swap': False,
            'names_validated': False,
        }
    
    def _sanitize_filename_part(self, name):
        """Sanitize a name segment for filesystem-safe filenames (keeps dots in player tags)."""
        if not name:
            return "UNKNOWN"
        safe = re.sub(r'[<>:"/\\|?*]', '', str(name).strip())
        safe = re.sub(r'\s+', '_', safe)
        safe = re.sub(r'_+', '_', safe).strip('._-')
        return safe or "UNKNOWN"
    
    def _push_result(self, frame_number, result):
        """Push processed detection to min-heap and flush in-order saves."""
        with self.heap_lock:
            heapq.heappush(self.output_heap, (frame_number, result))
            self._flush_heap()
    
    def _flush_heap(self):
        """Flush heap entries in strict detection-number order. Caller must hold heap_lock."""
        while self.output_heap and self.output_heap[0][0] == self.next_expected_frame:
            frame_num, result = heapq.heappop(self.output_heap)
            self._save_kill_block(result, frame_num)
            self.next_expected_frame += 1
    
    def _is_name_cooldown_duplicate(self, killer_name, victim_name, status):
        """Layer 1 duplicate check: same event within cooldown window (API/logging only)."""
        key = f"{killer_name}_{victim_name}_{status}"
        reverse_key = f"{victim_name}_{killer_name}_{status}"
        now = time.time()
        for cooldown_key in (key, reverse_key):
            last_seen = self.name_cooldown.get(cooldown_key)
            if last_seen is not None and (now - last_seen) < self.name_cooldown_window:
                return True
        self.name_cooldown[key] = now
        if len(self.name_cooldown) > 200:
            cutoff = now - self.name_cooldown_window
            self.name_cooldown = {
                k: v for k, v in self.name_cooldown.items() if v >= cutoff
            }
        return False
    
    def _save_kill_block(self, result, detection_number, skip_api_push=False):
        """Save killfeed crops into match status subfolders (verified or unverified)."""
        cropped_image = result.get('cropped_image')
        if cropped_image is None or cropped_image.size == 0:
            return False
        
        killer_name = result.get('killer_name') or "UNKNOWN"
        victim_name = result.get('victim_name') or "UNKNOWN"
        status = result.get('status', 'UNKNOWN')
        confidence = float(result.get('confidence', 0.0))
        suspect_swap = result.get('suspect_swap', False)
        names_validated = result.get('names_validated', False)
        verified = self._is_verified_killfeed(result)
        
        status_folder, status_label = self._status_to_save_folder(status)
        if status_folder is None:
            with self.stats_lock:
                self.stats['crops_skipped_unverified'] += 1
            print(
                f"⏭️  Skipped crop #{detection_number + 1} (unknown status={status}) "
                f"- raw copy should be in CROPS/"
            )
            return False
        
        if not verified and not self.save_unverified_crops:
            with self.stats_lock:
                self.stats['crops_skipped_unverified'] += 1
            print(
                f"⏭️  Skipped unverified crop #{detection_number + 1} "
                f"(killer={killer_name}, victim={victim_name}, status={status}) "
                f"- raw copy in CROPS/"
            )
            return False
        
        sequence_number = detection_number + 1
        safe_killer = self._sanitize_filename_part(killer_name)
        safe_victim = self._sanitize_filename_part(victim_name)
        filename = f"{sequence_number:03d}_{safe_killer}_{status_label}_{safe_victim}.png"

        skip_api_duplicate = False
        if not skip_api_push:
            skip_api_duplicate = self._is_name_cooldown_duplicate(killer_name, victim_name, status)

        # Push to TMS immediately from memory — don't wait for disk save
        if (
            not skip_api_push
            and self.api_push_enabled
            and not skip_api_duplicate
            and names_validated
        ):
            self._queue_tms_push(
                victim_name, killer_name, status, cropped_image, sequence_number
            )
            with self.stats_lock:
                self.stats['detections_processed'] += 1
        elif skip_api_duplicate:
            print(f"   🚫 Name cooldown duplicate - skipping TMS push")

        filepath = self._save_image(cropped_image, filename, status_folder)
        
        if not filepath:
            return False
        
        self._mirror_crop_to_legacy_folder(cropped_image, filename)
        
        rel_path = os.path.join(self.match_output_base, status_folder, filename)
        verified_label = "Yes" if verified else "No (OCR pending/failed)"
        print(f"📸 Saved: {rel_path}")
        print(f"   🎯 Killer: {killer_name} | 📊 Status: {status_label} | 👤 Victim: {victim_name}")
        print(f"   ✅ Confidence: {confidence:.2f} | Verified: {verified_label} | Order: {killer_name} → {status_label} → {victim_name}")
        if suspect_swap:
            print(f"   ⚠️  WARNING: Suspect swap detected - verify manually if needed")
        if not verified:
            print(f"   ℹ️  Saved with OCR names as-is; check match_*/CROPS/ for the raw YOLO crop")
        
        with self.stats_lock:
            self.stats['crops_saved'] += 1
        
        if killer_name != "UNKNOWN":
            with self.processing_lock:
                self.known_players.add(killer_name)
        if victim_name != "UNKNOWN":
            with self.processing_lock:
                self.known_players.add(victim_name)
        
        with self.processing_lock:
            self.detection_sequence = max(self.detection_sequence, sequence_number)
        
        return True
    
    def _queue_tms_push(self, victim_name, killer_name, status, cropped_image, sequence_number):
        """Queue TMS API in strict sequence order — one request at a time."""
        if not self.api_push_enabled:
            return
        image_copy = cropped_image.copy() if cropped_image is not None else None
        try:
            self.api_queue.put_nowait(
                {
                    "player_name": victim_name,
                    "enemy_name": killer_name,
                    "status": status,
                    "image_path": None,
                    "sequence_number": sequence_number,
                    "image_array": image_copy,
                }
            )
        except Full:
            print(f"⚠️ API queue full — could not queue killfeed #{sequence_number}")
        except Exception as e:
            print(f"⚠️ TMS push queue failed: {type(e).__name__}")
    
    def _send_to_api(self, player_name, enemy_name, status, image_path, sequence_number):
        """
        Legacy synchronous API method (kept for compatibility).
        New async pipeline uses _send_to_api_with_retry instead.
        """
        # Queue for async processing instead of blocking
        if self.api_enabled:
            api_request = {
                'player_name': player_name,
                'enemy_name': enemy_name,
                'status': status,
                'image_path': image_path,
                'sequence_number': sequence_number
            }
            try:
                self.api_queue.put_nowait(api_request)
                return True
            except:
                return False
        return False
    
    def _validate_names_fifo(self, killer_name, victim_name, confidence):
        """Relaxed validation for FIFO pipeline — fuzzy match handles OCR noise."""
        if confidence < 0.50:
            return False
        if not killer_name or not victim_name:
            return False
        if len(killer_name) < self.min_name_length or len(victim_name) < self.min_name_length:
            return False
        if killer_name.upper() == victim_name.upper():
            return False
        return True

    def _validate_names(self, killer_name, victim_name, confidence):
        """Validate that names meet quality requirements before pushing to TMS."""
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Check both names are present
        if not killer_name or not victim_name:
            return False
        
        # Check name lengths
        if (len(killer_name) < self.min_name_length or 
            len(killer_name) > self.max_name_length or
            len(victim_name) < self.min_name_length or 
            len(victim_name) > self.max_name_length):
            return False
        
        # Check names are different
        if killer_name.upper() == victim_name.upper():
            return False
        
        # Check names don't contain only numbers
        if killer_name.isdigit() or victim_name.isdigit():
            return False
        
        return True
    
    def _process_detection(self, result):
        """
        Queue detection for async processing with hash dedup and monotonic frame number.
        """
        cropped_image = result.get('cropped_image')
        if cropped_image is None or cropped_image.size == 0:
            return False
        
        if self._is_duplicate_frame(cropped_image):
            return False  # identical pixels — skip OCR and save (layer 2 dedup)
        
        with self.processing_lock:
            detection_number = self.next_detection_number
            self.next_detection_number += 1
        result['detection_number'] = detection_number
        
        self._save_raw_crop(cropped_image, detection_number, result.get('status', 'UNKNOWN'))
        
        try:
            self.detection_queue.put(result, timeout=2.0)
            return True
        except:
            # Queue saturated: run OCR inline so YOLO detection is never lost
            processed = self._process_detection_async(result)
            if processed is None:
                processed = self._fallback_result_payload(result)
            self._push_result(detection_number, processed)
            with self.stats_lock:
                self.stats['queue_drops'] += 1
            print(f"⚠️ Detection queue saturated - saved inline #{detection_number + 1}")
            return True
    
    def _apply_camera_source(self, source, size, label):
        """Store resolved OBS source for capture thread and legacy cap."""
        self.camera_source = source
        if isinstance(source, int):
            self.camera_index = source
        elif isinstance(source, str) and source.startswith("/dev/video"):
            try:
                self.camera_index = int(source.replace("/dev/video", ""))
            except ValueError:
                pass
        print(f"✅ Camera ready: {label} ({size[0]}x{size[1]})")

    def _initialize_camera(self):
        """Initialize OBS camera."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            source, size, label = _resolve_camera_source(self.camera_index, quiet=True)
            if size is None:
                return False
            self._apply_camera_source(source, size, label)

            self.cap = _open_video_capture(self.camera_source)
            if not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, 1)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(5):
                self.cap.read()

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                self.cap.release()
                self.cap = None
                return False
            return True
        except Exception:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def _capture_frame(self):
        """Capture frame from OBS camera. Returns None on failure but never raises exceptions."""
        try:
            if self.cap is None or not self.cap.isOpened():
                if not self._initialize_camera():
                    return None
            
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
            return None
        except Exception as e:
            # Log but don't raise - let caller handle retry
            return None
    
    def _ensure_camera_ready(self, max_wait=300):
        """Wait until OBS Virtual Camera delivers real frames (probe must pass)."""
        start = time.time()
        last_msg = 0.0
        while not self.stop_event.is_set():
            source, size, label = _resolve_camera_source(self.camera_index, quiet=True)
            if source is not None and size is not None:
                self._apply_camera_source(source, size, label)
                return True

            now = time.time()
            if now - last_msg >= 4.0:
                print(
                    "⏳ Waiting for OBS Virtual Camera frames…\n"
                    "   In OBS: Controls → Start Virtual Camera (must show 'Stop Virtual Camera')\n"
                    "   Make sure game scene is visible in OBS preview"
                )
                last_msg = now

            if max_wait and (now - start) > max_wait:
                print("❌ OBS Virtual Camera not streaming after 5 min — fix OBS then re-run.")
                return False
            time.sleep(2.0)
        return False

    def start_detection(self):
        """Main detection loop. Uses FIFO pipeline when enabled, else legacy per-crop."""
        print("\n🚀 Starting detection...")
        print("⚠️  Press Ctrl+C to stop\n")

        if not self._ensure_camera_ready():
            self._cleanup()
            return

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.use_fifo_pipeline:
            if self.local_model is None:
                print("⚠️  FIFO pipeline requires local YOLO model — falling back to legacy loop")
                self.use_fifo_pipeline = False
            else:
                self._run_fifo_pipeline_loop()
                self._cleanup()
                return

        print("💡 Legacy per-crop mode\n")
        while not self.stop_event.is_set():
            try:
                self._run_detection_loop()
            except KeyboardInterrupt:
                print("\n⏹️ Detection stopped by user")
                break
            except Exception as e:
                print(f"\n⚠️ Detection loop crashed: {type(e).__name__}")
                print(f"   Error: {str(e)[:200]}")
                print("   🔄 Restarting detection loop in 3 seconds...")
                time.sleep(3)
                continue

        self._cleanup()

    def _run_fifo_pipeline_loop(self):
        """Run time-series FIFO killfeed pipeline until stop."""
        from killfeed.pipeline import KillfeedPipeline

        print("🛡️  FIFO mode: one verified result per event — no raw UNKNOWN saves")
        print(f"📝 Killfeed log: {os.path.abspath(self.killfeed_log_path)}")
        print(f"📝 Killfeed JSONL: {os.path.abspath(self.killfeed_jsonl_path)}\n")
        if self.ocr_subprocess is None:
            ocr_cfg = _load_app_config().get("ocr", {})
            if ocr_cfg.get("use_subprocess", True):
                from killfeed.ocr_subprocess import SubprocessOCRProvider
                self.ocr_subprocess = SubprocessOCRProvider()
        ocr = _KillfeedOCRAdapter(self, subprocess_ocr=self.ocr_subprocess)
        fuzzy_threshold = float(
            _load_app_config().get("processing", {}).get("similarity_threshold_high", 0.90)
        ) * 100

        self.fifo_pipeline = KillfeedPipeline(
            model=self.local_model,
            ocr_provider=ocr,
            on_event=self._on_fifo_killfeed_event,
            name_validator=self._validate_names_fifo,
            camera_source=self.camera_source,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            capture_fps=self.capture_fps,
            frame_buffer_seconds=self.frame_buffer_seconds,
            kill_confidence=self.yolo_confidence,
            revive_confidence=self.revive_confidence,
            ocr_queue_max=self.ocr_queue_max,
            max_visible_slots=self.max_visible_slots,
            cache_ttl_seconds=self.cache_ttl_seconds,
            pair_cooldown_seconds=self.pair_cooldown_seconds,
            fuzzy_threshold=fuzzy_threshold,
            roi_change_threshold=self.roi_change_threshold,
            dhash_max_distance=self.dhash_max_distance,
            log_path=self.killfeed_log_path,
            jsonl_path=self.killfeed_jsonl_path,
            open_capture=_open_video_capture,
            yolo_lock=self.yolo_lock,
        )
        self.fifo_pipeline.start()

        last_status = time.time()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
                if time.time() - last_status >= 15:
                    st = self.fifo_pipeline.stats
                    print(
                        f"⏳ FIFO running | frames={st.get('frames_captured', 0)} "
                        f"| yolo={st.get('yolo_hits', 0)}/{st.get('yolo_misses', 0)} "
                        f"| OCR={st.get('ocr_runs', 0)} "
                        f"| emitted={st.get('events_emitted', 0)} "
                        f"| dup_skip={st.get('dup_skipped', 0)} "
                        f"| dropped={st.get('strips_dropped', 0)} "
                        f"| gate_skips={st.get('gate_skips', 0)} "
                        f"| dropped={st.get('strips_dropped', 0)}"
                    )
                    if st.get("frames_captured", 0) == 0:
                        print("   ⚠️  No camera frames — OBS Virtual Camera running?")
                    elif st.get("yolo_hits", 0) == 0 and st.get("frames_captured", 0) > 30:
                        print("   ⚠️  Camera OK but YOLO sees no killblocks — check OBS scene")
                    elif st.get("ocr_runs", 0) > 0 and st.get("events_emitted", 0) == 0:
                        print("   ⚠️  OCR ran but no events emitted — names/classifier may be failing")
                    last_status = time.time()
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        finally:
            if self.fifo_pipeline:
                self.fifo_pipeline.stop()
                self.fifo_pipeline = None

    def _on_fifo_killfeed_event(self, row, frame_num, sequence):
        """Push TMS first (speed), then save crop to disk."""
        if self.api_push_enabled:
            if self._is_name_cooldown_duplicate(row.killer, row.victim, row.tms_status):
                print("   🚫 TMS duplicate skipped (name cooldown)")
            else:
                self._queue_tms_push(
                    row.victim,
                    row.killer,
                    row.tms_status,
                    row.row_crop,
                    sequence,
                )
        result = {
            "cropped_image": row.row_crop,
            "status": row.tms_status,
            "detection_number": sequence - 1,
            "killer_name": row.killer,
            "victim_name": row.victim,
            "confidence": row.confidence,
            "suspect_swap": False,
            "names_validated": True,
        }
        self._save_kill_block(result, sequence - 1, skip_api_push=True)
    
    def _run_detection_loop(self):
        """Inner detection loop - runs until error or KeyboardInterrupt."""
        # Initialize camera
        while not self._initialize_camera():
            print("⚠️ Camera initialization failed, retrying in 2 seconds...")
            time.sleep(2)
        frame_count = 0
        last_detection_time = 0
        consecutive_errors = 0
        max_consecutive_errors = 1000  # Very high limit - will never stop on errors, only on manual stop
        
        # Main loop - runs forever until KeyboardInterrupt
        while not self.stop_event.is_set():
            try:
                # Capture frame
                frame = self._capture_frame()
                if frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"⚠️ Camera issue: {consecutive_errors} consecutive failed frames, retrying camera...")
                        try:
                            if not self._initialize_camera():
                                print("⚠️ Camera reinitialization failed, will keep trying...")
                        except Exception as e:
                            print(f"⚠️ Camera reinit error: {type(e).__name__}, will keep trying...")
                        consecutive_errors = 0  # Reset after warning
                    time.sleep(0.5)
                    continue
                
                # Reset error counter on successful frame capture
                consecutive_errors = 0
                frame_count += 1
                # Removed delay - process frames as fast as possible
                
                # Process frame (local or via gRPC)
                try:
                    if self.use_local_model:
                        results = self._process_frame_local(frame)
                    else:
                        results = self._process_frame_via_grpc(frame)
                except Exception as e:
                    # Processing errors are handled internally, but log if persistent
                    error_type = type(e).__name__
                    mode = "local" if self.use_local_model else "gRPC"
                    print(f"⚠️ {mode} processing error (frame #{frame_count}): {error_type}")
                    results = []
                    # Try to reconnect if using gRPC
                    if not self.use_local_model and self.grpc_stub is None:
                        print("🔄 Attempting to reconnect to gRPC server...")
                        self._connect_to_server()
                    continue
                
                # Queue every YOLO detection (never gate saves on OCR or time window)
                if results:
                    try:
                        current_time = time.time()
                        should_log_batch = (current_time - last_detection_time) > 0.1
                        if should_log_batch:
                            last_detection_time = current_time
                            print(f"\n🎯 Detection at frame #{frame_count} - Queued for processing")
                        
                        queued_count = 0
                        for result in results:
                            try:
                                if self._process_detection(result):
                                    queued_count += 1
                            except Exception as e:
                                print(f"⚠️ Error queuing detection: {type(e).__name__}")
                                continue
                        
                        if queued_count > 0 and should_log_batch:
                            print(f"✅ {queued_count} detection(s) queued for async processing\n")
                    except Exception as e:
                        # Log but continue loop
                        print(f"⚠️ Error in detection queuing (frame #{frame_count}): {type(e).__name__}")
                        continue
                
                if frame_count % 50 == 0:
                    with self.stats_lock:
                        queue_size = self.detection_queue.qsize()
                        api_queue_size = self.api_queue.qsize()
                    with self.heap_lock:
                        heap_pending = len(self.output_heap)
                    print(f"⏳ Running... Frame #{frame_count} | Detection Queue: {queue_size} | Output Heap: {heap_pending} | API Queue: {api_queue_size}")
                
            except KeyboardInterrupt:
                # User wants to stop - break out of inner loop
                raise
            except Exception as e:
                # Catch ANY error and continue running - never exit
                consecutive_errors += 1
                error_type = type(e).__name__
                
                # Only log errors occasionally to avoid spam
                if consecutive_errors <= 5 or consecutive_errors % 20 == 0:
                    print(f"⚠️ Error in detection loop (frame #{frame_count}, error #{consecutive_errors}): {error_type}")
                    
                    # If too many consecutive errors, try to recover
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"⚠️ Multiple consecutive errors detected, attempting recovery...")
                        # Try to reinitialize camera
                        try:
                            if not self._initialize_camera():
                                print("⚠️ Camera recovery failed, continuing anyway...")
                        except Exception as recover_error:
                            print(f"⚠️ Recovery error: {type(recover_error).__name__}, continuing...")
                        # Try to reconnect gRPC
                        try:
                            if not self.use_local_model and self.grpc_stub is None:
                                self._connect_to_server()
                        except Exception as recover_error:
                            pass
                        consecutive_errors = 0  # Reset counter after recovery attempt
                    
                    # Small delay before continuing to avoid tight error loop
                    time.sleep(0.5)
                # CRITICAL: Always continue - never break or exit
                continue
        
    def _start_workers(self):
        """Start background worker threads for async processing."""
        # Detection processing worker
        self.detection_worker = Thread(target=self._detection_worker, daemon=True, name="DetectionWorker")
        self.detection_worker.start()
        
        # API sending worker
        self.api_worker = Thread(target=self._api_worker, daemon=True, name="APIWorker")
        self.api_worker.start()
        
        print("✅ Background workers started")
    
    def _monitor_workers(self):
        """Monitor worker threads and restart them if they die."""
        while not self.stop_event.is_set():
            try:
                time.sleep(5)  # Check every 5 seconds
                
                # Check if detection worker is alive
                if not self.detection_worker.is_alive():
                    print("⚠️ Detection worker died! Restarting...")
                    with self.worker_restart_lock:
                        self.detection_worker = Thread(target=self._detection_worker, daemon=True, name="DetectionWorker")
                        self.detection_worker.start()
                        print("✅ Detection worker restarted")
                
                # Check if API worker is alive
                if not self.api_worker.is_alive():
                    print("⚠️ API worker died! Restarting...")
                    with self.worker_restart_lock:
                        self.api_worker = Thread(target=self._api_worker, daemon=True, name="APIWorker")
                        self.api_worker.start()
                        print("✅ API worker restarted")
                        
            except Exception as e:
                # Never exit monitor - keep checking
                time.sleep(5)
    
    def _detection_worker(self):
        """Background worker that processes detections from queue. Never stops unless explicitly stopped."""
        error_count = 0
        max_errors = 100  # Allow many errors before giving up
        
        while not self.stop_event.is_set():
            try:
                # Get detection from queue with timeout
                try:
                    result = self.detection_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Process detection asynchronously, then push to ordered output heap
                try:
                    queued_result = result
                    det_num = result.get('detection_number')
                    future = self.ocr_executor.submit(self._process_detection_async, queued_result)
                    
                    def _on_ocr_done(fut, original=queued_result, detection_num=det_num):
                        processed = None
                        try:
                            processed = fut.result()
                        except Exception:
                            processed = None
                        if processed is None and detection_num is not None:
                            cropped = original.get('cropped_image')
                            if cropped is not None and cropped.size > 0:
                                processed = self._fallback_result_payload(original)
                        if processed is not None and detection_num is not None:
                            self._push_result(detection_num, processed)
                    
                    future.add_done_callback(_on_ocr_done)
                    self.detection_queue.task_done()
                    error_count = 0
                except Exception as e:
                    det_num = result.get('detection_number')
                    if det_num is not None:
                        self._push_result(det_num, self._fallback_result_payload(result))
                    try:
                        self.detection_queue.task_done()
                    except:
                        pass
                    error_count += 1
                    if error_count < 10:
                        print(f"⚠️ Detection worker submission error: {type(e).__name__}")
                    time.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                if error_count < 10:
                    print(f"⚠️ Detection worker error: {type(e).__name__}")
                elif error_count == 10:
                    print(f"⚠️ Detection worker: Suppressing further error messages (error count: {error_count})")
                
                # Never exit - keep trying
                time.sleep(0.1)
                
                # Reset error count periodically to allow recovery
                if error_count >= max_errors:
                    error_count = 0
                    print("🔄 Detection worker: Resetting error counter, continuing...")
    
    def _api_worker(self):
        """Background worker that sends API requests from queue. Never stops unless explicitly stopped."""
        error_count = 0
        max_errors = 100  # Allow many errors before giving up
        
        while not self.stop_event.is_set():
            try:
                # Get API request from queue with timeout
                try:
                    api_request = self.api_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Send in strict sequence — wait for each request before next
                try:
                    self._send_to_api_with_retry(**api_request)
                    self.api_queue.task_done()
                    error_count = 0
                except Exception as e:
                    # Even if submission fails, mark task as done to prevent queue blocking
                    try:
                        self.api_queue.task_done()
                    except:
                        pass
                    error_count += 1
                    if error_count < 10:  # Only log first few errors
                        print(f"⚠️ API worker submission error: {type(e).__name__}")
                    time.sleep(0.1)  # Brief pause before retry
                
            except Exception as e:
                error_count += 1
                if error_count < 10:
                    print(f"⚠️ API worker error: {type(e).__name__}")
                elif error_count == 10:
                    print(f"⚠️ API worker: Suppressing further error messages (error count: {error_count})")
                
                # Never exit - keep trying
                time.sleep(0.1)
                
                # Reset error count periodically to allow recovery
                if error_count >= max_errors:
                    error_count = 0
                    print("🔄 API worker: Resetting error counter, continuing...")
    
    def _process_detection_async(self, result):
        """Process detection asynchronously (OCR). Always returns save payload if crop exists."""
        try:
            cropped_image = result.get('cropped_image')
            status = result.get('status', 'UNKNOWN')
            detection_number = result.get('detection_number')
            
            if cropped_image is None or cropped_image.size == 0:
                return None
            
            # Extract names using robust OCR (client-side) with strict left-to-right ordering
            killer_name, victim_name, confidence = self._extract_names(cropped_image)
            
            if not killer_name or len(killer_name.strip()) < self.min_name_length:
                killer_name = "UNKNOWN"
            if not victim_name or len(victim_name.strip()) < self.min_name_length:
                victim_name = "UNKNOWN"
            
            names_validated = self._validate_names(
                killer_name if killer_name != "UNKNOWN" else "",
                victim_name if victim_name != "UNKNOWN" else "",
                confidence
            )
            
            # One retry on failed validation — improves capture during fast killfeed bursts
            if not names_validated and self.ocr_retry_attempts > 0:
                retry_killer, retry_victim, retry_conf = self._extract_names(cropped_image)
                if retry_killer and len(retry_killer.strip()) >= self.min_name_length:
                    killer_name = retry_killer
                if retry_victim and len(retry_victim.strip()) >= self.min_name_length:
                    victim_name = retry_victim
                confidence = max(confidence, retry_conf)
                names_validated = self._validate_names(
                    killer_name if killer_name != "UNKNOWN" else "",
                    victim_name if victim_name != "UNKNOWN" else "",
                    confidence
                )
            
            suspect_swap = False
            if (
                killer_name != "UNKNOWN"
                and victim_name != "UNKNOWN"
                and killer_name.upper() != victim_name.upper()
                and self.tms_players_loaded
                and len(self.tms_players_list) > 0
            ):
                killer_as_victim_count = sum(
                    1 for h in self.name_history
                    if h.get('victim', '').upper() == killer_name.upper()
                )
                victim_as_killer_count = sum(
                    1 for h in self.name_history
                    if h.get('killer', '').upper() == victim_name.upper()
                )
                if victim_as_killer_count > killer_as_victim_count + 2:
                    suspect_swap = True
                    print(f"⚠️  LOGGING ONLY: '{victim_name}' often appears as killer in history")
                    print(f"   Current order (MAINTAINED): Killer='{killer_name}' → Victim='{victim_name}'")
                    print(f"   History pattern: Killer='{victim_name}' → Victim='{killer_name}'")
                    print(f"   ⚠️  NO SWAP PERFORMED - trusting strict left-to-right X-coordinate order")
            
            if killer_name != "UNKNOWN" and victim_name != "UNKNOWN":
                if killer_name.upper() == victim_name.upper():
                    print(f"⚠️  SUSPECT DETECTION: Killer and victim OCR match - saving with OCR names")
                    print(f"   Killer: '{killer_name}' | Victim: '{victim_name}'")
            
            return {
                'cropped_image': cropped_image,
                'status': status,
                'detection_number': detection_number,
                'killer_name': killer_name,
                'victim_name': victim_name,
                'confidence': confidence,
                'suspect_swap': suspect_swap,
                'names_validated': names_validated,
            }
            
        except Exception as e:
            print(f"⚠️ Error in async detection processing: {type(e).__name__}")
            cropped_image = result.get('cropped_image')
            if cropped_image is not None and cropped_image.size > 0:
                return self._fallback_result_payload(result)
            return None
    
    def _send_to_api_with_retry(
        self,
        player_name,
        enemy_name,
        status,
        image_path,
        sequence_number,
        image_array=None,
        max_retries=None,
    ):
        """
        Send killblock data to TMS API with retry logic.
        Uses in-memory JPEG when image_array is provided (fast path).
        """
        if not self.api_enabled:
            return False
        if not self.access_token:
            return False
        if max_retries is None:
            max_retries = self.api_max_retries
        
        for attempt in range(max_retries):
            try:
                base64_image = ""
                if image_array is not None and getattr(image_array, "size", 0) > 0:
                    ok, buf = cv2.imencode(
                        ".jpg", image_array, [cv2.IMWRITE_JPEG_QUALITY, 82]
                    )
                    if ok:
                        base64_image = base64.b64encode(buf.tobytes()).decode("utf-8")
                elif image_path and os.path.isfile(image_path):
                    with open(image_path, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode("utf-8")
                
                payload = {
                    "killerName": enemy_name,
                    "victimName": player_name,
                    "weaponUsed": status,
                    "imagePath": image_path or "",
                    "siftWeapon": "",
                    "image": base64_image,
                }
                
                headers = {
                    'accept': 'text/plain', 
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache',
                }
                if self.access_token:
                    headers['Authorization'] = f'Bearer {self.access_token}'
                
                read_timeout = min(self.api_timeout, 12)
                timeout = (2, read_timeout) if attempt == 0 else (2, max(5, read_timeout // 2))
                response = self.api_session.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code in [200, 201]:
                    print(f"✅ API: Sent killblock #{sequence_number} to TMS | {enemy_name} {status} {player_name}")
                    with self.stats_lock:
                        self.stats['api_success'] += 1
                    return True
                elif response.status_code == 401:
                    # Don't retry on auth errors
                    error_msg = "Authentication failed (401 Unauthorized)"
                    if not self.access_token:
                        error_msg += " - No access token provided"
                    print(f"⚠️ API: {error_msg}")
                    with self.stats_lock:
                        self.stats['api_failed'] += 1
                    return False
                else:
                    # Retry on other errors
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 0.1  # Exponential backoff: 0.1s, 0.2s, 0.4s
                        time.sleep(wait_time)
                        with self.stats_lock:
                            self.stats['api_retries'] += 1
                        continue
                    else:
                        detail = self._parse_api_error_message(response)
                        print(
                            f"⚠️ API: Request failed (Status: {response.status_code}) "
                            f"after {max_retries} attempts"
                        )
                        if detail:
                            print(f"   TMS says: {detail}")
                        with self.stats_lock:
                            self.stats['api_failed'] += 1
                        return False
                        
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.2  # Slightly longer backoff
                    time.sleep(wait_time)
                    with self.stats_lock:
                        self.stats['api_retries'] += 1
                    # Try to reset session connection on timeout
                    try:
                        self.api_session.close()
                        adapter = requests.adapters.HTTPAdapter(
                            pool_connections=10,
                            pool_maxsize=20,
                            max_retries=0,
                            pool_block=False
                        )
                        self.api_session.mount('http://', adapter)
                        self.api_session.mount('https://', adapter)
                    except:
                        pass
                    continue
                else:
                    print(f"⚠️ API: Timeout after {max_retries} attempts for #{sequence_number} (TMS server may be slow)")
                    with self.stats_lock:
                        self.stats['api_failed'] += 1
                    return False
            except requests.exceptions.ConnectionError as e:
                # Connection errors - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.3
                    time.sleep(wait_time)
                    with self.stats_lock:
                        self.stats['api_retries'] += 1
                    # Reset session on connection error
                    try:
                        self.api_session.close()
                        adapter = requests.adapters.HTTPAdapter(
                            pool_connections=10,
                            pool_maxsize=20,
                            max_retries=0,
                            pool_block=False
                        )
                        self.api_session.mount('http://', adapter)
                        self.api_session.mount('https://', adapter)
                    except:
                        pass
                    continue
                else:
                    print(f"⚠️ API: Connection error after {max_retries} attempts for #{sequence_number}")
                    with self.stats_lock:
                        self.stats['api_failed'] += 1
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.1
                    time.sleep(wait_time)
                    with self.stats_lock:
                        self.stats['api_retries'] += 1
                    continue
                else:
                    print(f"⚠️ API error: {type(e).__name__} for #{sequence_number}")
                    with self.stats_lock:
                        self.stats['api_failed'] += 1
                    return False
        
        return False
    
    def _cleanup(self):
        """Cleanup resources."""
        print("🛑 Shutting down workers...")

        if self.fifo_pipeline is not None:
            try:
                self.fifo_pipeline.stop()
            except Exception:
                pass
            self.fifo_pipeline = None

        if self.ocr_subprocess is not None:
            try:
                self.ocr_subprocess.shutdown()
            except Exception:
                pass
            self.ocr_subprocess = None
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for queues to drain (with timeout)
        try:
            if not self.use_fifo_pipeline:
                self.detection_queue.join()
            self.ocr_executor.shutdown(wait=True, timeout=5)
            self.api_queue.join()
        except:
            pass
        
        with self.heap_lock:
            pending_heap = len(self.output_heap)
            if pending_heap:
                print(f"⚠️ {pending_heap} detection(s) still pending ordered output (waiting on sequence)")
        
        # Shutdown executors
        try:
            self.api_executor.shutdown(wait=True, timeout=5)
        except:
            pass
        
        # Wait for worker threads
        try:
            if hasattr(self, "detection_worker") and self.detection_worker:
                self.detection_worker.join(timeout=2)
            if hasattr(self, "api_worker") and self.api_worker:
                self.api_worker.join(timeout=2)
        except:
            pass
        
        if self.fifo_pipeline is not None:
            try:
                st = self.fifo_pipeline.stats
                print(f"   FIFO events emitted: {st.get('events_emitted', 0)}")
                print(f"   FIFO OCR runs: {st.get('ocr_runs', 0)}")
                print(f"   FIFO strips dropped: {st.get('strips_dropped', 0)}")
            except Exception:
                pass
        
        # Print statistics
        with self.stats_lock:
            print(f"\n📊 Processing Statistics:")
            print(f"   Detections processed: {self.stats['detections_processed']}")
            print(f"   API success: {self.stats['api_success']}")
            print(f"   API failed: {self.stats['api_failed']}")
            print(f"   API retries: {self.stats['api_retries']}")
            print(f"   Queue drops: {self.stats['queue_drops']}")
            print(f"   OCR errors: {self.stats['ocr_errors']}")
            print(f"   Named crops saved: {self.stats['crops_saved']}")
            print(f"   Raw YOLO crops saved: {self.stats.get('raw_crops_saved', 0)}")
            print(f"   Crops skipped (unknown status): {self.stats['crops_skipped_unverified']}")
        
        # Cleanup other resources
        try:
            # Close API session
            if hasattr(self, 'api_session'):
                self.api_session.close()
            if self.cap is not None:
                self.cap.release()
            if self.grpc_channel:
                self.grpc_channel.close()
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
        except:
            pass
        print("✅ Cleanup complete")


def main():
    """Main entry point."""
    print("=== Free Fire Killblock Detector ===")
    print("")
    print("💡 Processing Modes:")
    print("   - FIFO pipeline (default): strip OCR + time-series diff — one result per event")
    print("   - Legacy mode: set detection.use_fifo_pipeline=false in config.json")
    print("   - Local mode: Uses YOLO model directly (required for FIFO)")
    print("   - Server mode: Uses gRPC server (legacy only)")
    print("")
    
    auto_run = os.environ.get("FF_AUTO", "").lower() in ("1", "true", "yes") or (
        len(__import__("sys").argv) > 1 and __import__("sys").argv[1] in ("--run", "-y")
    )
    
    # Get configuration (defaults from local_config.json if present)
    local_cfg = _load_local_config()
    app_cfg = _load_app_config()
    default_match = local_cfg.get("match_id") or app_cfg.get("match", {}).get("default_match_id", "1")
    saved_token = local_cfg.get("access_token")
    default_api = local_cfg.get("api_enabled", True)

    camera_cfg = app_cfg.get("camera", {})
    default_camera = camera_cfg.get("default_index", 2)
    model_path = "best.pt"

    if auto_run:
        match_id = default_match
        access_token = saved_token
        api_enabled = default_api
        use_local = True
        camera_index = default_camera
        print(f"⚡ Auto-run: match={match_id}, API={'on' if api_enabled else 'off'}, camera={default_camera}")
    else:
        match_id = input(f"Enter Match ID (Enter for '{default_match}'): ").strip() or default_match
        token_prompt = "Enter Access Token (Enter to use saved token): " if saved_token else "Enter Access Token (or press Enter to skip): "
        access_token = input(token_prompt).strip() or saved_token
        if access_token and not _is_valid_jwt(access_token):
            print("❌ Saved token is corrupted — login at http://192.168.29.47:5173 and paste a fresh token")
            access_token = input("Enter Access Token: ").strip() or None
        api_default = "Y" if default_api else "n"
        api_enabled = input(f"Enable API integration? (Y/n, default: {api_default}): ").strip().lower() != 'n'
        use_local = input("Use local model? (Y/n, default: Y): ").strip().lower() != 'n'
        if use_local:
            custom_model = input(f"Model path (or press Enter for '{model_path}'): ").strip()
            if custom_model:
                model_path = custom_model
        camera_input = input(
            f"Camera index (Enter for config default {default_camera}, 'auto' to scan): "
        ).strip()
        if camera_input.lower() == "auto":
            camera_index = None
        elif camera_input:
            camera_index = int(camera_input)
        else:
            camera_index = default_camera
    
    try:
        detector = KillblockDetector(
            match_id=match_id, 
            access_token=access_token, 
            api_enabled=api_enabled,
            use_local_model=use_local,
            model_path=model_path,
            camera_index=camera_index,
        )

        # Load current-match roster for fast OCR snapping (zero latency impact)
        detector.set_match_roster(MATCH_PLAYERS)

        # Check if initialization was successful
        if use_local:
            if detector.local_model is None:
                print("❌ Failed to load local model")
                return
            else:
                print("✅ Local model ready - no gRPC server needed")
        else:
            if detector.grpc_stub is None:
                print("❌ Failed to connect to gRPC server")
                print("")
                print("💡 Troubleshooting:")
                print("   1. Make sure port 50051 is not in use by another program")
                print("   2. If you manually started grpc_block.py, close it first")
                print("   3. Or run: python grpc_block.py --serve --model best.pt --port 50051")
                print("      Then run this program again")
                return
            else:
                print("✅ gRPC server connected")
        
        print("\n📹 Waiting for OBS Virtual Camera (start it in OBS if not running)…")
        detector.start_detection()
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except RuntimeError as e:
        error_msg = str(e)
        print(f"\n❌ Error: {e}")
        
        # Only show gRPC troubleshooting if it's actually a gRPC error
        if "gRPC" in error_msg or "port" in error_msg.lower() or "50051" in error_msg:
            print("")
            print("💡 Troubleshooting:")
            print("   1. Close any other instances of grpc_block.py")
            print("   2. Check if another program is using port 50051")
            print("   3. On Windows, you can check with: netstat -ano | findstr :50051")
            print("   4. Kill the process if needed, then try again")
            print("   5. Or use local model mode (select 'Y' when asked)")
        else:
            print("")
            print("💡 This error occurred during initialization.")
            print("   Please check the error message above for details.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
