"""Free Fire Killblock Detector - Optimized client for killfeed detection and API integration."""
import paddle_env  # noqa: F401 — must run before paddleocr import
import cv2
import os
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

# Import generated gRPC code
try:
    import killfeed_detection_pb2
    import killfeed_detection_pb2_grpc
except ImportError:
    killfeed_detection_pb2 = None
    killfeed_detection_pb2_grpc = None

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings("ignore")


class KillblockDetector:
    """Optimized killblock detector with three core responsibilities:
    1. Capture frames from OBS
    2. Process frames locally OR via gRPC server (configurable)
    3. Receive results, perform OCR, and send to TMS API
    """
    
    GRPC_SERVER_PORT = 50051
    GRPC_SERVER_HOST = "localhost"
    
    def __init__(self, match_id="1", access_token=None, api_enabled=True, grpc_port=50051, 
                 use_local_model=True, model_path="best (1).pt", camera_index=None, stop_flag=None):
        """Initialize detector with local or gRPC processing, OCR, and API configuration.
        
        Args:
            match_id: Match ID for TMS API
            access_token: Access token for TMS API
            api_enabled: Enable API integration
            grpc_port: Port for gRPC server (if using server mode)
            use_local_model: If True, use local YOLO model. If False, use gRPC server
            model_path: Path to YOLO model file (for local mode)
            camera_index: OBS Virtual Camera index (None = auto-detect best camera)
            stop_flag: threading.Event from desktop app to signal stop
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
            if killfeed_detection_pb2 is None or killfeed_detection_pb2_grpc is None:
                print(
                    "⚠️ gRPC protobuf files missing. Generate with:\n"
                    "  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. "
                    "killfeed_detection.proto",
                    flush=True,
                )
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
        
        # OCR loads in background — PaddleOCR can take 30–90s on first run
        self.text_detector = None
        self._ocr_ready = Event()
        self._ocr_init_lock = Lock()
        self._ocr_init_thread = Thread(
            target=self._load_ocr_background, daemon=True, name="OCRInit"
        )
        self._ocr_init_thread.start()
        print("🔍 Loading OCR in background (first run may take up to a minute)...", flush=True)
        
        # Camera (OBS Virtual Camera — index varies: often 0 or 1 on Windows)
        self.cap = None
        self.camera_index = camera_index
        self.yolo_conf_threshold = self._load_yolo_conf_threshold()
        
        # API configuration
        self.match_id = match_id
        self.access_token = access_token
        self.api_enabled = api_enabled
        self.api_push_enabled = bool(api_enabled and access_token)
        self.api_url = self._build_killfeed_api_url(match_id)
        
        if self.api_push_enabled:
            print(f"🌐 TMS killfeed API: {self.api_url}")
        elif self.api_enabled and not self.access_token:
            print("⚠️  WARNING: No access token — killfeeds will NOT be pushed to TMS.")
            print("   Log in via the desktop app, or paste a beta token in local_config.json")
        
        # Detection tracking
        self.detection_sequence = 0
        self.recent_detections = set()  # Name-based duplicate prevention (API/logging)
        self.name_cooldown = {}  # name key -> last seen timestamp (3s window for API)
        self.name_cooldown_window = 3.0
        self.crop_output_dir = self._init_crop_output_dir()
        self.status_detected_dir = self._init_status_detected_dir()
        self.status_detected_sequence = 0
        self.status_detected_lock = Lock()
        
        # Ordered output: min-heap by detection number (assigned at queue time)
        self.output_heap = []
        self.heap_lock = Lock()
        self.next_expected_frame = 0
        self.next_detection_number = 0
        
        # Visual killfeed deduplication (before OCR / status / save)
        self.killfeed_event_cooldown = 3.0  # seconds — typical on-screen killfeed duration
        self.visual_event_cache = deque(maxlen=48)   # queued / in-flight events
        self.saved_visual_cache = deque(maxlen=48)   # verified cropkillblock + API saves
        self.status_detected_visual_cache = deque(maxlen=48)  # debug status_detected/ only
        self.exact_crop_hashes = set()
        self.visual_dedup_lock = Lock()
        self.visual_dhash_threshold = 12       # max Hamming distance (of 64)
        self.visual_hist_threshold = 0.82        # histogram correlation minimum
        self.visual_corr_threshold = 0.86        # normalized gray correlation minimum
        
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
        
        # Async processing pipeline (work queue in; min-heap orders saves out)
        self.detection_queue = Queue(maxsize=500)  # Large buffer for burst killfeeds
        self.api_queue = Queue(maxsize=200)  # Queue for API calls
        self.processing_lock = Lock()  # Lock for sequence number updates
        self.stop_event = stop_flag if stop_flag is not None else Event()  # Event to signal shutdown
        
        # Thread pool for parallel processing
        # Reduced OCR workers to 1 because PaddleOCR is not fully thread-safe
        self.ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OCR")
        self.api_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="API")
        
        # OCR lock for thread safety (PaddleOCR is not thread-safe)
        self.ocr_lock = Lock()
        
        # Create persistent HTTP session for API calls (reuses connections)
        self.api_session = requests.Session()
        # Configure session with connection pooling and keep-alive
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0,  # We handle retries manually
            pool_block=False
        )
        self.api_session.mount('http://', adapter)
        self.api_session.mount('https://', adapter)

        if self.api_push_enabled:
            self._verify_tms_api()

        self._load_tms_players_list()   # from TMS API (may be empty if offline)
        self._load_roster_from_local_files()  # local roster.json / players.txt fallback
        
        # Statistics
        self.stats = {
            'detections_processed': 0,
            'api_success': 0,
            'api_failed': 0,
            'api_retries': 0,
            'queue_drops': 0,
            'ocr_errors': 0,
            'crops_saved': 0,
            'crops_skipped_unverified': 0,
            'crops_skipped_duplicate': 0,
            'status_detected_saved': 0,
            'yolo_rejected_shape': 0,
            'ocr_no_text': 0,
        }
        self.stats_lock = Lock()
        
        # Worker health monitoring
        self.worker_restart_lock = Lock()
        self.last_worker_check = time.time()
        self._last_yolo_hits = 0
        self.verbose_status_saves = False
        self._load_runtime_config()
        
        # Start background workers
        self._start_workers()
        
        # Start worker health monitor
        self.worker_monitor = Thread(target=self._monitor_workers, daemon=True, name="WorkerMonitor")
        self.worker_monitor.start()
        
        print("✅ Detector ready with async pipeline!")
    
    @staticmethod
    def _api_backend() -> str:
        """TMS API base URL — SCORE_API_URL env (desktop) or config fallback."""
        backend = os.environ.get("SCORE_API_URL", "").strip().rstrip("/")
        if backend:
            return backend
        return "http://3.7.109.218:5005"

    def _build_killfeed_api_url(self, match_id) -> str:
        backend = self._api_backend()
        return (
            f"{backend}/LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed"
            f"?matchId={match_id}"
        )

    def _build_teams_players_api_url(self) -> str:
        backend = self._api_backend()
        return (
            f"{backend}/LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players"
            f"?matchId={self.match_id}"
        )

    def _verify_tms_api(self):
        """Verify beta token before detection starts."""
        if not self.access_token:
            self.api_push_enabled = False
            return
        try:
            backend = self._api_backend()
            url = f"{backend}/User/GetUserFromToken"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.api_session.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                print("✅ TMS API token verified")
            elif response.status_code == 401:
                print("❌ TMS token rejected (401) — get a fresh token from beta-tms.16score.com")
                self.api_push_enabled = False
            else:
                print(f"⚠️ TMS API check: HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️ TMS API check failed: {type(e).__name__}")
            self.api_push_enabled = False

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
            print("⚠️ PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr==2.10.0")
        except (OSError, Exception) as e:
            error_msg = str(e)
            if "shm.dll" in error_msg or "WinError 127" in error_msg:
                print("❌ Visual C++ Redistributables required")
                print("   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            else:
                print(f"⚠️ OCR initialization error: {e}")
        
        print("⚠️ OCR disabled - names will be empty")
        return None

    def _load_ocr_background(self):
        """Load PaddleOCR off the main thread so capture can start sooner."""
        try:
            with self._ocr_init_lock:
                if self.text_detector is None:
                    self.text_detector = self._init_ocr()
        except Exception as exc:
            print(f"⚠️ Background OCR init failed: {exc}", flush=True)
        finally:
            self._ocr_ready.set()

    def _ensure_ocr(self, timeout: float = 120.0) -> bool:
        """Wait for background OCR init; return True when text_detector is ready."""
        if self.text_detector is not None:
            return True
        if not self._ocr_ready.wait(timeout=timeout):
            print("⚠️ OCR still loading — detection will retry when ready", flush=True)
            return False
        return self.text_detector is not None
    
    def _process_frame_local(self, frame):
        """Process frame locally using YOLO model."""
        if self.local_model is None or frame is None or frame.size == 0:
            return []
        
        try:
            # Detect killblocks and revives
            detections = self._detect_killblocks_and_revives_local(frame)
            if not detections:
                self._last_yolo_hits = 0
                return []
            
            merged_detections = self._merge_killfeed_detections(detections)
            self._last_yolo_hits = len(merged_detections)

            # Sort by Y-position ASCENDING: top of screen first (Slot 1 = newest kill
            # in Free Fire's killfeed layout).  This gives the newest kill the lowest
            # detection number so TMS sequence #1 == the most-recent event, matching
            # the on-screen killfeed order the operator expects to see in TMS.
            merged_detections.sort(key=lambda d: d['bbox'][1], reverse=False)

            results = []
            frame_signatures = []
            for detection in merged_detections:
                try:
                    x1, y1, x2, y2 = detection['bbox']
                    cropped_image = frame[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        continue
                    
                    if self._is_duplicate_killfeed_event(cropped_image, check_saved=True):
                        with self.stats_lock:
                            self.stats['crops_skipped_duplicate'] += 1
                        continue
                    
                    frame_sig = self._compute_killfeed_signature(cropped_image)
                    if frame_sig and any(
                        self._signatures_match(frame_sig, seen_sig)
                        for seen_sig in frame_signatures
                    ):
                        with self.stats_lock:
                            self.stats['crops_skipped_duplicate'] += 1
                        continue
                    if frame_sig:
                        frame_signatures.append(frame_sig)
                    
                    results.append({
                        'cropped_image': cropped_image,
                        'class_name': 'killblock',
                        'has_revive': detection.get('has_revive', False),
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                    })
                except Exception as e:
                    print(f"⚠️ Crop processing error: {type(e).__name__}: {e}")
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
            
            results = self.local_model.predict(
                processed_image,
                conf=self.yolo_conf_threshold,
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
                    
                    # Detect both killblock and revive classes
                    if (class_name_lower == "killblock" or class_name_lower == "revive") and confidence >= self.yolo_conf_threshold:
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
    
    def _bbox_iou(self, box_a, box_b):
        """Intersection-over-union for two [x1, y1, x2, y2] boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _revive_inside_killblock(self, revive_bbox, killblock_bbox):
        """True when a revive YOLO box belongs inside a full killfeed block."""
        if not revive_bbox or not killblock_bbox:
            return False
        rx1, ry1, rx2, ry2 = revive_bbox
        kx1, ky1, kx2, ky2 = killblock_bbox
        rcx = (rx1 + rx2) * 0.5
        rcy = (ry1 + ry2) * 0.5
        if kx1 <= rcx <= kx2 and ky1 <= rcy <= ky2:
            return True
        inter_x1 = max(rx1, kx1)
        inter_y1 = max(ry1, ky1)
        inter_x2 = min(rx2, kx2)
        inter_y2 = min(ry2, ky2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        revive_area = max(1, (rx2 - rx1) * (ry2 - ry1))
        return (inter_area / revive_area) >= 0.3

    def _merge_killfeed_detections(self, detections):
        """
        One event per full killblock crop.
        Revive boxes inside a killblock set has_revive=True; standalone revive crops are dropped.
        """
        if not detections:
            return []

        killblocks = [d for d in detections if d.get('class_name') == 'killblock']
        revives = [d for d in detections if d.get('class_name') == 'revive']

        merged = []
        for kb in killblocks:
            kb_bbox = kb.get('bbox')
            if not kb_bbox:
                continue
            has_revive = any(
                self._revive_inside_killblock(rv.get('bbox'), kb_bbox)
                for rv in revives
                if rv.get('bbox')
            )
            merged.append({
                'bbox': kb_bbox,
                'confidence': float(kb.get('confidence', 0.0)),
                'class_name': 'killblock',
                'has_revive': has_revive,
            })

        if not merged:
            return []

        return self._dedupe_overlapping_killblocks(merged)

    def _dedupe_overlapping_killblocks(self, events, iou_threshold=0.55):
        """Drop overlapping killblock events in the same frame; preserve revive flag on merge."""
        if not events:
            return []
        sorted_events = sorted(
            events,
            key=lambda d: (
                0 if d.get('has_revive') else 1,
                -float(d.get('confidence', 0.0)),
            ),
        )
        kept = []
        for candidate in sorted_events:
            bbox = candidate.get('bbox')
            if not bbox:
                continue
            merged_into_existing = False
            for kept_det in kept:
                if self._bbox_iou(bbox, kept_det['bbox']) >= iou_threshold:
                    if candidate.get('has_revive'):
                        kept_det['has_revive'] = True
                    merged_into_existing = True
                    with self.stats_lock:
                        self.stats['crops_skipped_duplicate'] += 1
                    break
            if merged_into_existing:
                continue
            kept.append(dict(candidate))
        return kept

    def _determine_status_local(self, cropped_image, detected_class=None, has_revive=False, verbose=False):
        """
        Status priority on the complete killfeed block:
        1. REVIVE — revive YOLO detected anywhere inside this killblock (final)
        2. KILL/KNOCK — victim text color only when no revive present
        """
        if has_revive:
            if verbose:
                print("   🟢 Status: REVIVE (revive icon inside killblock — final)")
            return 'revive'

        victim_color = self._analyze_victim_text_color(cropped_image)
        if victim_color == 'red':
            if verbose:
                print("   🔴 Status: KILL (victim text red)")
            return 'kill'
        if victim_color == 'white':
            if verbose:
                print("   ⚪ Status: KNOCK (victim text white)")
            return 'gun knockout'

        if verbose:
            print("   ⚪ Status: KNOCK (victim text unclear — default knock)")
        return 'gun knockout'

    def _extract_victim_text_roi(self, cropped_image):
        """Isolate victim name region; ignore killer, weapon, and trailing icons."""
        if cropped_image is None or cropped_image.size == 0:
            return None, None
        height, width = cropped_image.shape[:2]
        if height < 8 or width < 20:
            return None, None
        x1 = int(width * 0.48)
        x2 = int(width * 0.88)
        if x2 - x1 < 8:
            return None, None
        victim_roi = cropped_image[:, x1:x2]
        gray = cv2.cvtColor(victim_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(victim_roi, cv2.COLOR_BGR2HSV)
        bright_mask = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)[1]
        white_mask = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 70, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([168, 70, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        text_mask = cv2.bitwise_or(bright_mask, cv2.bitwise_or(white_mask, red_mask))
        if cv2.countNonZero(text_mask) < 6:
            text_mask = bright_mask
        return victim_roi, text_mask

    def _analyze_victim_text_color(self, cropped_image):
        """
        Analyze ONLY victim name text pixels (background ignored).
        Returns: 'red', 'white', or 'unknown'
        """
        try:
            victim_roi, text_mask = self._extract_victim_text_roi(cropped_image)
            if victim_roi is None or text_mask is None:
                return 'unknown'

            text_pixels = cv2.countNonZero(text_mask)
            if text_pixels < 6:
                return 'unknown'

            hsv = cv2.cvtColor(victim_roi, cv2.COLOR_BGR2HSV)
            bgr = victim_roi

            red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([168, 80, 80]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_and(cv2.bitwise_or(red_mask1, red_mask2), text_mask)

            white_mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 60, 255]))
            white_mask = cv2.bitwise_and(white_mask, text_mask)

            b, g, r = cv2.split(bgr)
            r_f = r.astype(np.float32)
            g_f = g.astype(np.float32)
            b_f = b.astype(np.float32)
            r_dom = ((r_f > g_f + 25) & (r_f > b_f + 25)).astype(np.uint8) * 255
            r_dom = cv2.bitwise_and(r_dom, text_mask)
            red_mask = cv2.bitwise_or(red_mask, r_dom)

            red_pixels = cv2.countNonZero(red_mask)
            white_pixels = cv2.countNonZero(white_mask)
            red_ratio = red_pixels / text_pixels
            white_ratio = white_pixels / text_pixels

            if red_ratio >= 0.06 and red_ratio >= white_ratio:
                return 'red'
            if white_ratio >= 0.06 and white_ratio > red_ratio:
                return 'white'
            if red_ratio >= 0.03:
                return 'red'
            if white_ratio >= 0.03:
                return 'white'
            return 'unknown'
        except Exception:
            return 'unknown'

    def _check_green_color_in_image(self, cropped_image):
        """Legacy helper — not used for status priority (revive = YOLO only)."""
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
                    
                    if self._is_duplicate_killfeed_event(cropped_image, check_saved=True):
                        with self.stats_lock:
                            self.stats['crops_skipped_duplicate'] += 1
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
    
    def _load_tms_players_list(self):
        """Load active TMS players list from API for fuzzy matching."""
        if not self.api_enabled or not self.access_token or not self.match_id:
            print("⚠️  TMS players list not loaded: API disabled or missing credentials")
            return
        
        try:
            # Use config.json endpoint or construct URL
            api_url = self._build_teams_players_api_url()
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            response = self.api_session.get(api_url, headers=headers, timeout=5)
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
    
    def _load_roster_from_local_files(self):
        """Load player roster from local roster.json / players.txt.

        Merges into tms_players_list so fuzzy matching works even when the
        TMS API is unavailable or hasn't returned data yet.  Also pre-populates
        known_players so _normalize_name can correct OCR errors from frame 1.

        Search order: script directory → parent directory → cwd.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [
            script_dir,
            os.path.dirname(script_dir),
            os.getcwd(),
        ]

        roster_players: set = set()

        # --- roster.json ---
        for base in search_dirs:
            roster_path = os.path.join(base, 'roster.json')
            if not os.path.exists(roster_path):
                continue
            try:
                with open(roster_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                for p in data.get('players', []):
                    if isinstance(p, str) and p.strip():
                        roster_players.add(p.strip())
                print(f"📋 Loaded {len(roster_players)} roster players from {roster_path}")
                break
            except Exception as exc:
                print(f"⚠️  roster.json read error: {exc}")

        # --- players.txt ---
        for base in search_dirs:
            players_path = os.path.join(base, 'players.txt')
            if not os.path.exists(players_path):
                continue
            try:
                added_txt = 0
                with open(players_path, 'r', encoding='utf-8') as fh:
                    for raw_line in fh:
                        line = raw_line.strip().lstrip('- ').strip()
                        # Accept lines that look like IGNs: contain a dot,
                        # no spaces, not a comment
                        if line and '.' in line and ' ' not in line and not line.startswith('#'):
                            if line.upper() not in {p.upper() for p in roster_players}:
                                roster_players.add(line)
                                added_txt += 1
                if added_txt:
                    print(f"📋 Merged {added_txt} extra players from {players_path}")
                break
            except Exception as exc:
                print(f"⚠️  players.txt read error: {exc}")

        if not roster_players:
            print("⚠️  No local roster files found — fuzzy matching relies solely on TMS API data")
            return

        # Merge into tms_players_list (de-duplicate case-insensitively)
        existing_upper = {p.upper() for p in self.tms_players_list}
        added = 0
        for p in sorted(roster_players):
            if p.upper() not in existing_upper:
                self.tms_players_list.append(p)
                existing_upper.add(p.upper())
                added += 1

        # Pre-populate known_players so _normalize_name benefits from frame 1
        with self.processing_lock:
            self.known_players.update(roster_players)

        self.tms_players_loaded = True
        total = len(self.tms_players_list)
        sample = ', '.join(sorted(roster_players)[:8])
        print(f"✅ Roster merged: {total} players for fuzzy matching (+{added} from local files)")
        print(f"   Sample: {sample}")

    def _fuzzy_match_player_name(self, ocr_name: str, min_similarity: float = 0.65) -> tuple:
        """IGN-aware fuzzy match of an OCR-extracted name against the roster.

        IGN format is TEAM.PLAYERNAME (e.g. GODL.ECOECO, K9.HUNNYSUNY).
        OCR commonly makes these errors with the dot separator:
          • drops the dot entirely  → "GODLECOECO"
          • reads dot as space       → "GODL ECOECO"
          • reads dot as underscore  → "GODL_ECOECO"
        We build normalised variants and score each with multiple RapidFuzz
        strategies (ratio, partial_ratio, token_sort_ratio), then take the
        highest scoring candidate.  A same-team-prefix bonus is also applied
        so that e.g. "GODL.EC0EC0" confidently resolves to "GODL.ECOECO".

        Returns (canonical_name, similarity_score, matched).
        """
        if not ocr_name or len(ocr_name) < 2:
            return ocr_name, 0.0, False

        if not self.tms_players_loaded or len(self.tms_players_list) == 0:
            return ocr_name, 0.0, False

        ocr_clean = ocr_name.strip()
        ocr_lower = ocr_clean.lower()
        candidates_lower = [p.lower() for p in self.tms_players_list]

        # Build OCR variants to handle common dot-separator OCR errors
        variants: set = {ocr_lower}
        variants.add(re.sub(r'[\s_]+', '.', ocr_lower))          # space/underscore → dot
        variants.add(re.sub(r'[^a-z0-9]', '', ocr_lower))        # strip all separators
        # Also try restoring a dot after a known team prefix length (2-4 chars)
        for prefix_len in (2, 3, 4, 5):
            if len(ocr_lower) > prefix_len and ocr_lower[prefix_len].isalpha():
                candidate = ocr_lower[:prefix_len] + '.' + ocr_lower[prefix_len:]
                variants.add(candidate)
        variants.discard('')

        if RAPIDFUZZ_AVAILABLE:
            try:
                best_score_val: float = 0.0
                best_idx: int = 0

                for variant in variants:
                    for scorer in (fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio):
                        result = process.extractOne(variant, candidates_lower, scorer=scorer)
                        if result:
                            _, raw_score, idx = result
                            if raw_score > best_score_val:
                                best_score_val = raw_score
                                best_idx = idx

                # Same-team-prefix bonus: if the OCR team prefix matches exactly,
                # re-score all same-team candidates at higher priority so minor
                # typos in the player suffix don't get beaten by a different team.
                if '.' in ocr_lower:
                    ocr_prefix = ocr_lower.split('.')[0]
                    for i, cand in enumerate(candidates_lower):
                        if '.' in cand and cand.split('.')[0] == ocr_prefix:
                            r = fuzz.ratio(ocr_lower, cand)
                            # Accept same-team match if within 8 pts of global best
                            if r >= best_score_val - 8:
                                best_score_val = max(best_score_val, float(r))
                                best_idx = i

                similarity = best_score_val / 100.0
                canonical_name = self.tms_players_list[best_idx]

                if similarity >= min_similarity:
                    return canonical_name, similarity, True
                return ocr_clean, similarity, False

            except Exception as exc:
                print(f"⚠️  RapidFuzz matching error: {type(exc).__name__}")

        # Fallback: plain Levenshtein distance over all variants
        best_match: str = ocr_clean
        best_similarity: float = 0.0
        for tms_name in self.tms_players_list:
            tms_lower = tms_name.lower().strip()
            max_len = max(len(ocr_lower), len(tms_lower))
            if max_len == 0:
                continue
            for variant in variants:
                dist = self._levenshtein_distance(variant, tms_lower)
                sim = 1.0 - (dist / max_len)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = tms_name

        if best_similarity >= min_similarity:
            return best_match, best_similarity, True
        return ocr_clean, best_similarity, False
    
    def _extract_names_with_ocr(self, cropped_image, preprocessing_strategy='default'):
        """Extract names using PaddleOCR with specified preprocessing and robust error handling."""
        if cropped_image is None or cropped_image.size == 0 or not self._ensure_ocr():
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
    
    def _is_plausible_killfeed_crop(self, cropped_image):
        """Reject obvious non-killfeed YOLO boxes (wrong shape/size)."""
        if cropped_image is None or cropped_image.size == 0:
            return False
        h, w = cropped_image.shape[:2]
        if w < 80 or h < 16 or h > 140:
            return False
        return (w / max(h, 1)) >= 3.0

    def _extract_names(self, cropped_image, log_failures=True):
        """
        Robust name extraction with STRICT left-to-right ordering (killer → victim).
        Uses sequential text reading to ensure correct sequence and prevents swapping.
        Applies fuzzy matching against TMS players list for canonical name replacement.
        """
        if cropped_image is None or cropped_image.size == 0:
            return "", "", 0.0
        
        if not self._ensure_ocr():
            return "", "", 0.0
        
        # Use new left-to-right extraction method with strict validation
        try:
            # Save to temp file for processing
            temp_path = None
            try:
                temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                cv2.imwrite(temp_path, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
                # Load image and extract sequence (with strict ordering)
                image = cv2.imread(temp_path)
                if image is not None:
                    result = self.text_detector.extract_killfeed_sequence(image)
                    
                    killer_raw = result.get('killer', '')
                    victim_raw = result.get('victim', '')
                    confidence = result.get('confidence', 0.0)
                    
                    if not killer_raw or not victim_raw:
                        if log_failures:
                            if not killer_raw and not victim_raw:
                                print(f"⚠️  Left-to-right extraction failed - No text regions detected (OCR found 0 names)")
                            elif not killer_raw:
                                print(f"⚠️  Left-to-right extraction failed - Killer name missing (victim: '{victim_raw}')")
                            else:
                                print(f"⚠️  Left-to-right extraction failed - Victim name missing (killer: '{killer_raw}')")
                    else:
                        # Normalize names first
                        killer_raw = self._normalize_name(killer_raw)
                        victim_raw = self._normalize_name(victim_raw)
                        
                        # STRICT VALIDATION: Ensure names are different and valid
                        if killer_raw.upper() == victim_raw.upper():
                            if log_failures:
                                print(f"⚠️  Left-to-right extraction failed - Names are identical: '{killer_raw}' == '{victim_raw}'")
                        elif len(killer_raw) < self.min_name_length:
                            if log_failures:
                                print(f"⚠️  Left-to-right extraction failed - Killer name too short: '{killer_raw}' (min: {self.min_name_length})")
                        elif len(victim_raw) < self.min_name_length:
                            if log_failures:
                                print(f"⚠️  Left-to-right extraction failed - Victim name too short: '{victim_raw}' (min: {self.min_name_length})")
                        else:
                            # FUZZY MATCHING: Match against TMS players list
                            killer_canonical, killer_sim, killer_matched = self._fuzzy_match_player_name(killer_raw)
                            victim_canonical, victim_sim, victim_matched = self._fuzzy_match_player_name(victim_raw)
                            
                            # Log matching results
                            if killer_matched:
                                print(f"   ✅ Killer matched: '{killer_raw}' → '{killer_canonical}' (similarity: {killer_sim:.2f})")
                            elif killer_sim > 0.5:
                                print(f"   ⚠️  Killer low-confidence: '{killer_raw}' (best match similarity: {killer_sim:.2f})")
                            
                            if victim_matched:
                                print(f"   ✅ Victim matched: '{victim_raw}' → '{victim_canonical}' (similarity: {victim_sim:.2f})")
                            elif victim_sim > 0.5:
                                print(f"   ⚠️  Victim low-confidence: '{victim_raw}' (best match similarity: {victim_sim:.2f})")
                            
                            # Use canonical names (or raw if no match)
                            killer = killer_canonical
                            victim = victim_canonical
                            
                            # Apply temporal smoothing (preserves order)
                            smoothed_killer, smoothed_victim, smoothed_conf = self._apply_temporal_smoothing(
                                killer, victim, confidence
                            )
                            
                            # Final validation: ensure order is maintained
                            if smoothed_killer and smoothed_victim and smoothed_killer.upper() != smoothed_victim.upper():
                                return smoothed_killer, smoothed_victim, smoothed_conf
                            else:
                                if log_failures:
                                    print(f"⚠️  Left-to-right extraction failed - Temporal smoothing returned invalid result (killer: '{smoothed_killer}', victim: '{smoothed_victim}')")
                else:
                    if log_failures:
                        print(f"⚠️  Left-to-right extraction failed - Could not load image from temp file")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        except Exception as e:
            if log_failures:
                print(f"⚠️  Error in left-to-right extraction: {type(e).__name__}: {str(e)}")
            import traceback
            # Only print full traceback in debug mode to avoid spam
            if hasattr(self, 'debug_mode') and self.debug_mode:
                traceback.print_exc()
        
        # Fallback: Try old method with multiple strategies
        # NOTE: This fallback should also respect left-to-right ordering
        # But since extract_killfeed_sequence failed, we'll skip fallback to avoid potential swaps
        # (The detailed error message was already printed above)
        return "", "", 0.0
    
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
    
    def _init_crop_output_dir(self):
        """Create single output folder for all verified killfeed crop saves."""
        output_dir = "cropkillblock"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Killfeed crops: {output_dir}/")
        return output_dir

    def _init_status_detected_dir(self):
        """Test output folder: sequence + status only (no player names)."""
        output_dir = "status_detected"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Status test crops: {os.path.abspath(output_dir)}")
        return output_dir

    def _status_to_test_label(self, status):
        """Map status to short test filename label: revive / kill / knock."""
        normalized = (status or "").lower().strip()
        if normalized == 'revive':
            return 'revive'
        if normalized == 'kill':
            return 'kill'
        if normalized in ('gun knockout', 'knockout', 'knock', 'knocked'):
            return 'knock'
        return None

    def _is_duplicate_status_detected(self, crop_image):
        """True if this visual event was already saved under status_detected/ (debug only)."""
        signature = self._compute_killfeed_signature(crop_image)
        if signature is None:
            return True
        with self.visual_dedup_lock:
            self._prune_visual_caches()
            for entry in self.status_detected_visual_cache:
                if self._signatures_match(signature, entry['signature']):
                    return True
        return False

    def _register_status_detected_event(self, crop_image):
        """Remember status_detected debug save — does not block API/cropkillblock."""
        signature = self._compute_killfeed_signature(crop_image)
        if signature is None:
            return
        with self.visual_dedup_lock:
            self._prune_visual_caches()
            for entry in self.status_detected_visual_cache:
                if self._signatures_match(signature, entry['signature']):
                    entry['registered_at'] = time.time()
                    return
            self.status_detected_visual_cache.append({
                'signature': signature,
                'registered_at': time.time(),
            })

    def _is_duplicate_saved_killfeed(self, crop_image):
        """True only if this visual event was already saved to cropkillblock/ and sent to API."""
        signature = self._compute_killfeed_signature(crop_image)
        if signature is None:
            return True
        with self.visual_dedup_lock:
            self._prune_visual_caches()
            for entry in self.saved_visual_cache:
                if self._signatures_match(signature, entry['signature']):
                    return True
        return False

    def _save_status_detected_image(self, cropped_image, status):
        """
        Save one test crop per unique killfeed event: 001_revive.png, 002_kill.png, etc.
        Only blocks re-save of the same visual event (not in-flight queue cache).
        """
        if cropped_image is None or cropped_image.size == 0:
            return False

        test_label = self._status_to_test_label(status)
        if test_label is None:
            return False

        if self._is_duplicate_status_detected(cropped_image):
            with self.stats_lock:
                self.stats['crops_skipped_duplicate'] += 1
            return False

        with self.status_detected_lock:
            self.status_detected_sequence += 1
            sequence = self.status_detected_sequence

        filename = f"{sequence:03d}_{test_label}.png"
        try:
            os.makedirs(self.status_detected_dir, exist_ok=True)
            filepath = os.path.join(self.status_detected_dir, filename)
            cv2.imwrite(filepath, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self._register_status_detected_event(cropped_image)
            with self.stats_lock:
                self.stats['status_detected_saved'] += 1
            abs_path = os.path.abspath(filepath)
            if getattr(self, 'verbose_status_saves', False):
                print(f"📸 Status crop saved: {abs_path}")
            return True
        except Exception as e:
            print(f"⚠️ Status test save error: {e}")
            return False
    
    def _status_to_filename_label(self, status):
        """Map classified status to filename label."""
        normalized = (status or "").lower().strip()
        if normalized == "kill":
            return "KILL"
        if normalized in ("gun knockout", "knockout", "knock", "knocked"):
            return "KNOCK"
        if normalized == "revive":
            return "REVIVE"
        if normalized in ("eliminate", "eliminated", "elimination"):
            return "ELIMINATE"
        return None
    
    def _is_verified_killfeed(self, result):
        """True only when OCR names and status classification are both complete."""
        if not result.get('names_validated'):
            return False
        killer = result.get('killer_name') or ""
        victim = result.get('victim_name') or ""
        if not killer or not victim or killer == "UNKNOWN" or victim == "UNKNOWN":
            return False
        return self._status_to_filename_label(result.get('status')) is not None
    
    def _save_image(self, image, filename):
        """Save verified crop into cropkillblock/."""
        try:
            os.makedirs(self.crop_output_dir, exist_ok=True)
            filepath = os.path.join(self.crop_output_dir, filename)
            cv2.imwrite(filepath, image)
            return filepath
        except Exception as e:
            print(f"⚠️ Save error: {e}")
            return None
    
    def _crop_hash(self, crop_image):
        if crop_image is None or crop_image.size == 0:
            return None
        return hashlib.md5(crop_image.tobytes()).hexdigest()

    def _normalize_killfeed_for_signature(self, crop_image):
        """Resize killfeed crop to a stable grayscale representation for visual matching."""
        if crop_image is None or crop_image.size == 0:
            return None
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        target_w = 128
        target_h = max(16, int(round(h * (target_w / max(w, 1)))))
        normalized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
        return normalized

    def _compute_dhash(self, gray_image):
        """Difference hash (64-bit) — tolerant to background/compression noise."""
        if gray_image is None:
            return 0
        resized = cv2.resize(gray_image, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bit)
        return value

    def _compute_hist_signature(self, gray_image):
        """Compact grayscale histogram signature."""
        if gray_image is None:
            return None
        hist = cv2.calcHist([gray_image], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _compute_killfeed_signature(self, crop_image):
        """
        Visual fingerprint for duplicate killfeed detection.
        Uses both the killer+weapon region (left ~72%) AND the victim name region
        (right portion) so that different victims killed by the same player with
        the same weapon are never confused as duplicate events.
        """
        gray = self._normalize_killfeed_for_signature(crop_image)
        if gray is None:
            return None
        h, w = gray.shape[:2]
        struct_w = max(24, int(w * 0.72))
        gray_struct = gray[:, :struct_w]
        edges_struct = cv2.Canny(gray_struct, 40, 120)

        # Victim name region: ~48%-88% of normalised width.
        # Stored as a grayscale patch so _signatures_match can run a spatial
        # correlation (much more discriminative than a 9×8 dHash for text).
        victim_x1 = max(0, int(w * 0.48))
        victim_x2 = min(w, int(w * 0.88))
        gray_victim = gray[:, victim_x1:victim_x2] if victim_x2 > victim_x1 + 4 else None

        return {
            'dhash': self._compute_dhash(gray_struct),
            'dhash_edge': self._compute_dhash(edges_struct),
            'hist': self._compute_hist_signature(gray_struct),
            'gray': gray_struct,
            'dhash_full': self._compute_dhash(gray),
            'hist_full': self._compute_hist_signature(gray),
            # Keep coarse dhash as a fast pre-filter; gray_victim is the precise check.
            'dhash_victim': self._compute_dhash(gray_victim) if gray_victim is not None else 0,
            'gray_victim': gray_victim,
            'exact': self._crop_hash(crop_image),
        }

    def _hamming_distance(self, hash_a, hash_b):
        return (hash_a ^ hash_b).bit_count()

    def _hist_similarity(self, hist_a, hist_b):
        if hist_a is None or hist_b is None:
            return 0.0
        return float(cv2.compareHist(
            hist_a.reshape(-1, 1).astype(np.float32),
            hist_b.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL,
        ))

    def _gray_correlation(self, gray_a, gray_b):
        if gray_a is None or gray_b is None:
            return 0.0
        a = gray_a.astype(np.float32)
        b = cv2.resize(gray_b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        return float(np.sum(a * b) / denom)

    def _signatures_match(self, sig_a, sig_b):
        """True when two killfeed crops represent the same on-screen event.

        A match requires BOTH the killer+weapon region (left ~72%) AND the
        victim name region to be visually similar.  This prevents the common
        false-positive where the same player kills two different victims with
        the same weapon: the structural (left) hash is identical but the
        victim (right) region differs — those are NOT duplicates.
        """
        if sig_a is None or sig_b is None:
            return False
        if sig_a.get('exact') and sig_a['exact'] == sig_b.get('exact'):
            return True

        dham = self._hamming_distance(sig_a['dhash'], sig_b['dhash'])
        hist_sim = self._hist_similarity(sig_a['hist'], sig_b['hist'])
        gray_corr = self._gray_correlation(sig_a['gray'], sig_b['gray'])
        dham_edge = self._hamming_distance(
            sig_a.get('dhash_edge', sig_a['dhash']),
            sig_b.get('dhash_edge', sig_b['dhash']),
        )
        dham_full = self._hamming_distance(
            sig_a.get('dhash_full', sig_a['dhash']),
            sig_b.get('dhash_full', sig_b['dhash']),
        )
        hist_full = self._hist_similarity(
            sig_a.get('hist_full', sig_a['hist']),
            sig_b.get('hist_full', sig_b['hist']),
        )
        # Victim-region discriminator.
        # Use spatial gray-correlation when the full victim patch is available —
        # it is far more reliable than a 9×8 dHash for distinguishing different
        # player names whose text pixels fall in different spatial positions.
        # Same event with a colour change (red kill → white knock due to OBS
        # compression) still has the same text positions → correlation stays high.
        # Different victims have different letter shapes → correlation drops.
        gv_a = sig_a.get('gray_victim')
        gv_b = sig_b.get('gray_victim')
        if gv_a is not None and gv_b is not None:
            victim_corr = self._gray_correlation(gv_a, gv_b)
            victim_differs = victim_corr < 0.55
        else:
            # Fall back to coarse dhash when patch is unavailable (legacy cache entries)
            dham_victim = self._hamming_distance(
                sig_a.get('dhash_victim', 0),
                sig_b.get('dhash_victim', 0),
            )
            victim_differs = dham_victim > 18

        # Primary: killer + weapon structure matches — only a duplicate when the
        # victim region also matches (same text, possibly different colour).
        if dham <= self.visual_dhash_threshold and hist_sim >= self.visual_hist_threshold:
            return not victim_differs
        if dham_edge <= 10 and hist_sim >= 0.75:
            return not victim_differs
        # Structural hash alone: must also require victim region similarity
        if dham <= 8 and not victim_differs:
            return True
        if gray_corr >= self.visual_corr_threshold and dham <= 16:
            return not victim_differs
        if hist_sim >= 0.90 and dham <= 20:
            return not victim_differs
        # Secondary: full banner is very close (same frame / tiny animation jitter)
        if dham_full <= 14 and hist_full >= 0.88:
            return True
        if dham <= 18 and dham_full <= 22 and hist_sim >= 0.84:
            return not victim_differs
        if dham_edge <= 8 and dham_full <= 24:
            return not victim_differs
        return False

    def _prune_visual_caches(self, now=None):
        now = now or time.time()
        cutoff = now - self.killfeed_event_cooldown
        while self.visual_event_cache and self.visual_event_cache[0]['registered_at'] < cutoff:
            entry = self.visual_event_cache.popleft()
            exact = entry.get('signature', {}).get('exact')
            if exact in self.exact_crop_hashes:
                self.exact_crop_hashes.discard(exact)
        while self.saved_visual_cache and self.saved_visual_cache[0]['registered_at'] < cutoff:
            self.saved_visual_cache.popleft()
        while self.status_detected_visual_cache and self.status_detected_visual_cache[0]['registered_at'] < cutoff:
            self.status_detected_visual_cache.popleft()

    def _is_duplicate_killfeed_event(self, crop_image, check_saved=True, register=False):
        """
        Visual duplicate check BEFORE OCR/status/save.
        Matches same killfeed across frames despite background/compression changes.
        """
        signature = self._compute_killfeed_signature(crop_image)
        if signature is None:
            return True

        with self.visual_dedup_lock:
            self._prune_visual_caches()
            exact = signature.get('exact')
            if exact in self.exact_crop_hashes:
                return True

            caches = [self.visual_event_cache]
            if check_saved:
                caches.append(self.saved_visual_cache)

            for cache in caches:
                for entry in cache:
                    if self._signatures_match(signature, entry['signature']):
                        return True

            if register:
                self.visual_event_cache.append({
                    'signature': signature,
                    'registered_at': time.time(),
                })
                if exact:
                    self.exact_crop_hashes.add(exact)
                if len(self.exact_crop_hashes) > 300:
                    self.exact_crop_hashes = {
                        e.get('signature', {}).get('exact')
                        for e in list(self.visual_event_cache) + list(self.saved_visual_cache)
                        if e.get('signature', {}).get('exact')
                    }
            return False

    def _register_saved_killfeed_event(self, crop_image):
        """Remember a successfully saved killfeed to block repeat saves of the same UI event."""
        signature = self._compute_killfeed_signature(crop_image)
        if signature is None:
            return
        with self.visual_dedup_lock:
            self._prune_visual_caches()
            for entry in self.saved_visual_cache:
                if self._signatures_match(signature, entry['signature']):
                    entry['registered_at'] = time.time()
                    return
            self.saved_visual_cache.append({
                'signature': signature,
                'registered_at': time.time(),
            })
            exact = signature.get('exact')
            if exact:
                self.exact_crop_hashes.add(exact)
    
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
        """Flush heap entries in strict detection-number order. Caller must hold heap_lock.

        Safety watchdog: if the heap's smallest entry is more than 10 positions
        ahead of next_expected_frame, the expected numbers were never pushed
        (lost before reaching the heap).  Advance next_expected_frame to unblock.
        """
        # Watchdog: skip over any detection numbers that were permanently lost
        if self.output_heap:
            smallest_waiting = self.output_heap[0][0]
            if smallest_waiting > self.next_expected_frame + 10:
                # Gap too large — advance past the missing numbers
                self.next_expected_frame = smallest_waiting

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
    
    def _save_kill_block(self, result, detection_number):
        """Save only verified killfeeds (OCR names + status) into cropkillblock/."""
        cropped_image = result.get('cropped_image')
        if cropped_image is None or cropped_image.size == 0:
            return False
        
        killer_name = result.get('killer_name') or "UNKNOWN"
        victim_name = result.get('victim_name') or "UNKNOWN"
        status = result.get('status', 'UNKNOWN')
        confidence = float(result.get('confidence', 0.0))
        suspect_swap = result.get('suspect_swap', False)
        names_validated = result.get('names_validated', False)
        
        if not self._is_verified_killfeed(result):
            with self.stats_lock:
                self.stats['crops_skipped_unverified'] += 1
            if killer_name == "UNKNOWN" and victim_name == "UNKNOWN":
                with self.stats_lock:
                    self.stats['ocr_no_text'] += 1
                print(
                    f"⏭️  YOLO false positive #{detection_number + 1} — "
                    f"no player names in crop (status={status})"
                )
            else:
                print(
                    f"⏭️  Skipped unverified crop #{detection_number + 1} "
                    f"(killer={killer_name}, victim={victim_name}, status={status})"
                )
            return False

        if self._is_duplicate_saved_killfeed(cropped_image):
            with self.stats_lock:
                self.stats['crops_skipped_duplicate'] += 1
            print(
                f"⏭️  Skipped duplicate OCR save #{detection_number + 1} "
                f"(killer={killer_name}, victim={victim_name}, status={status})"
            )
            return False
        
        status_label = self._status_to_filename_label(status)
        sequence_number = detection_number + 1
        safe_killer = self._sanitize_filename_part(killer_name)
        safe_victim = self._sanitize_filename_part(victim_name)
        filename = f"{safe_killer}_{status_label}_{safe_victim}.png"
        filepath = self._save_image(cropped_image, filename)
        
        if not filepath:
            return False
        
        rel_path = os.path.join(self.crop_output_dir, filename)
        print(f"📸 Saved: {rel_path}")
        print(f"   🎯 Killer: {killer_name} | 📊 Status: {status_label} | 👤 Victim: {victim_name}")
        print(f"   ✅ Confidence: {confidence:.2f} | Verified: Yes | Order: {killer_name} → {status_label} → {victim_name}")
        if suspect_swap:
            print(f"   ⚠️  WARNING: Suspect swap detected - verify manually if needed")
        
        with self.stats_lock:
            self.stats['crops_saved'] += 1
        self._register_saved_killfeed_event(cropped_image)
        
        skip_api_duplicate = self._is_name_cooldown_duplicate(killer_name, victim_name, status)
        if skip_api_duplicate:
            print(f"   🚫 Name cooldown duplicate - saved image, skipping API/log score")
        
        if self.api_push_enabled and not skip_api_duplicate and names_validated:
            api_request = {
                'player_name': victim_name,
                'enemy_name': killer_name,
                'status': status,
                'image_path': filepath,
                'sequence_number': sequence_number
            }
            try:
                self.api_queue.put_nowait(api_request)
                with self.stats_lock:
                    self.stats['detections_processed'] += 1
            except:
                with self.stats_lock:
                    self.stats['queue_drops'] += 1
                print(f"⚠️ API queue full - dropped detection #{sequence_number}")
        
        if killer_name != "UNKNOWN":
            with self.processing_lock:
                self.known_players.add(killer_name)
        if victim_name != "UNKNOWN":
            with self.processing_lock:
                self.known_players.add(victim_name)
        
        with self.processing_lock:
            self.detection_sequence = max(self.detection_sequence, sequence_number)
        
        return True
    
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

        if not self._is_plausible_killfeed_crop(cropped_image):
            with self.stats_lock:
                self.stats['yolo_rejected_shape'] += 1
            return False
        
        if self._is_duplicate_killfeed_event(cropped_image, check_saved=True, register=True):
            with self.stats_lock:
                self.stats['crops_skipped_duplicate'] += 1
            return False

        # Status + status_detected save happen immediately (before OCR queue)
        status = self._determine_status_local(
            cropped_image,
            result.get('class_name'),
            has_revive=result.get('has_revive', False),
            verbose=False,
        )
        result['status'] = status
        result['status_detected_saved'] = self._save_status_detected_image(cropped_image, status)
        
        with self.processing_lock:
            detection_number = self.next_detection_number
            self.next_detection_number += 1
        result['detection_number'] = detection_number
        
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
    
    def _load_yolo_conf_threshold(self):
        """Load YOLO confidence from config.json when available."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                return float(config.get("detection", {}).get("confidence_threshold", 0.35))
        except Exception:
            pass
        return 0.35

    def _load_runtime_config(self):
        """Optional logging toggles from config.json."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                det_cfg = config.get("detection", {})
                self.verbose_status_saves = bool(det_cfg.get("verbose_status_saves", False))
        except Exception:
            pass

    def _count_killblock_detections(self, image):
        """Return number of killblock/revive boxes YOLO finds in a frame."""
        return len(self._detect_killblocks_and_revives_local(image))

    @staticmethod
    def _open_video_capture(index):
        """Open camera with platform-appropriate backend (CAP_DSHOW is Windows-only)."""
        if sys.platform == "win32":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(index)

    def _probe_camera_index(self):
        """Pick the camera that shows game content with visible killfeeds."""
        if self.local_model is None:
            return self.camera_index if self.camera_index is not None else 1

        print("🔎 Scanning camera indices for OBS Virtual Camera with killfeeds...")
        best_idx = self.camera_index if self.camera_index is not None else 1
        best_score = -1
        for idx in range(0, 6):
            cap = None
            try:
                cap = self._open_video_capture(idx)
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(4):
                    cap.read()
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print(f"   Camera {idx}: no frame")
                    continue
                brightness = float(frame.mean())
                if brightness < 8:
                    print(f"   Camera {idx}: black/empty frame (brightness={brightness:.1f})")
                    continue
                hits = self._count_killblock_detections(frame)
                h, w = frame.shape[:2]
                print(f"   Camera {idx}: {w}x{h}, brightness={brightness:.1f}, yolo_hits={hits}")
                if hits > best_score:
                    best_score = hits
                    best_idx = idx
            except Exception:
                continue
            finally:
                if cap is not None:
                    cap.release()

        if best_score <= 0:
            print(
                f"⚠️  No killfeeds found on any camera during probe — using camera {best_idx}. "
                "Ensure OBS Virtual Camera shows the game with killfeed visible."
            )
        else:
            print(f"✅ Selected camera {best_idx} ({best_score} killfeed detection(s) in probe frame)")
        return best_idx

    def _initialize_camera(self):
        """Initialize OBS Virtual Camera."""
        try:
            if self.cap is not None:
                self.cap.release()

            if self.camera_index is None:
                self.camera_index = self._probe_camera_index()

            self.cap = self._open_video_capture(self.camera_index)

            if not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(5):
                self.cap.read()

            ret, test_frame = self.cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                h, w = test_frame.shape[:2]
                print(f"📷 Camera {self.camera_index} active: {w}x{h} | YOLO conf={self.yolo_conf_threshold}")
            return True
        except Exception:
            return False

    def _log_no_detection_diagnostics(self, frame, frame_count, yolo_hits=None):
        """Lightweight idle diagnostics — uses cached YOLO hits (no extra model inference)."""
        try:
            if frame is None or frame.size == 0:
                print(f"⚠️  Frame #{frame_count}: camera returned empty frame")
                return

            h, w = frame.shape[:2]
            brightness = float(frame.mean())
            hits = self._last_yolo_hits if yolo_hits is None else yolo_hits
            print(
                f"ℹ️  Frame #{frame_count} diagnostics: camera={self.camera_index}, size={w}x{h}, "
                f"brightness={brightness:.1f}, yolo_killfeed_hits={hits}"
            )
            if hits == 0:
                print(
                    "   No killfeed in this frame yet — still scanning. Check OBS Virtual Camera "
                    "shows the game with killfeed visible."
                )
                try:
                    os.makedirs("debug_capture", exist_ok=True)
                    debug_path = os.path.join("debug_capture", f"frame_{frame_count}.jpg")
                    cv2.imwrite(debug_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    print(f"   Saved debug frame: {debug_path}")
                except Exception:
                    pass
            print("   💤 Still running — waiting for killfeed (Ctrl+C to stop)")
        except Exception as e:
            print(f"⚠️  Diagnostics skipped (frame #{frame_count}): {type(e).__name__}")
    
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
    
    def start_detection(self):
        """Main detection loop: capture → gRPC → OCR → API. Runs forever until KeyboardInterrupt.
        This function will NEVER stop automatically - it will keep running through all errors
        and will only stop when the user manually presses Ctrl+C."""
        self.stop_event.clear()
        print("\n🚀 Starting detection loop...")
        print("⚠️  Press Ctrl+C to stop\n")
        print("💡 System will keep running FOREVER even if errors occur - it will auto-recover\n")
        print("🛡️  Robust mode: Will never discard the match until you manually stop it\n")
        print("📌 No killfeed = keep scanning (never auto-stops on idle)\n")
        
        # Outer loop to restart if main loop crashes
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
                # Loop will restart automatically
                continue
        
        # Cleanup when actually stopping
        self._cleanup()
    
    def _run_detection_loop(self):
        """Inner detection loop - runs until error or KeyboardInterrupt."""
        # Initialize camera
        while not self._initialize_camera():
            print("⚠️ Camera initialization failed, retrying in 2 seconds...")
            time.sleep(2)
        
        print("✅ Camera ready")
        frame_count = 0
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
                    self._last_yolo_hits = 0
                    results = []
                    # Try to reconnect if using gRPC
                    if not self.use_local_model and self.grpc_stub is None:
                        print("🔄 Attempting to reconnect to gRPC server...")
                        self._connect_to_server()
                    continue
                
                if results:
                    try:
                        queued_count = 0
                        for result in results:
                            try:
                                if self._process_detection(result):
                                    queued_count += 1
                            except Exception as e:
                                print(f"⚠️ Error queuing detection: {type(e).__name__}")
                                continue
                        
                        if queued_count > 0:
                            print(f"🎯 YOLO hit at frame #{frame_count} ({queued_count} crop(s)) — OCR verifying...")
                    except Exception as e:
                        # Log but continue loop
                        print(f"⚠️ Error in detection queuing (frame #{frame_count}): {type(e).__name__}")
                        continue
                
                if frame_count % 50 == 0:
                    with self.stats_lock:
                        queue_size = self.detection_queue.qsize()
                        api_queue_size = self.api_queue.qsize()
                        crops_saved = self.stats.get('crops_saved', 0)
                    with self.heap_lock:
                        heap_pending = len(self.output_heap)
                    skipped_dup = self.stats.get('crops_skipped_duplicate', 0)
                    status_saved = self.stats.get('status_detected_saved', 0)
                    ocr_no_text = self.stats.get('ocr_no_text', 0)
                    yolo_rejected = self.stats.get('yolo_rejected_shape', 0)
                    print(
                        f"⏳ Running... Frame #{frame_count} | Queue: {queue_size} | "
                        f"Heap: {heap_pending} | API: {api_queue_size} | "
                        f"Status test: {status_saved} | OCR saved: {crops_saved} | "
                        f"Dup skipped: {skipped_dup} | False YOLO: {ocr_no_text} | "
                        f"Bad shape: {yolo_rejected}"
                    )
                    if crops_saved == 0 and skipped_dup == 0 and queue_size == 0:
                        try:
                            self._log_no_detection_diagnostics(
                                frame, frame_count, yolo_hits=self._last_yolo_hits
                            )
                        except Exception as diag_error:
                            print(
                                f"⚠️  Diagnostics error (frame #{frame_count}): "
                                f"{type(diag_error).__name__} — continuing"
                            )
                
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
        
        print("⚠️ Inner detection loop ended — outer loop will restart if still active")
        
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
                        # CRITICAL: we MUST call _push_result for every detection_num.
                        # If we skip a number, next_expected_frame never advances and the
                        # entire heap is frozen — all subsequent killfeeds silently vanish.
                        try:
                            processed = None
                            try:
                                processed = fut.result()
                            except Exception:
                                processed = None
                            if processed is None:
                                processed = self._fallback_result_payload(original)
                            if detection_num is not None:
                                self._push_result(detection_num, processed)
                        except Exception:
                            # Last-resort: push a bare sentinel so the heap unblocks
                            try:
                                if detection_num is not None:
                                    self._push_result(detection_num, self._fallback_result_payload(original))
                            except Exception:
                                pass
                    
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
                
                # Send API request asynchronously
                try:
                    future = self.api_executor.submit(self._send_to_api_with_retry, **api_request)
                    # Don't wait - let it send in background
                    self.api_queue.task_done()
                    error_count = 0  # Reset error count on success
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
            detection_number = result.get('detection_number')
            
            if cropped_image is None or cropped_image.size == 0:
                return None

            # Status already set + saved to status_detected/ in _process_detection
            status = result.get('status')
            if not status or status == 'UNKNOWN':
                status = self._determine_status_local(
                    cropped_image,
                    result.get('class_name'),
                    has_revive=result.get('has_revive', False),
                    verbose=False,
                )
                result['status'] = status
                if not result.get('status_detected_saved'):
                    self._save_status_detected_image(cropped_image, status)

            # OCR for cropkillblock / API
            killer_name, victim_name, confidence = self._extract_names(cropped_image, log_failures=False)
            
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
                retry_killer, retry_victim, retry_conf = self._extract_names(
                    cropped_image, log_failures=False
                )
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

            if names_validated:
                print(
                    f"✅ Killfeed #{detection_number + 1}: "
                    f"{killer_name} → {victim_name} ({status})"
                )
            
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
    
    def _send_to_api_with_retry(self, player_name, enemy_name, status, image_path, sequence_number, max_retries=3):
        """
        Send killblock data to TMS API with retry logic.
        Ensures instant updates to TMS panel for real-time killfeed display.
        """
        if not self.api_push_enabled:
            return False
        
        for attempt in range(max_retries):
            try:
                # Convert image to base64
                with open(image_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                
                # Payload with correct order: killer → status → victim
                # TMS API expects: killerName (gets kill/knockout points), victimName (gets death)
                # player_name is victim, enemy_name is killer (as per killfeed display)
                payload = {
                    "killerName": enemy_name,
                    "victimName": player_name,
                    "weaponUsed": status,
                    "imagePath": image_path,
                    "siftWeapon": "",
                    "image": base64_image
                }
                
                headers = {
                    'accept': 'text/plain', 
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache'  # Ensure fresh data
                }
                if self.access_token:
                    headers['Authorization'] = f'Bearer {self.access_token}'
                
                # Use adaptive timeout: longer for first attempt, shorter for retries
                # Increased initial timeout to handle slow TMS server responses
                timeout = 10 if attempt == 0 else 5
                response = self.api_session.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code in [200, 201]:
                    print(f"✅ API: Sent killblock #{sequence_number} to TMS | {enemy_name} {status} {player_name}")
                    print(f"   🚀 TMS panel updated instantly - {enemy_name} gets {status} points, {player_name} gets death")
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
                        body = response.text[:200] if response.text else ""
                        print(f"⚠️ API: Request failed (Status: {response.status_code}) after {max_retries} attempts")
                        if body:
                            print(f"   Response: {body}")
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
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for queues to drain (with timeout)
        try:
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
            if hasattr(self, 'detection_worker'):
                self.detection_worker.join(timeout=2)
            if hasattr(self, 'api_worker'):
                self.api_worker.join(timeout=2)
        except:
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
            print(f"   Status test saved: {self.stats['status_detected_saved']}")
            print(f"   Verified crops saved: {self.stats['crops_saved']}")
            print(f"   Crops skipped (unverified): {self.stats['crops_skipped_unverified']}")
            print(f"   Crops skipped (duplicate): {self.stats['crops_skipped_duplicate']}")
        
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


def run_obs_capture(
    match_id=1,
    access_token=None,
    camera_index=1,
    stop_flag=None,
    use_local_model=True,
    grpc_port=50051,
    model_path="best (1).pt",
):
    """
    Entry point for 16score-desktop when Free Fire is selected.

    Runs ffkillblock.py with the local YOLO model (best (1).pt by default).
    """
    print(
        f"[ffkillblock] match={match_id} | camera={camera_index} | "
        f"local_model={use_local_model} | weights={model_path}",
        flush=True,
    )
    detector = KillblockDetector(
        match_id=str(match_id),
        access_token=access_token,
        api_enabled=True,
        grpc_port=grpc_port,
        use_local_model=use_local_model,
        model_path=model_path,
        camera_index=camera_index,
        stop_flag=stop_flag,
    )
    if use_local_model and detector.local_model is None:
        raise RuntimeError(f"Failed to load YOLO model: {model_path}")
    print("[ffkillblock] Detector ready — starting camera capture", flush=True)
    detector.start_detection()


def main():
    """Main entry point."""
    print("=== Free Fire Killblock Detector ===")
    print("")
    print("💡 Processing Modes:")
    print("   - Local mode: Uses YOLO model directly (faster, no server needed)")
    print("   - Server mode: Uses gRPC server (requires port 50051)")
    print("")
    
    # Get configuration
    match_id = input("Enter Match ID (or press Enter for default '1'): ").strip() or "1"
    access_token = input("Enter Access Token (or press Enter to skip): ").strip() or None
    api_enabled = input("Enable API integration? (Y/n, default: Y): ").strip().lower() != 'n'
    
    # Ask for processing mode
    use_local = input("Use local model? (Y/n, default: Y): ").strip().lower() != 'n'
    model_path = "best (1).pt"
    if use_local:
        custom_model = input(f"Model path (or press Enter for '{model_path}'): ").strip()
        if custom_model:
            model_path = custom_model

    camera_index = None
    cam_input = input("OBS camera index (Enter=auto-detect, or 0/1/2): ").strip()
    if cam_input:
        try:
            camera_index = int(cam_input)
        except ValueError:
            print("⚠️  Invalid camera index — using auto-detect")
    
    try:
        detector = KillblockDetector(
            match_id=match_id, 
            access_token=access_token, 
            api_enabled=api_enabled,
            use_local_model=use_local,
            model_path=model_path,
            camera_index=camera_index,
        )
        
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
        
        print("\n📹 Please start OBS Virtual Camera")
        input("Press Enter to begin detection...")
        
        while True:
            try:
                detector.start_detection()
                break
            except KeyboardInterrupt:
                print("\n⏹️ Interrupted by user")
                break
            except Exception as e:
                print(f"\n⚠️ Detection crashed: {type(e).__name__}: {str(e)[:200]}")
                print("🔄 Auto-restarting in 3 seconds... (Ctrl+C now to quit)")
                try:
                    time.sleep(3)
                except KeyboardInterrupt:
                    print("\n⏹️ Interrupted by user")
                    break
                detector.stop_event.clear()
                if detector.cap is not None:
                    try:
                        detector.cap.release()
                    except Exception:
                        pass
                    detector.cap = None
                continue
    
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
    import faulthandler
    faulthandler.enable()
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
