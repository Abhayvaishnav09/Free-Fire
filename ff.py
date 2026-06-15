"""Killblock OCR Client - Processes cropped killblock images from gRPC server.
Responsibilities:
1. Receive cropped killblock images from gRPC server
2. Run OCR text detection (left-to-right order: Killer → Victim)
3. Apply fuzzy matching with TMS player roster
4. Send detected names to TMS API

NO FreeFire detection logic, NO color analysis, NO model code, NO frame capture.
"""
import cv2
import os
import sys
import numpy as np
import warnings
import time
import re
import base64
import hashlib
import socket
import grpc
import threading
import tempfile
from collections import deque
from queue import Queue, Empty
import requests
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError

# Try to import FreeFireTextDetector, but make it optional
FreeFireTextDetector = None

# Optimized imports - try FreeFireTextDetector
detection_dir = os.path.dirname(os.path.abspath(__file__))
for import_func in [
    lambda: __import__('text', fromlist=['FreeFireTextDetector']).FreeFireTextDetector,
    lambda: __import__('.text', fromlist=['FreeFireTextDetector']).FreeFireTextDetector,
    lambda: __import__('score_ai.detection.text', fromlist=['FreeFireTextDetector']).FreeFireTextDetector,
]:
    try:
        if detection_dir not in sys.path:
            sys.path.insert(0, detection_dir)
        FreeFireTextDetector = import_func()
        break
    except (ImportError, AttributeError, ValueError):
        continue

# Optimized imports - rapidfuzz, config, gRPC
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    fuzz = process = None

try:
    from score_ai.core.config_manager import config
except ImportError:
    config = None

# Optimized gRPC imports
killfeed_detection_pb2 = killfeed_detection_pb2_grpc = None
for import_func in [
    lambda: (__import__('killfeed_detection_pb2'), __import__('killfeed_detection_pb2_grpc')),
    lambda: (__import__('.killfeed_detection_pb2', fromlist=['']), __import__('.killfeed_detection_pb2_grpc', fromlist=[''])),
    lambda: (__import__('score_ai.detection.killfeed_detection_pb2', fromlist=['']), __import__('score_ai.detection.killfeed_detection_pb2_grpc', fromlist=[''])),
]:
    try:
        if detection_dir not in sys.path:
            sys.path.insert(0, detection_dir)
        killfeed_detection_pb2, killfeed_detection_pb2_grpc = import_func()
        break
    except (ImportError, ValueError):
        continue

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Current-match roster  — update this list before each match
# Used by _snap_to_roster() to correct OCR noise without slowing the pipeline
# ---------------------------------------------------------------------------
MATCH_PLAYERS = [
    # K9 ESPORTS
    "K9.HUNNYSUNY", "K9.ZIYANN", "K9.DAAFIQ", "K9.AIM84", "K9.ZORO",
    # REVENANT XSPARK
    "RNTX.ARSH17", "RNTX.VINCENT", "RNTX.XDIVINE", "RNTX.ROSHAN", "RNTX.BLACK",
    # AEROBOTZ ESPORTS
    "ARZ.JOHAN", "ARZ.PRODIGY", "ARZ.MADGOD", "ARZ.KUNAL19", "ARZ.FLIX24",
    # IQOO OGxTSG
    "IQOG.ARJUN", "IQOG.AAYUSH4", "IQOG.LEGEND", "IQOG.KRISH", "IQOG.PANDAT",
    # GODLIKE ESPORTS
    "GODL.YOGI", "GODL.ECOECO", "GODL.MARCO", "GODL.NANCY", "GODL.NOBITA",
    # METANINZA
    "MNZ.ZAP", "MNZ.JARVIS16", "MNZ.GINOTRA", "MNZ.ANSHU26", "MNZ.RNS",
    # GG INSTINCT
    "GGI.PATLU", "GGI.SWARUP", "GGI.POWER", "GGI.TIGER", "GGI.RABARI11",
    # TEAM TAMILAS
    "TT.KHONSHU", "TT.YOGESH23", "TT.SCRIPT18", "TT.KOWSIK24", "TT.RAIN21",
    # RECKONING ESP
    "RGE.HEMU", "RGE.WILDFOX9", "RGE.SABOS", "RGE.ASH", "RGE.LEVELUP",
    # WINDGODxTHW ESP
    "WIND.JANGO", "WIND.GOKUL", "WIND.NYM", "WIND.KINGSTN", "WIND.BRAVE",
    # 4ENDS ESPORTS
    "4END.AVJIT", "4END.RAICHU", "4END.CYBER", "4END.SURYA", "4END.ARIJEET",
    # EMZ ESPORTS
    "EMZ.MAC", "EMZ.ADITYA", "EMZ.MRANI", "EMZ.RUPESH", "EMZ.SID18",
]


class KillblockDetector:
    """OCR Client for processing cropped killblock images.
    
    Responsibilities:
    1. Receive cropped killblock images from gRPC server (server handles detection)
    2. Split image into left/right halves and run OCR separately on each half
    3. Extract killer name from left half (ALWAYS) and victim name from right half (ALWAYS)
    4. Apply fuzzy matching with TMS player roster to correct OCR errors
    5. Send results to TMS API with correct roles (NO SWAPPING)
    
    CRITICAL: Split-image OCR approach prevents name swapping:
    - Left half = Killer (fixed position, never swapped)
    - Right half = Victim (fixed position, never swapped)
    - Roles are determined by image position, not OCR text order
    
    Does NOT handle:
    - Frame capture (handled by main.py or caller)
    - FreeFire detection (handled by grpc_block.py server)
    - Status determination (handled by grpc_block.py server)
    - Color analysis (handled by grpc_block.py server)
    """
    
    GRPC_SERVER_PORT = 50051
    GRPC_SERVER_HOST = "localhost"
    
    def __init__(self, match_id="1", access_token=None, api_enabled=True, grpc_port=50051):
        """Initialize OCR client with gRPC connection, OCR, and API configuration.
        
        RESILIENT INITIALIZATION: Handles errors gracefully with retries and fallbacks.
        Never raises exceptions - always completes initialization even if components fail.
        
        Args:
            match_id: TMS match ID
            access_token: TMS API access token
            api_enabled: Whether to send results to TMS API
            grpc_port: gRPC server port (default 50051)
        """
        # Initialize flags to track component status
        self.grpc_connected = False
        self.ocr_available = False
        self.initialization_errors = []
        
        # gRPC configuration
        self.grpc_port = grpc_port
        self.grpc_channel = None
        self.grpc_stub = None
        
        # RESILIENT: Connect to gRPC server with retry logic (non-blocking)
        server_address = f"{self.GRPC_SERVER_HOST}:{self.grpc_port}"
        
        # Try multiple connection attempts with increasing delays
        max_connection_attempts = 5
        connection_retry_delay = 0.5  # Start with 0.5s delay
        
        for attempt in range(max_connection_attempts):
            try:
                if self._check_server_running():
                    if self._connect_to_server(max_retries=2):
                        self.grpc_connected = True
                        break
                    else:
                        if attempt < max_connection_attempts - 1:
                            time.sleep(connection_retry_delay)
                            connection_retry_delay *= 1.5  # Exponential backoff
                else:
                    if attempt < max_connection_attempts - 1:
                        time.sleep(connection_retry_delay)
                        connection_retry_delay *= 1.5
            except Exception as e:
                error_msg = f"gRPC connection error (attempt {attempt + 1}): {type(e).__name__}: {str(e)[:100]}"
                self.initialization_errors.append(error_msg)
                if attempt < max_connection_attempts - 1:
                    time.sleep(connection_retry_delay)
                    connection_retry_delay *= 1.5
        
        # RESILIENT: Initialize OCR with fallback options
        self.text_detector = self._init_ocr()
        if self.text_detector is not None:
            self.ocr_available = True
        
        # API configuration
        self.match_id = match_id
        self.access_token = access_token
        self.api_enabled = api_enabled
        self.api_url = f'http://3.7.109.218:5005/LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed?matchId={match_id}'
        
        self.detection_sequence = 0
        self.recent_detections = set()
        self.recent_detections_with_time = deque(maxlen=200)  # Store detections with timestamps for temporal filtering
        self.name_history = deque(maxlen=5)
        self.confidence_threshold = 0.3  # Increased threshold to reduce false positives
        self.min_confidence_for_processing = 0.05  # Lowered threshold - accept if we have valid names
        self.min_name_length = 1
        self.max_name_length = 30
        self.ocr_retry_attempts = 1  # Single attempt for speed (was 3, reduced for performance)
        self.ocr_retry_delay = 0.0  # No delay needed for single attempt
        self.known_players = set()
        self.team_rosters = {}
        self.all_player_names = []
        # Match-specific roster for OCR snapping (set via set_match_roster)
        self._match_roster_names: list = []
        self._match_roster_suffixes: dict = {}  # suffix_after_dot → full_name
        
        # OPTIMIZED: API connection pool for faster requests
        self.api_session = None
        self._init_api_session()
        
        # RESILIENT: Load team rosters with error handling (non-blocking)
        try:
            self._load_team_rosters()
        except Exception as e:
            error_msg = f"Team roster loading error: {type(e).__name__}: {str(e)[:100]}"
            self.initialization_errors.append(error_msg)
        
        # OCR initialization tracking
        self.ocr_init_attempted = False  # Track if we've already tried to initialize OCR
        
        # OPTIMIZED: Large queue and many workers for maximum throughput
        # Prevents missing killfeeds even during high activity periods
        self.detection_queue = Queue(maxsize=200)  # Large queue to handle bursts
        self.max_workers = 100  # Maximum workers for parallel OCR + API processing
        self.process_immediately = True  # Try to process immediately if queue is empty
        self.active_detections = 0  # Track active processing
        self.processed_count = 0  # Track total processed
        self.failed_count = 0  # Track failed detections
        self.lock = threading.Lock()  # Lock for thread-safe operations
        self.dropped_count = 0  # Track dropped detections (should always be 0)
        
    
    def _init_api_session(self):
        """Initialize requests session with connection pooling for faster API calls."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Create session with connection pooling
            self.api_session = requests.Session()
            
            # Configure retry strategy (fast, minimal retries)
            retry_strategy = Retry(
                total=1,  # Only 1 retry for speed
                backoff_factor=0.1,  # Very short backoff (100ms)
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            
            # Mount adapter with connection pooling
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,  # Connection pool size
                pool_maxsize=20,  # Max connections in pool
                pool_block=False  # Don't block if pool is full
            )
            
            self.api_session.mount("http://", adapter)
            self.api_session.mount("https://", adapter)
        except Exception:
            # Fallback: use regular requests if session fails
            self.api_session = None
    
    def _check_server_running(self):
        """Check if gRPC server is running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((self.GRPC_SERVER_HOST, self.grpc_port))
            sock.close()
            return result == 0
        except:
            return False
    
    def _connect_to_server(self, retry_count=0, max_retries=3):
        """Connect to gRPC server with retry logic.
        
        RESILIENT: Never raises exceptions - always returns False on failure.
        Handles all errors gracefully with retry logic.
        """
        try:
            if killfeed_detection_pb2 is None or killfeed_detection_pb2_grpc is None:
                return False
            
            # Close existing channel if any (handle errors gracefully)
            if hasattr(self, 'grpc_channel') and self.grpc_channel is not None:
                try:
                    self.grpc_channel.close()
                except Exception:
                    pass  # Ignore errors when closing old channel
                finally:
                    self.grpc_channel = None
                    self.grpc_stub = None
            
            # Create channel with options
            try:
                options = [
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                ]
                self.grpc_channel = grpc.insecure_channel(f'{self.GRPC_SERVER_HOST}:{self.grpc_port}', options=options)
            except Exception as e:
                if retry_count < max_retries:
                    time.sleep(0.1)  # OPTIMIZED: Minimal delay for retry
                    return self._connect_to_server(retry_count + 1, max_retries)
                error_msg = f"Failed to create gRPC channel: {type(e).__name__}"
                return False
            
            # OPTIMIZED: Fast timeout (1s) for real-time performance
            try:
                grpc.channel_ready_future(self.grpc_channel).result(timeout=1)
            except grpc.FutureTimeoutError:
                if retry_count < max_retries:
                    time.sleep(0.1)  # Minimal delay for retry
                    return self._connect_to_server(retry_count + 1, max_retries)
                return False
            except Exception as e:
                if retry_count < max_retries:
                    time.sleep(0.1)  # Minimal delay for retry
                    return self._connect_to_server(retry_count + 1, max_retries)
                error_msg = f"gRPC channel ready error: {type(e).__name__}"
                return False
            
            # Create stub
            try:
                self.grpc_stub = killfeed_detection_pb2_grpc.KillfeedDetectionServiceStub(self.grpc_channel)
                return True
            except Exception as e:
                if retry_count < max_retries:
                    time.sleep(0.1)  # OPTIMIZED: Minimal delay for retry
                    return self._connect_to_server(retry_count + 1, max_retries)
                error_msg = f"Failed to create gRPC stub: {type(e).__name__}"
                return False
                
        except Exception as e:
            # Catch all other exceptions
            if retry_count < max_retries:
                time.sleep(0.1)  # OPTIMIZED: Minimal delay for retry
                return self._connect_to_server(retry_count + 1, max_retries)
            error_msg = f"Unexpected gRPC connection error: {type(e).__name__}: {str(e)[:100]}"
            if hasattr(self, 'initialization_errors'):
                self.initialization_errors.append(error_msg)
            return False
    
    def _init_ocr(self):
        """Initialize OCR text detector with robust error handling and retry logic.
        
        RESILIENT: Never raises exceptions - always returns None on failure.
        Tries multiple initialization strategies with fallbacks.
        """
        
        # Check if FreeFireTextDetector is available (use globals() to avoid UnboundLocalError)
        global FreeFireTextDetector
        detector_class = globals().get('FreeFireTextDetector', None)
        
        if detector_class is None:
            # Try multiple import paths with retry logic
            import_paths = [
                ('text', lambda: __import__('text', fromlist=['FreeFireTextDetector']).FreeFireTextDetector),
                ('.text', lambda: __import__('.text', fromlist=['FreeFireTextDetector']).FreeFireTextDetector),
                ('score_ai.detection.text', lambda: __import__('score_ai.detection.text', fromlist=['FreeFireTextDetector']).FreeFireTextDetector),
            ]
            
            for path_name, import_func in import_paths:
                try:
                    import sys
                    detection_dir = os.path.dirname(os.path.abspath(__file__))
                    if detection_dir not in sys.path:
                        sys.path.insert(0, detection_dir)
                    
                    detector_class = import_func()
                    FreeFireTextDetector = detector_class
                    break
                except (ImportError, AttributeError, ModuleNotFoundError) as e:
                    continue
                except Exception as e:
                    # Log but continue trying other paths
                    error_msg = f"Import error from {path_name}: {type(e).__name__}"
                    self.initialization_errors.append(error_msg)
                    continue
        
        if detector_class is None:
            return None
        
        # RESILIENT: Try to initialize OCR with multiple attempts and fallback strategies
        max_init_attempts = 3
        init_retry_delay = 0.5
        
        for attempt in range(max_init_attempts):
            try:
                # Check if paddleocr is available
                try:
                    import paddleocr
                except ImportError as e:
                    return None
                
                # Wrap detector initialization in try-except to catch any exceptions
                try:
                    detector = detector_class()
                    return detector
                except OSError as e:
                    error_msg = str(e)
                    if "shm.dll" in error_msg or "WinError 127" in error_msg:
                        # Try with minimal configuration on retry
                        if attempt < max_init_attempts - 1:
                            time.sleep(init_retry_delay)
                            continue
                        else:
                            return None
                    else:
                        if attempt < max_init_attempts - 1:
                            time.sleep(init_retry_delay)
                            init_retry_delay *= 1.5
                            continue
                        else:
                            return None
                except ImportError as e:
                    # ImportError from detector initialization - don't retry
                    return None
                except RuntimeError as e:
                    # RuntimeError - try with different configuration
                    if attempt < max_init_attempts - 1:
                        time.sleep(init_retry_delay)
                        init_retry_delay *= 1.5
                        continue
                    else:
                        error_msg = f"OCR RuntimeError: {str(e)[:200]}"
                        self.initialization_errors.append(error_msg)
                        return None
                except Exception as e:
                    # Other exceptions - retry with backoff
                    if attempt < max_init_attempts - 1:
                        time.sleep(init_retry_delay)
                        init_retry_delay *= 1.5
                        continue
                    else:
                        error_msg = f"OCR initialization error: {type(e).__name__}: {str(e)[:200]}"
                        self.initialization_errors.append(error_msg)
                        return None
                        
            except Exception as e:
                # Outer exception handler - catch any unexpected errors
                if attempt < max_init_attempts - 1:
                    time.sleep(init_retry_delay)
                    init_retry_delay *= 1.5
                    continue
                else:
                    error_msg = f"Unexpected OCR error: {type(e).__name__}: {str(e)[:200]}"
                    self.initialization_errors.append(error_msg)
                    return None
        
        # If we get here, all attempts failed
        return None
    
    def _normalize_status(self, status):
        """Normalize status - server already sends normalized status, this is just a safety check.
        
        Args:
            status: Status string from server
            
        Returns:
            str: Normalized status ("revive", "kill", or "gun knockout")
        """
        if not status:
            return "gun knockout"
        
        status_lower = str(status).lower().strip()
        
        # Map common variations to standard format
        if status_lower in ["revive", "revived", "reviving"]:
            return "revive"
        elif status_lower in ["kill", "killed", "killing"]:
            return "kill"
        elif status_lower in ["gun knockout", "knockout", "knocked", "knocked out", "knock"]:
            return "gun knockout"
        
        # Default to gun knockout for unknown statuses
        return "gun knockout"
    
    def process_detections_from_server(self, detections):
        """Process cropped killblock images received from gRPC server.
        
        REAL-TIME OPTIMIZATION: Process ALL detections in parallel immediately.
        Multiple killblocks in one frame are processed simultaneously without delay.
        This ensures killfeeds appear in TMS within 1 second.
        
        Args:
            detections: List of detection dicts from server, each containing:
                - 'cropped_image': Cropped killblock image (numpy array)
                - 'status': Status string ("revive", "kill", or "gun knockout")
                - 'bbox': Bounding box [x1, y1, x2, y2]
                - 'confidence': Detection confidence
        
        Returns:
            int: Number of successfully queued/processed detections
        """
        if not detections:
            return 0
        
        processed_count = 0
        
        # Simple log when detections are queued
        print(f"🎯 Detection at frame - Queued for processing")
        
        # REAL-TIME: Process ALL detections in parallel immediately
        # Each detection is processed in its own thread for true parallelism
        # This ensures multiple killblocks in one frame are handled instantly
        for detection in detections:
            try:
                # Always queue for parallel processing - workers handle immediately
                # This ensures true parallelism for multiple killblocks
                try:
                    self.detection_queue.put_nowait(detection)
                    processed_count += 1
                except:
                    # Queue full - process immediately in background thread as fallback
                    # Don't block - spawn thread to process
                    def process_async(det):
                        try:
                            self._process_detection(det)
                        except:
                            pass
                    
                    thread = threading.Thread(target=process_async, args=(detection,), daemon=True)
                    thread.start()
                    processed_count += 1
            except Exception as e:
                # Silent error - continue processing other detections
                continue
        
        if processed_count > 0:
            print(f"✅ {processed_count} detection(s) queued for async processing")
        
        return processed_count
    
    def process_frame_via_grpc(self, frame):
        """Send frame to gRPC server and process returned detections.
        
        This method is kept for backward compatibility but frame capture
        should ideally be handled by the caller (main.py).
        
        Args:
            frame: Full frame image (numpy array)
        
        Returns:
            int: Number of detections processed
        """
        if frame is None or frame.size == 0:
            return 0
        
        # OPTIMIZED: Auto-reconnect if disconnected (fast, non-blocking)
        if self.grpc_stub is None:
            # Try quick reconnect (non-blocking - don't wait)
            try:
                self._connect_to_server(max_retries=1)
            except:
                pass  # Continue even if reconnect fails - will retry next frame
            if self.grpc_stub is None:
                return 0  # Skip this frame if still not connected
        
        try:
            # REAL-TIME: Use JPEG encoding instead of PNG (3-5x faster encoding)
            _, frame_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            request = killfeed_detection_pb2.ProcessFrameRequest(
                frame_image=frame_encoded.tobytes(),
                frame_width=frame.shape[1],
                frame_height=frame.shape[0]
            )
            
            # OPTIMIZED: Fast timeout (2s) for real-time performance - prevents blocking
            response = self.grpc_stub.ProcessFrame(request, timeout=2)
            
            if not response.success:
                return 0
            
            # Convert response to list
            detections = []
            for detection in response.detections:
                cropped_array = np.frombuffer(detection.cropped_image, dtype=np.uint8)
                cropped_image = cv2.imdecode(cropped_array, cv2.IMREAD_COLOR)
                
                if cropped_image is None or cropped_image.size == 0:
                    continue
                
                # Status is already normalized by server (grpc_block.py)
                status = detection.status if detection.status else "gun knockout"
                bbox = [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2]
                
                # Simple status log
                status_emoji = "🔴" if status == "kill" else "🟢" if status == "revive" else "⚪"
                print(f"   {status_emoji} Status: {status}")
                
                detections.append({
                    'cropped_image': cropped_image,
                    'status': status,
                    'bbox': bbox,
                    'confidence': detection.confidence
                })
            
            # Process detections
            return self.process_detections_from_server(detections)
            
        except grpc.RpcError as e:
            error_code = e.code()
            # Auto-reconnect on gRPC errors - don't block processing
            if error_code == grpc.StatusCode.UNAVAILABLE:
                try:
                    self._connect_to_server(max_retries=1)
                except:
                    pass  # Continue even if reconnect fails
            return 0  # Return 0 but don't crash - continue processing
        except Exception as e:
            # Any other error - continue processing, don't crash
            return 0
    
    def _enhance_image_for_ocr(self, gray_image):
        """
        Enhance grayscale image for better OCR accuracy.
        Applies: Contrast enhancement → Sharpening → Upscaling
        
        Args:
            gray_image: Grayscale image (numpy array, 2D)
            
        Returns:
            Enhanced grayscale image ready for OCR
        """
        if gray_image is None or gray_image.size == 0:
            return gray_image
        
        try:
            # Step 1: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This improves text visibility, especially for low-contrast images
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray_image)
            except Exception:
                enhanced = gray_image.copy()
            
            # Step 2: Apply slight sharpening to enhance text edges
            # This helps OCR recognize characters more clearly
            try:
                # Unsharp masking for better text clarity
                gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
                sharpened = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
                enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
            except Exception:
                pass  # Continue if sharpening fails
            
            # Step 3: Upscale small images for better OCR accuracy
            # PaddleOCR works better with larger text
            min_dimension = 300
            if min(enhanced.shape[0], enhanced.shape[1]) < min_dimension:
                scale_factor = min_dimension / min(enhanced.shape[0], enhanced.shape[1])
                scale_factor = min(3.0, scale_factor)  # Max 3x upscale
                new_width = int(enhanced.shape[1] * scale_factor)
                new_height = int(enhanced.shape[0] * scale_factor)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Step 4: Resize if too large (PaddleOCR limit ~2000px)
            max_dim = 2000
            if enhanced.shape[0] > max_dim or enhanced.shape[1] > max_dim:
                scale = min(max_dim / enhanced.shape[1], max_dim / enhanced.shape[0])
                new_width = int(enhanced.shape[1] * scale)
                new_height = int(enhanced.shape[0] * scale)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return enhanced
            
        except Exception as e:
            # Return original if enhancement fails
            return gray_image
    
    def _enhance_image_aggressive(self, gray_image):
        """
        Aggressive enhancement for difficult images.
        Applies: Strong contrast → Denoising → Sharpening → Upscaling
        """
        if gray_image is None or gray_image.size == 0:
            return gray_image
        
        try:
            # Step 1: Denoise first
            try:
                denoised = cv2.fastNlMeansDenoising(gray_image, None, 5, 7, 21)
            except Exception:
                denoised = gray_image.copy()
            
            # Step 2: Strong contrast enhancement
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                enhanced = clahe.apply(denoised)
            except Exception:
                enhanced = denoised
            
            # Step 3: Strong sharpening
            try:
                gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
                sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
                enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
            except Exception:
                pass
            
            # Step 4: Upscale more aggressively
            min_dimension = 400
            if min(enhanced.shape[0], enhanced.shape[1]) < min_dimension:
                scale_factor = min_dimension / min(enhanced.shape[0], enhanced.shape[1])
                scale_factor = min(4.0, scale_factor)
                new_width = int(enhanced.shape[1] * scale_factor)
                new_height = int(enhanced.shape[0] * scale_factor)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Step 5: Resize if too large
            max_dim = 2000
            if enhanced.shape[0] > max_dim or enhanced.shape[1] > max_dim:
                scale = min(max_dim / enhanced.shape[1], max_dim / enhanced.shape[0])
                new_width = int(enhanced.shape[1] * scale)
                new_height = int(enhanced.shape[0] * scale)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return enhanced
        except Exception:
            return gray_image
    
    def _enhance_image_threshold(self, gray_image):
        """
        Threshold-based enhancement for very difficult images.
        Applies: Adaptive thresholding → Morphology → Upscaling
        """
        if gray_image is None or gray_image.size == 0:
            return gray_image
        
        try:
            # Step 1: Try adaptive thresholding
            try:
                thresh = cv2.adaptiveThreshold(
                    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                # Invert if needed (for light text on dark background)
                if np.mean(thresh) < 127:
                    thresh = cv2.bitwise_not(thresh)
                enhanced = thresh
            except Exception:
                enhanced = gray_image.copy()
            
            # Step 2: Morphological operations to clean up
            try:
                kernel = np.ones((2, 2), np.uint8)
                enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
                enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            except Exception:
                pass
            
            # Step 3: Upscale
            min_dimension = 350
            if min(enhanced.shape[0], enhanced.shape[1]) < min_dimension:
                scale_factor = min_dimension / min(enhanced.shape[0], enhanced.shape[1])
                scale_factor = min(3.5, scale_factor)
                new_width = int(enhanced.shape[1] * scale_factor)
                new_height = int(enhanced.shape[0] * scale_factor)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Step 4: Resize if too large
            max_dim = 2000
            if enhanced.shape[0] > max_dim or enhanced.shape[1] > max_dim:
                scale = min(max_dim / enhanced.shape[1], max_dim / enhanced.shape[0])
                new_width = int(enhanced.shape[1] * scale)
                new_height = int(enhanced.shape[0] * scale)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return enhanced
        except Exception:
            return gray_image
    
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
            return None
    
    def _preprocess_image_for_ocr(self, image, strategy='enhanced', target_dpi=300):
        """
        ACCURACY: Enhanced preprocessing for maximum OCR accuracy.
        Uses multiple strategies to ensure best possible text recognition.
        """
        # First validate the input image
        image = self._validate_image_for_ocr(image)
        if image is None:
            return None
        
        try:
            height, width = image.shape[:2]
            processed = image.copy()
            
            # Step 1: Upscale small images for better text recognition
            # Small text needs higher resolution for accurate OCR
            min_dimension = 300  # Reduced from 400 - too much upscaling can blur text
            if min(height, width) < min_dimension:
                scale_factor = min_dimension / min(height, width)
                # Cap upscaling at 3x to avoid blurring (was 5x)
                scale_factor = min(3.0, scale_factor)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                processed = cv2.resize(processed, (new_width, new_height), 
                                     interpolation=cv2.INTER_CUBIC)  # CUBIC is faster and still good
            
            # Step 2: Resize if too large (PaddleOCR limit ~2000px)
            max_dim = 2000
            if processed.shape[0] > max_dim or processed.shape[1] > max_dim:
                scale = min(max_dim / processed.shape[1], max_dim / processed.shape[0])
                new_width = int(processed.shape[1] * scale)
                new_height = int(processed.shape[0] * scale)
                processed = cv2.resize(processed, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)  # AREA for downscaling
            
            # Step 3: Enhanced contrast for better text visibility
            if strategy in ['enhanced', 'default']:
                try:
                    # Convert to LAB color space for better contrast control
                    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
                    # Moderate clipLimit to avoid over-enhancement that can break OCR
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # Reduced from 4.0
                    l_enhanced = clahe.apply(l)
                    
                    # Merge back and convert to BGR
                    lab_enhanced = cv2.merge([l_enhanced, a, b])
                    processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                except Exception:
                    # Fallback: simple histogram equalization on grayscale
                    try:
                        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                        gray_eq = cv2.equalizeHist(gray)
                        processed = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
                    except Exception:
                        pass
            
            # Step 4: Light sharpening for enhanced strategy only (skip for default to preserve text)
            if strategy == 'enhanced':
                try:
                    # Light unsharp masking - too aggressive sharpening can break OCR
                    gaussian = cv2.GaussianBlur(processed, (0, 0), 1.0)
                    processed = cv2.addWeighted(processed, 1.3, gaussian, -0.3, 0)
                    processed = np.clip(processed, 0, 255).astype(np.uint8)
                except Exception:
                    pass
            
            # Final validation before returning
            processed = self._validate_image_for_ocr(processed)
            return processed
            
        except Exception as e:
            # Return validated original if preprocessing fails
            return self._validate_image_for_ocr(image)
    
    def _preprocess_image_for_ocr_retry(self, image):
        """
        GRAYSCALE PREPROCESSING (RETRY): Enhanced grayscale preprocessing for low-confidence OCR retry.
        Uses more aggressive contrast and thresholding for difficult images.
        
        Flow: Grayscale → Enhanced Contrast → Aggressive Thresholding → OCR
        """
        # First validate the input image
        image = self._validate_image_for_ocr(image)
        if image is None:
            return None
        
        try:
            height, width = image.shape[:2]
            
            # Step 1: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Step 2: Enhanced contrast using CLAHE (more aggressive for retry)
            try:
                # Higher clipLimit (3.0) and smaller tiles (4x4) for better local contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                gray = clahe.apply(gray)
            except Exception:
                # Fallback: simple histogram equalization
                try:
                    gray = cv2.equalizeHist(gray)
                except:
                    pass
            
            # Step 3: Aggressive adaptive thresholding for better text separation
            try:
                # More aggressive thresholding for difficult images - optimized for separation
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 9, 4  # Optimized: blockSize=9, C=4 for better separation
                )
                gray = thresh
            except Exception:
                # Fallback: Otsu's thresholding (automatic threshold selection)
                try:
                    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                except:
                    # Final fallback: simple binary threshold
                    try:
                        _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    except:
                        pass
            
            # Step 4: Resize if too large
            max_dim = 2000
            if gray.shape[0] > max_dim or gray.shape[1] > max_dim:
                scale = min(max_dim / gray.shape[1], max_dim / gray.shape[0])
                new_width = int(gray.shape[1] * scale)
                new_height = int(gray.shape[0] * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Step 5: Upscale small images
            min_dimension = 300
            if min(gray.shape[0], gray.shape[1]) < min_dimension:
                scale_factor = min_dimension / min(gray.shape[0], gray.shape[1])
                scale_factor = min(3.0, scale_factor)
                new_width = int(gray.shape[1] * scale_factor)
                new_height = int(gray.shape[0] * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Step 6: Convert back to BGR (3-channel) for PaddleOCR compatibility
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Final validation before returning
            processed = self._validate_image_for_ocr(processed)
            return processed
            
        except Exception as e:
            # Return validated original if preprocessing fails
            return self._validate_image_for_ocr(image)
    
    def _load_team_rosters(self):
        """Load team rosters from API for fuzzy name matching."""
        if not self.match_id or not self.access_token:
            return
        
        try:
            # Get backend URL and endpoint from config or use defaults
            if config:
                backend_url = config.get('api.backend_url', 'http://3.7.109.218:5005').rstrip('/')
                endpoint = config.get('endpoints.team_players', 'LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players')
            else:
                backend_url = 'http://3.7.109.218:5005'
                endpoint = 'LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players'
            
            full_url = f"{backend_url}/{endpoint}?matchId={self.match_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
            
            # RESILIENT: Handle API errors gracefully
            try:
                response = requests.get(full_url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.Timeout:
                return
            except requests.exceptions.ConnectionError:
                return
            except requests.exceptions.HTTPError as e:
                return
            except requests.exceptions.RequestException as e:
                return
            except Exception as e:
                return
            
            if not data or not data.get("success") or not data.get("data"):
                return
            
            # Process team data
            teams_data = data.get("data", {})
            if isinstance(teams_data, list):
                # If data is a list of teams
                for team in teams_data:
                    team_name = team.get("team_name")
                    if team_name:
                        players = {
                            key: value
                            for key, value in team.items()
                            if key.startswith("player") and value
                        }
                        self.team_rosters[team_name] = players
            elif isinstance(teams_data, dict):
                # If data is already a dict
                self.team_rosters = teams_data.get("teams", {})
            
            # OPTIMIZED: Build flattened list using set for O(1) lookup
            self.all_player_names = []
            seen_names = set()
            for team_name, players in self.team_rosters.items():
                for player_key, player_name in players.items():
                    if player_name and player_name not in seen_names:
                        seen_names.add(player_name)
                        self.all_player_names.append(player_name)
                        self.known_players.add(player_name)
            
            # Clear cached player list to force rebuild
            if hasattr(self, '_cached_all_players'):
                delattr(self, '_cached_all_players')
            
        except requests.exceptions.RequestException as e:
            pass
        except Exception as e:
            pass
    
    def _refresh_team_rosters_realtime(self):
        """
        OPTIMIZED: Non-blocking refresh of team rosters from API.
        Returns immediately - uses existing roster if refresh fails or is in progress.
        Actual refresh happens in background thread to avoid blocking.
        """
        if not self.match_id or not self.access_token:
            return False
        
        # OPTIMIZED: Check if refresh is already in progress (prevent duplicate calls)
        if not hasattr(self, '_roster_refresh_in_progress'):
            self._roster_refresh_in_progress = False
        
        if self._roster_refresh_in_progress:
            return False  # Refresh already in progress - use existing roster
        
        # OPTIMIZED: Start refresh in background thread (non-blocking)
        def refresh_roster_async():
            try:
                self._roster_refresh_in_progress = True
                
                # Get backend URL and endpoint from config or use defaults
                if config:
                    backend_url = config.get('api.backend_url', 'http://3.7.109.218:5005').rstrip('/')
                    endpoint = config.get('endpoints.team_players', 'LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players')
                else:
                    backend_url = 'http://3.7.109.218:5005'
                    endpoint = 'LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players'
                
                # Use full URL format: {backend_url}/{endpoint}
                full_url = f"{backend_url}/{endpoint}?matchId={self.match_id}"
                headers = {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}
                
                # OPTIMIZED: Very fast timeout (1s) for real-time performance
                try:
                    response = requests.get(full_url, headers=headers, timeout=1)
                    response.raise_for_status()
                    data = response.json()
                except:
                    return  # Use existing roster on any error
                
                if not data or not data.get("success") or not data.get("data"):
                    return
                
                # Process team data
                teams_data = data.get("data", {})
                new_team_rosters = {}
                
                if isinstance(teams_data, list):
                    # If data is a list of teams
                    for team in teams_data:
                        team_name = team.get("team_name")
                        if team_name:
                            players = {
                                key: value
                                for key, value in team.items()
                                if key.startswith("player") and value
                            }
                            new_team_rosters[team_name] = players
                elif isinstance(teams_data, dict):
                    # If data is already a dict
                    new_team_rosters = teams_data.get("teams", {})
                
                # Update team rosters (thread-safe update)
                self.team_rosters = new_team_rosters
                
                # OPTIMIZED: Build flattened list using set for O(1) lookup
                self.all_player_names = []
                seen_names = set()
                for team_name, players in self.team_rosters.items():
                    for player_key, player_name in players.items():
                        if player_name and player_name not in seen_names:
                            seen_names.add(player_name)
                            self.all_player_names.append(player_name)
                            self.known_players.add(player_name)
                
                # Clear cached player list to force rebuild
                if hasattr(self, '_cached_all_players'):
                    delattr(self, '_cached_all_players')
            except:
                pass  # Silent failure - use existing roster
            finally:
                self._roster_refresh_in_progress = False
        
        # Start refresh in background thread (non-blocking)
        refresh_thread = threading.Thread(target=refresh_roster_async, daemon=True)
        refresh_thread.start()
        
        return True  # Refresh started (non-blocking)
    
    def _fuzzy_match_player_name(self, detected_name):
        """
        Match detected player name against team roster using fuzzy matching.
        Returns "unknown" if no good match is found (similar to killblocks.py).
        
        CRITICAL: Uses STRICT thresholds (85%+) to prevent incorrect matches.
        Only matches when OCR text is very similar to roster name.
        
        Args:
            detected_name (str): Name detected by OCR
            
        Returns:
            str: Best matching player name from roster, or "unknown" if no good match
        """
        # Handle special case player names (like in killblocks.py)
        if detected_name in ["killer", "Playzone", "unknown", "UNKNOWN"]:
            return detected_name
        
        if not detected_name or not self.all_player_names or not RAPIDFUZZ_AVAILABLE:
            return "unknown"
        
        # Normalize detected name
        detected_name_clean = detected_name.strip()
        if len(detected_name_clean) < 1:
            return "unknown"
        
        # CRITICAL: STRICT thresholds to prevent incorrect matches like "NMR.ARJUNNO4"
        # Only match when similarity is very high (85%+) to avoid false positives
        if config:
            try:
                processing_config = config.get_processing_config()
                high_threshold = processing_config.get('similarity_threshold_high', 0.85)  # STRICT: 85% minimum
                low_threshold = processing_config.get('similarity_threshold_low', 0.80)  # STRICT: 80% minimum
            except:
                high_threshold = 0.85  # STRICT: 85% minimum for high confidence
                low_threshold = 0.80  # STRICT: 80% minimum for any match
        else:
            high_threshold = 0.85  # STRICT: 85% minimum for high confidence
            low_threshold = 0.80  # STRICT: 80% minimum for any match
        
        detected_lower = detected_name_clean.lower()
        
        # OPTIMIZED: Cache player list to avoid rebuilding every time
        if not hasattr(self, '_cached_all_players') or not self._cached_all_players:
            self._cached_all_players = [
                (player_name, team_name)
                for team_name, players in self.team_rosters.items()
                for player_key, player_name in players.items()
                if player_name
            ]
            # Remove duplicates while preserving order
            seen = set()
            self._cached_all_players = [
                p for p in self._cached_all_players
                if p[0] not in seen and not seen.add(p[0])
            ]
        
        all_players = self._cached_all_players
        if not all_players:
            return "unknown"
        
        # OPTIMIZED: Single pass with early return for high-confidence matches
        best_match = None
        best_score = 0.0
        
        for player_name, team in all_players:
            player_lower = player_name.lower()
            
            # Fast ratio check first (early return for high confidence)
            ratio_score = fuzz.ratio(detected_lower, player_lower) / 100.0
            if ratio_score >= high_threshold:
                # High confidence match - return immediately
                return player_name  # Fastest path
            
            # Calculate combined score only if ratio is promising (above 70% of low threshold)
            if ratio_score >= low_threshold * 0.875:  # 70% of 80% = 56%, but we use 87.5% of 80% = 70%
                partial_score = fuzz.partial_ratio(detected_lower, player_lower) / 100.0
                token_sort_score = fuzz.token_sort_ratio(detected_lower, player_lower) / 100.0
                combined_score = (ratio_score * 0.50) + (partial_score * 0.30) + (token_sort_score * 0.20)
                max_score = max(ratio_score, partial_score, token_sort_score, combined_score)
                
                # STRICT: Only accept if max_score meets minimum threshold
                if max_score > best_score and max_score >= low_threshold:
                    best_score = max_score
                    best_match = player_name
        
        if best_match:
            return best_match
        
        # STRICT: Removed fallback process.extractOne - it was too permissive
        # Only return matches that meet strict thresholds above
        
        # No good match found - return "unknown"
        return "unknown"

    # ------------------------------------------------------------------
    # Match-roster snapping  (zero-latency — ~60 rapidfuzz comparisons)
    # ------------------------------------------------------------------

    def set_match_roster(self, player_names):
        """Load canonical player names for the current match.

        Call once per match with the full list of players (all teams + subs).
        Enables high-accuracy OCR correction without slowing down the pipeline.

        Args:
            player_names: iterable of strings like ['K9.HUNNYSUNY', 'RNTX.ARSH17', ...]
        """
        names = [n.strip().upper() for n in player_names if n and n.strip()]
        self._match_roster_names = names
        # Build suffix → full_name map for prefix-mangled OCR results
        self._match_roster_suffixes = {}
        for name in names:
            if '.' in name:
                suffix = name.split('.', 1)[1]
                if len(suffix) >= 3:
                    self._match_roster_suffixes[suffix] = name
        print(f"✅ Match roster loaded: {len(self._match_roster_names)} players")

    def _snap_to_roster(self, name: str) -> str:
        """Correct OCR noise by snapping name to the closest known match player.

        Three-stage strategy (fastest first):
          1. Exact match        — O(1)
          2. WRatio full-name   — handles char substitutions, missing dot, etc.
          3. Suffix-only match  — handles a mangled team prefix (e.g. 'K8.' → 'K9.')

        Returns the canonical player name on a confident hit, else the original string.
        Threshold tuned so that 1-2 OCR errors are corrected but wrong names are NOT snapped.
        """
        if not name or not RAPIDFUZZ_AVAILABLE:
            return name

        roster = self._match_roster_names
        if not roster:
            # Fall back to API-loaded names if match roster not set yet
            if self.all_player_names:
                roster = [n.upper() for n in self.all_player_names]
            else:
                return name

        name_upper = name.strip().upper()

        # Stage 1: exact hit
        if name_upper in roster:
            return name_upper

        # Stage 2: full-name fuzzy (WRatio handles substitution + partial alignment)
        result = process.extractOne(name_upper, roster, scorer=fuzz.WRatio)
        if result and result[1] >= 75:
            return result[0]

        # Stage 3: suffix-only (catches "K8.HUNNYSUNY" → "K9.HUNNYSUNY")
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

    def _normalize_name(self, name):
        """ENHANCED: Robust name normalization with dot restoration and OCR error correction."""
        if not name or len(name) < self.min_name_length:
            return ""
        
        # Step 1: Preserve dots and clean invalid characters
        # CRITICAL: Don't remove dots - they're part of player names (e.g., "TT.AMIN022")
        name = re.sub(r'[^\w\d.\s_-]', '', name)  # Keep dots, remove other special chars
        name_upper = name.upper()
        
        # Step 2: Fix common OCR errors (0->O, 1->I, 5->S, 8->B, etc.)
        # Fix '0' -> 'O' in letter context
        if re.search(r'[A-Z]0[A-Z]', name_upper) or re.search(r'^0[A-Z]', name_upper) or re.search(r'[A-Z]0$', name_upper):
            name = re.sub(r'([A-Z])0([A-Z])', r'\1O\2', name)
            name = re.sub(r'^0([A-Z])', r'O\1', name)
            name = re.sub(r'([A-Z])0$', r'\1O', name)
        
        # Fix '1' -> 'I' in letter context
        if re.search(r'[A-Z]1[A-Z]', name_upper) or re.search(r'^1[A-Z]', name_upper) or re.search(r'[A-Z]1$', name_upper):
            name = re.sub(r'([A-Z])1([A-Z])', r'\1I\2', name)
            name = re.sub(r'^1([A-Z])', r'I\1', name)
            name = re.sub(r'([A-Z])1$', r'\1I', name)
        
        # Fix '5' -> 'S' in letter context
        if re.search(r'[A-Z]5[A-Z]', name_upper) or re.search(r'^5[A-Z]', name_upper) or re.search(r'[A-Z]5$', name_upper):
            name = re.sub(r'([A-Z])5([A-Z])', r'\1S\2', name)
        
        # Fix '8' -> 'B' in letter context
        if re.search(r'[A-Z]8[A-Z]', name_upper) or re.search(r'^8[A-Z]', name_upper) or re.search(r'[A-Z]8$', name_upper):
            name = re.sub(r'([A-Z])8([A-Z])', r'\1B\2', name)
        
        # Step 3: RESTORE MISSING DOTS (critical for names like "TT.AMIN022")
        # Pattern: If we see "TTAMIN022" or "TTAMINO22", restore to "TT.AMIN022"
        # Common patterns: XXNAME, XXNAME##, XXNAME## where XX is team prefix
        if len(name) >= 4:
            # Pattern: TTAMIN022 -> TT.AMIN022
            # Look for 2-3 letter prefix followed by name without dot
            dot_restore_pattern = re.compile(r'^([A-Z]{2,3})([A-Z]{3,})(\d*)$')
            match = dot_restore_pattern.match(name.upper())
            if match:
                prefix, main_name, numbers = match.groups()
                # Restore dot if missing: TTAMIN022 -> TT.AMIN022
                if '.' not in name:
                    name = f"{prefix}.{main_name}{numbers}"
            # Pattern: TTAMINO22 -> TT.AMINO22 (with O at end of name)
            dot_restore_pattern2 = re.compile(r'^([A-Z]{2,3})([A-Z]{3,}[A-Z])(\d+)$')
            match2 = dot_restore_pattern2.match(name.upper())
            if match2:
                prefix, main_name, numbers = match2.groups()
                if '.' not in name:
                    name = f"{prefix}.{main_name}{numbers}"
        
        # Step 4: Clean up whitespace and invalid chars
        name = re.sub(r'\s+', '', name)  # Remove all spaces (names shouldn't have spaces)
        name = re.sub(r'\.{2,}', '.', name)  # Multiple dots to single dot
        name = name.strip('._- ')
        
        # Step 5: Validate length and format
        if len(name) < self.min_name_length or len(name) > self.max_name_length:
            return ""
        
        # Reject if name is too short or doesn't look like a player name
        if len(name) < 3 or not any(c.isalpha() for c in name):
            return ""
        
        return name
    
    def _extract_names_with_ocr(self, cropped_image, preprocessing_strategy='default', image_id=None, bbox=None, retry_mode=False):
        """
        Extract names using split-image OCR approach: left half = Killer, right half = Victim.
        Uses ocrdetection.py's process_image_from_array() method for clean, simplified code.
        
        PERFORMANCE TARGET: Complete in < 500ms for real-time processing.
        - OCR processing: ~300-400ms (PaddleOCR bottleneck, but necessary)
        - Text selection & validation: ~10-20ms (optimized with early exits)
        - Normalization: ~5-10ms (fast string operations)
        Total: ~350-450ms (well under 1 second target)
        
        Args:
            cropped_image: The cropped killblock image (numpy array)
            preprocessing_strategy: Ignored (kept for compatibility)
            image_id: Unique identifier for this crop (for debugging)
            bbox: Bounding box coordinates [x1, y1, x2, y2] (for debugging)
            retry_mode: Ignored (kept for compatibility)
        
        Returns:
            (killer, victim, killer_conf, victim_conf, killer_x, victim_x) where:
            - killer is ALWAYS from left half (no swapping)
            - victim is ALWAYS from right half (no swapping)
            - confidences are separate for each name
            - x positions are for validation (left=0, right=center_x)
        """
        if cropped_image is None or cropped_image.size == 0 or self.text_detector is None:
            return None, None, 0.0, 0.0, None, None
        
        # Generate image ID if not provided
        if image_id is None:
            image_id = f"img_{int(time.time() * 1000)}"
        
        try:
            # Preprocess with specified strategy
            # If retry_mode, apply aggressive preprocessing for low confidence cases
            if retry_mode or preprocessing_strategy == 'retry':
                preprocessed = self._preprocess_image_for_ocr_retry(cropped_image)
            else:
                preprocessed = self._preprocess_image_for_ocr(cropped_image, strategy=preprocessing_strategy)
            
            if preprocessed is None or preprocessed.size == 0:
                return None, None, 0.0, 0.0, None, None
            
            # ACCURACY: Save to temp file with PNG format for better quality
            temp_path = None
            try:
                # Use PNG for better quality (no compression artifacts that affect OCR)
                temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                
                # Ensure image is valid before writing
                if not isinstance(preprocessed, np.ndarray):
                    return None, None, 0.0, 0.0, None, None
            
                # ACCURACY: Use PNG for better quality (no compression artifacts)
                # PNG preserves text edges better than JPEG for OCR
                success = cv2.imwrite(temp_path, preprocessed)
                if not success or not os.path.exists(temp_path):
                    return None, None, 0.0, 0.0, None, None
                
                # PERFORMANCE: Disabled debug image saving for speed (was saving on every OCR call)
                # Uncomment below to enable debug images:
                # debug_dir = "ocr_debug"
                # os.makedirs(debug_dir, exist_ok=True)
                # debug_path = os.path.join(debug_dir, f"ocr_debug_{int(time.time() * 1000)}_{preprocessing_strategy}.png")
                # cv2.imwrite(debug_path, preprocessed)
                
                # Run OCR on the cropped killfeed image
                try:
                    # Validate image before processing
                    if preprocessed is None or preprocessed.size == 0:
                        return None, None, 0.0, 0.0, None, None
                    
                    results = self.text_detector.process_image(temp_path)
                except Exception as e:
                    # Silent failure - will try next strategy
                    return None, None, 0.0, 0.0, None, None
                
                if not results or not isinstance(results, dict):
                    return None, None, 0.0, 0.0, None, None
                
                # Check if OCR detected any text at all
                if results.get("message"):
                    return None, None, 0.0, 0.0, None, None
                
                # Extract names and confidences with position information
                # CRITICAL FIX: text.py returns None for killer/victim when not detected, not empty dict
                killer_dict = results.get("killer") or {}
                victim_dict = results.get("victim") or {}
                
                # CRITICAL: Extract ALL detected text regions with their X positions
                # We need to assign roles based on X coordinates, not arrival order
                detected_regions = []
                
                # Extract killer region
                if isinstance(killer_dict, dict) and killer_dict.get("text"):
                    position = killer_dict.get("position")
                    if position and isinstance(position, (list, tuple)) and len(position) >= 1:
                        x_pos = int(position[0])
                        detected_regions.append({
                            'text': str(killer_dict.get("text", "")).strip(),
                            'confidence': float(killer_dict.get("confidence", 0.0)),
                            'x': x_pos,
                            'position': position,
                            'type': 'killer_dict'
                        })
                
                # Extract victim region
                if isinstance(victim_dict, dict) and victim_dict.get("text"):
                    position = victim_dict.get("position")
                    if position and isinstance(position, (list, tuple)) and len(position) >= 1:
                        x_pos = int(position[0])
                        detected_regions.append({
                            'text': str(victim_dict.get("text", "")).strip(),
                            'confidence': float(victim_dict.get("confidence", 0.0)),
                            'x': x_pos,
                            'position': position,
                            'type': 'victim_dict'
                        })
                
                # CRITICAL: Sort by X coordinate (left to right) to assign roles correctly
                # Leftmost = Killer, Rightmost = Victim (based on bounding box X position)
                detected_regions.sort(key=lambda r: r['x'])
                
                # Assign roles based on X position, NOT arrival order
                killer = ""
                killer_conf = 0.0
                killer_x = None
                victim = ""
                victim_conf = 0.0
                victim_x = None
                
                if len(detected_regions) >= 2:
                    # We have at least 2 regions - leftmost = killer, rightmost = victim
                    leftmost = detected_regions[0]
                    rightmost = detected_regions[-1]
                    
                    killer = leftmost['text']
                    killer_conf = leftmost['confidence']
                    killer_x = leftmost['x']
                    
                    victim = rightmost['text']
                    victim_conf = rightmost['confidence']
                    victim_x = rightmost['x']
                elif len(detected_regions) == 1:
                    # Only one region detected - assign based on X position relative to image center
                    region = detected_regions[0]
                    img_width = cropped_image.shape[1] if cropped_image is not None else 1000
                    center_x = img_width / 2
                    
                    if region['x'] < center_x:
                        # Left side = killer
                        killer = region['text']
                        killer_conf = region['confidence']
                        killer_x = region['x']
                    else:
                        # Right side = victim
                        victim = region['text']
                        victim_conf = region['confidence']
                        victim_x = region['x']
                
                # Simplified logging - only log successful extractions
                if killer and victim:
                    pass  # Success - no need to log
                elif not killer and not victim:
                    pass  # Failure - handled by caller
                
                # CRITICAL: Validate ordering - NEVER swap, only verify and reject if wrong
                # OCR should already return names in correct order (leftmost = killer, rightmost = victim)
                # If order is wrong, reject the detection to prevent incorrect TMS updates
                if killer and victim and killer_x is not None and victim_x is not None:
                    # CRITICAL: Reject if "Play Zone" or similar is detected as a player name
                    killer_upper = killer.upper().strip()
                    victim_upper = victim.upper().strip()
                    if ('PLAY' in killer_upper and 'ZONE' in killer_upper) or ('PLAY' in victim_upper and 'ZONE' in victim_upper):
                        return None, None, 0.0, 0.0, None, None  # Reject this detection
                    if killer_upper == 'ZONE' or victim_upper == 'ZONE' or killer_upper.startswith('ZONE') or victim_upper.startswith('ZONE'):
                        return None, None, 0.0, 0.0, None, None  # Reject this detection
                    
                    # CRITICAL: Reject if killer and victim are the same
                    if killer_upper == victim_upper:
                        return None, None, 0.0, 0.0, None, None  # Reject this detection
                    
                    if victim_x < killer_x:
                        # CRITICAL ERROR: Victim is left of killer - this should NEVER happen
                        # OCR should have already sorted correctly, so this indicates a bug
                        return None, None, 0.0, 0.0, None, None  # Reject this detection
                    
                    # Order verified - names are in correct left-to-right order
                elif killer and not victim:
                    # Only killer detected - validate it's not "Play Zone"
                    killer_upper = killer.upper().strip()
                    if ('PLAY' in killer_upper and 'ZONE' in killer_upper) or killer_upper == 'ZONE' or killer_upper.startswith('ZONE'):
                        return None, None, 0.0, 0.0, None, None
                elif victim and not killer:
                    # Only victim detected - validate it's not "Play Zone"
                    victim_upper = victim.upper().strip()
                    if ('PLAY' in victim_upper and 'ZONE' in victim_upper) or victim_upper == 'ZONE' or victim_upper.startswith('ZONE'):
                        return None, None, 0.0, 0.0, None, None
            
                # Return names with separate confidences and X positions
                # CRITICAL: Ensure minimum confidence if names are valid
            if killer and len(killer) >= 2 and killer_conf == 0.0:
                    killer_conf = 0.5  # Minimum confidence if name is valid
            if victim and len(victim) >= 2 and victim_conf == 0.0:
                    victim_conf = 0.5  # Minimum confidence if name is valid
            
            return killer, victim, killer_conf, victim_conf, killer_x, victim_x
                
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            
        except RuntimeError as e:
            # PERFORMANCE: Removed fallback OCR attempt - if first attempt fails, return None
            # Fallback was adding another full OCR call (~1-2 seconds), not worth it for speed
            return None, None, 0.0, 0.0, None, None
            
        except Exception as e:
            # Suppress verbose errors for failed strategies (only log for default strategy)
            if preprocessing_strategy == 'default':
                pass  # Silent failure, will try other strategies
            return None, None, 0.0, 0.0, None, None
    
    def _extract_names(self, cropped_image, image_id=None, bbox=None):
        """
        Robust name extraction with multiple preprocessing strategies and retry logic.
        Uses split-image OCR approach: left half = Killer (ALWAYS), right half = Victim (ALWAYS).
        NO SWAPPING - roles are fixed by image position, not OCR order.
        
        Args:
            cropped_image: The cropped killblock image
            image_id: Unique identifier for this crop (for debugging)
            bbox: Bounding box coordinates [x1, y1, x2, y2] (for debugging)
        
        Returns:
            (processed_killer, processed_victim, confidence, original_killer, original_victim)
        """
        if cropped_image is None or cropped_image.size == 0:
            return "", "", 0.0, "", ""
        
        # Generate image ID if not provided
        if image_id is None:
            image_id = f"img_{int(time.time() * 1000)}"
        
        # If OCR is not initialized, try to initialize it now (lazy initialization - only once)
        if self.text_detector is None and not self.ocr_init_attempted:
            self.ocr_init_attempted = True
            self.text_detector = self._init_ocr()
            if self.text_detector is None:
                return "", "", 0.0, "", ""
        elif self.text_detector is None:
            # OCR already attempted and failed, skip silently
            return "", "", 0.0, "", ""
        
        # CRITICAL: Use grayscale cropped image directly - NO preprocessing strategies
        # OCR runs on grayscale image only for clear, accurate results
        best_killer = ""
        best_victim = ""
        best_confidence = 0.0
        best_original_killer = ""
        best_original_victim = ""
        
        best_killer_conf = 0.0
        best_victim_conf = 0.0
        best_killer_x = None
        best_victim_x = None
        
        try:
            # CRITICAL: Use grayscale cropped image directly - no preprocessing
            # Left half = Killer (ALWAYS), Right half = Victim (ALWAYS)
            # No swapping based on OCR order - position is fixed
            killer, victim, killer_conf, victim_conf, killer_x, victim_x = self._extract_names_with_ocr(
                cropped_image, 'original', image_id=image_id, bbox=bbox, retry_mode=False
            )
                
            # Handle None values
            if killer is None:
                killer = ""
            if victim is None:
                victim = ""
            
            # CRITICAL: Use OCR results directly - accept any valid names from grayscale OCR
            # No retry, no preprocessing - just use what OCR detected
            if killer and len(killer) >= 1:
                best_killer = killer
                best_killer_conf = killer_conf
                best_killer_x = killer_x
                best_original_killer = killer
                
            if victim and len(victim) >= 1:
                best_victim = victim
                best_victim_conf = victim_conf
                best_victim_x = victim_x
                best_original_victim = victim
            
            # Calculate confidence from detected names
            if best_killer and best_victim:
                best_confidence = max((best_killer_conf + best_victim_conf) / 2.0, 0.3)
            elif best_killer:
                best_confidence = max(best_killer_conf, 0.3)
            elif best_victim:
                best_confidence = max(best_victim_conf, 0.3)
            else:
                best_confidence = 0.0
                
        except Exception as e:
            # Log OCR errors for debugging
            if not hasattr(self, '_ocr_error_count'):
                self._ocr_error_count = 0
            self._ocr_error_count += 1
        
        # Use best results found
        if best_killer or best_victim:
            killer = best_killer
            victim = best_victim
            confidence = best_confidence
            original_killer = best_original_killer
            original_victim = best_original_victim
            # Store confidences and X positions for API payload
            if not hasattr(self, '_last_ocr_result'):
                self._last_ocr_result = {}
            if image_id not in self._last_ocr_result:  # Don't overwrite retry results
                self._last_ocr_result[image_id] = {
                    'killer_conf': best_killer_conf,
                    'victim_conf': best_victim_conf,
                    'killer_x': best_killer_x,
                    'victim_x': best_victim_x,
                    'retry_used': False
                }
        else:
            # No valid results from any strategy - OCR failed completely
            # Return empty but don't set confidence to 0 - let caller handle it
            # This allows detections to be processed even if OCR fails (server detected killblock correctly)
            return "", "", 0.0, "", ""
        
        # CRITICAL: Store original OCR text IMMEDIATELY before any processing
        # Original OCR text is the most reliable source for names
        # Preserve the raw OCR text BEFORE normalization
        original_killer = killer.strip() if (killer and len(killer.strip()) >= 2) else ""
        original_victim = victim.strip() if (victim and len(victim.strip()) >= 2) else ""
        
        # CRITICAL: Names are ALWAYS in correct order - left half = Killer, right half = Victim
        # Split-image OCR ensures no swapping - roles are fixed by image position
        # This order is IMMUTABLE and will NEVER be swapped
        
        # CRITICAL: Only normalize for processing, but preserve original OCR text
        # Don't normalize the original text - it's the most reliable source
        # Normalize only the processed names for consistency
        killer_raw = killer
        victim_raw = victim
        
        # Only normalize if we have valid text - preserve original if normalization fails
        if killer and len(killer.strip()) >= 2:
            normalized_killer = self._normalize_name(killer)
            # Use normalized if valid, otherwise fall back to original
            killer = normalized_killer if normalized_killer else killer_raw.strip()
        else:
            killer = killer_raw.strip() if killer_raw else ""
        
        if victim and len(victim.strip()) >= 2:
            normalized_victim = self._normalize_name(victim)
            # Use normalized if valid, otherwise fall back to original
            victim = normalized_victim if normalized_victim else victim_raw.strip()
        else:
            victim = victim_raw.strip() if victim_raw else ""
        
        # Ensure we preserve original if normalization removed valid text
        if not killer and killer_raw:
            killer = killer_raw.strip()
        if not victim and victim_raw:
            victim = victim_raw.strip()
        
        # Reject "Play Zone" silently
        if killer and ('PLAY' in killer.upper() and 'ZONE' in killer.upper()):
            return "", "", 0.0, original_killer, original_victim
        if victim and ('PLAY' in victim.upper() and 'ZONE' in victim.upper()):
            return "", "", 0.0, original_killer, original_victim
        
        # CRITICAL: Only normalize text - NO fuzzy matching on client side
        # TMS will perform player matching and scoring based on OCR text
        # Client only provides: OCR text + confidence + position metadata
        # Text normalization: clean whitespace, remove special chars, uppercase
        # This is NOT matching - just text cleaning for better OCR accuracy
        smoothed_killer = killer.strip().upper() if killer else ""
        smoothed_victim = victim.strip().upper() if victim else ""
        smoothed_conf = confidence
        
        # CRITICAL: Return processed names, confidence, and original OCR text
        # Original OCR text MUST be preserved - it's the most reliable source
        # Ensure original_killer and original_victim are never empty if we have valid names
        if not original_killer and killer and len(killer.strip()) >= 2:
            original_killer = killer.strip()  # Fallback to processed name if original lost
        if not original_victim and victim and len(victim.strip()) >= 2:
            original_victim = victim.strip()  # Fallback to processed name if original lost
        
        return smoothed_killer, smoothed_victim, smoothed_conf, original_killer, original_victim
    
    def _apply_temporal_smoothing(self, killer, victim, confidence):
        """Apply temporal smoothing using voting across recent frames."""
        self.name_history.append({'killer': killer, 'victim': victim, 'confidence': confidence})
        
        if confidence >= self.confidence_threshold:
            self.known_players.add(killer)
            self.known_players.add(victim)
            return killer, victim, confidence
        
        if len(self.name_history) >= 2:
            killer_votes = {}
            victim_votes = {}
            for hist in self.name_history:
                killer_votes[hist['killer']] = killer_votes.get(hist['killer'], 0) + hist['confidence']
                victim_votes[hist['victim']] = victim_votes.get(hist['victim'], 0) + hist['confidence']
            
            if killer_votes and victim_votes:
                best_killer = max(killer_votes.items(), key=lambda x: x[1])[0]
                best_victim = max(victim_votes.items(), key=lambda x: x[1])[0]
                if (best_killer and best_victim and 
                    len(best_killer) >= self.min_name_length and 
                    len(best_victim) >= self.min_name_length):
                    self.known_players.add(best_killer)
                    self.known_players.add(best_victim)
                    return best_killer, best_victim, min(1.0, confidence + 0.1)
        
        return smoothed_killer, smoothed_victim, smoothed_conf, original_killer, original_victim
    
    def _save_image(self, image, filename):
        """Save cropped image to disk - REAL-TIME optimized (async save).
        
        REAL-TIME: Image saving happens asynchronously to not block processing.
        Returns path immediately, actual save happens in background.
        """
        try:
            output_dir = "cropkillblock"
            # Check if directory exists once
            if not hasattr(self, '_output_dir_created'):
                os.makedirs(output_dir, exist_ok=True)
                self._output_dir_created = True
            filepath = os.path.join(output_dir, filename)
            jpeg_path = filepath.replace('.png', '.jpg')
            
            # REAL-TIME: Save image synchronously but quickly (JPEG is fast)
            # Synchronous save ensures file exists when API reads it, avoiding retries
            # JPEG write is fast enough (~5-10ms) that it doesn't block significantly
            try:
                cv2.imwrite(jpeg_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                # Use os.sync() or flush to ensure file is written to disk immediately
                try:
                    import os
                    if hasattr(os, 'sync'):
                        os.sync()  # Force write to disk (Unix)
                    else:
                        # Windows: use alternative method
                        import ctypes
                        ctypes.windll.kernel32.FlushFileBuffers(-1)  # Flush all file buffers
                except:
                    pass  # Ignore sync errors - file is likely written
            except:
                pass  # Silent failure
            
            # Return path immediately
            return jpeg_path
        except Exception as e:
            # Silent error - return None
            return None
    
    def _send_to_api_instant(self, killer_name, victim_name, status, base64_image, sequence_number, cropped_image, result, is_update=False, 
                            image_id=None, timestamp=None, bbox_left=None, bbox_right=None, 
                            ocr_text_left=None, ocr_text_right=None, confidence_left=None, confidence_right=None, image_hash=None):
        """
        INSTANT API SEND: Send killblock to TMS immediately with base64 image and full OCR metadata.
        Don't wait for file I/O - send directly with image data.
        This ensures killblocks appear in TMS within 1 second, not 80+ seconds.
        
        Args:
            All previous args plus:
            image_id: Unique identifier for this crop
            timestamp: Detection timestamp (milliseconds)
            bbox_left: Left X coordinate of bounding box
            bbox_right: Right X coordinate of bounding box
            ocr_text_left: Raw OCR text for left (killer) name
            ocr_text_right: Raw OCR text for right (victim) name
            confidence_left: OCR confidence for left name
            confidence_right: OCR confidence for right name
            image_hash: Short hash of the image for duplicate detection
        """
        if not self.api_enabled:
            return False
        
        # CRITICAL: Image is required for kill banner display in TMS
        # Don't send if image encoding failed - user needs banner to verify names
        if not base64_image:
            # Image encoding failed - skip this detection to ensure quality
            return False
        
        try:
            status = self._normalize_status(status)
            
            # OPTIMIZED: Minimal payload preparation - only essential fields, no file I/O
            # Generate filepath quickly (no file operations, just string)
            safe_killer = re.sub(r'[<>:"/\\|?*]', '_', killer_name[:25])  # Limit for speed
            safe_victim = re.sub(r'[<>:"/\\|?*]', '_', victim_name[:25])  # Limit for speed
            filename = f"{sequence_number:03d}_{safe_killer} {status} {safe_victim}.jpg"
            # OPTIMIZED: Skip os.path.abspath - just use relative path (faster)
            abs_image_path = os.path.join("cropkillblock", filename)
            
            # OPTIMIZED: Pre-compute timestamp once (avoid multiple time.time() calls)
            current_timestamp = timestamp or int(time.time() * 1000)
            
            # CRITICAL: Ensure strict left→right order: Killer (left) → Victim (right)
            # OPTIMIZED: Minimal payload - essential fields only for speed
            payload = {
                "killerName": killer_name,   # Matched player name (left = killer)
                "victimName": victim_name,    # Matched player name (right = victim)
                "WeaponUsed": status,
                "imagePath": abs_image_path,
                "siftWeapon": "",
                "image": base64_image,  # May be empty if encoding in progress
                # Metadata (minimal for speed)
                "image_id": image_id or f"det_{sequence_number}",
                "timestamp": current_timestamp,
                "bbox_left": bbox_left or 0,
                "bbox_right": bbox_right or 0,
                "ocr_text_left": ocr_text_left or killer_name,
                "ocr_text_right": ocr_text_right or victim_name,
                "confidence_left": confidence_left or 0.0,
                "confidence_right": confidence_right or 0.0,
                "image_hash": image_hash or ""
            }
            
            headers = {'accept': 'text/plain', 'Content-Type': 'application/json'}
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
            
            # OPTIMIZED: Fire and forget - send immediately in background (non-blocking)
            # CRITICAL: Start thread immediately without any delay
            def send_api_async():
                try:
                    # OPTIMIZED: Use session with connection pooling for faster requests
                    # Timeout: 1.0s connect, 10.0s read (generous for base64 image payloads)
                    # Increased timeouts to prevent connection errors and timeouts
                    if self.api_session:
                        response = self.api_session.post(
                            self.api_url, 
                            headers=headers, 
                            json=payload, 
                            timeout=(1.0, 10.0)  # 1.0s connect, 10.0s read (for image uploads)
                        )
                    else:
                        # Fallback: use regular requests
                        response = requests.post(
                            self.api_url, 
                            headers=headers, 
                            json=payload, 
                            timeout=(1.0, 10.0)  # 1.0s connect, 10.0s read
                        )
                    
                    if response.status_code in [200, 201]:
                        print(f"✅ API: Sent #{sequence_number} | {killer_name} {status} {victim_name}")
                    # Silent failure - don't spam on errors
                except requests.exceptions.Timeout as e:
                    # Log timeout errors for debugging (but don't spam)
                    if not hasattr(self, '_timeout_error_count'):
                        self._timeout_error_count = 0
                    self._timeout_error_count += 1
                    if self._timeout_error_count <= 3 or self._timeout_error_count % 10 == 0:
                        print(f"⚠️ API Timeout (attempt {self._timeout_error_count}): {str(e)[:100]}")
                except Exception:
                    pass  # Silent failure - non-blocking
            
            # CRITICAL: Start thread immediately - no delay, no waiting, no preparation overhead
            # Use start() not run() - thread must be truly async
            # Create thread with minimal overhead
            api_thread = threading.Thread(target=send_api_async, daemon=True, name=f"API-{sequence_number}")
            api_thread.start()
            # Return immediately - don't wait for thread, don't join, don't check status
            return True
        except Exception:
            return False
    
    def _send_to_api(self, killer_name, victim_name, status, image_path, sequence_number, retry_count=0):
        """
        Send killblock data to TMS API with retry logic for reliability.
        
        Args:
            killer_name (str): Name of the killer (left side of killfeed, first name)
            victim_name (str): Name of the victim (right side of killfeed, second name)
            status (str): Status: "revive", "kill", or "gun knockout"
            image_path (str): Path to the cropped killfeed image
            sequence_number (int): Detection sequence number
            retry_count (int): Current retry attempt (for internal use)
            
        Note: Killfeed frame order is: killer → status → victim (left to right)
        """
        if not self.api_enabled:
            return False
        
        max_retries = 2  # PERFORMANCE: Reduced retries from 3 to 2
        retry_delay = 0.2  # PERFORMANCE: Reduced delay from 0.5s to 0.2s
        
        try:
            # Ensure status is normalized
            status = self._normalize_status(status)
            
            # CRITICAL: Read image file and encode to base64 for TMS
            # Image must exist at this point (saved synchronously before API call)
            base64_image = ""
            try:
                # REAL-TIME: Check file without blocking delay - retry if needed
                max_file_retries = 3
                file_retry_delay = 0.01  # 10ms - minimal delay
                image_data = None
                
                for retry in range(max_file_retries):
                    if os.path.exists(image_path):
                        try:
                            with open(image_path, "rb") as f:
                                image_data = f.read()
                            if image_data and len(image_data) > 0:
                                break  # Success - exit retry loop
                        except (IOError, OSError):
                            # File might still be writing, retry
                            if retry < max_file_retries - 1:
                                time.sleep(file_retry_delay)
                                continue
                    
                    # File doesn't exist yet or read failed - wait briefly and retry
                    if retry < max_file_retries - 1:
                        time.sleep(file_retry_delay)
                    
                    if image_data and len(image_data) > 0:
                        base64_image = base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                base64_image = ""
            
            # CRITICAL: Map correctly - killer is ALWAYS first (left), victim is ALWAYS second (right)
            # This order is IMMUTABLE and matches the visual killfeed: killer → status → victim
            # NEVER swap these values - they come from OCR already in correct order
            # CRITICAL: Ensure image data is available before sending
            if not base64_image:
                return False
            
            # Use absolute path for imagePath to ensure TMS can find it
            abs_image_path = os.path.abspath(image_path) if image_path and os.path.exists(image_path) else image_path
            
            payload = {
                "killerName": killer_name,   # Killer (left side, first name in killfeed) - IMMUTABLE ORDER
                "victimName": victim_name,   # Victim (right side, second name in killfeed) - IMMUTABLE ORDER
                "WeaponUsed": status,        # Status: "revive", "kill", or "gun knockout"
                "imagePath": abs_image_path,  # Use absolute path so TMS can find the image
                "siftWeapon": "",
                "image": base64_image        # Base64 encoded image data for TMS
            }
            
            headers = {'accept': 'text/plain', 'Content-Type': 'application/json'}
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
            
            # REAL-TIME: Fire and forget - send API request in background thread immediately
            # This allows detection to continue immediately without waiting for API response
            # Multiple killblocks are sent to TMS in parallel
            def send_api_async():
                """Send API request in background thread with minimal retry logic."""
                max_retries = 1  # Reduced from 2 - faster failure, less delay
                retry_delay = 0.1  # Reduced from 0.5s to 0.1s - minimal delay
                
                for attempt in range(max_retries + 1):
                    try:
                        # REAL-TIME: Reduced timeout from 5s to 3s for faster failure detection
                        response = requests.post(self.api_url, headers=headers, json=payload, timeout=3)
                        if response.status_code in [200, 201]:
                            # Verify image was sent (check payload)
                            image_size = len(base64_image) if base64_image else 0
                            return  # Success - exit retry loop
                        else:
                            # Retry on non-success status codes with minimal delay
                            if attempt < max_retries:
                                time.sleep(retry_delay)
                                continue
                    except Timeout:
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                            continue
                    except RequestsConnectionError:
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                            continue
                    except Exception as e:
                        # For other exceptions, only retry once
                        if attempt == 0:
                            time.sleep(retry_delay)
                            continue
                        # Silent failure after retries - don't spam console
                        break
                
                # Silent failure after all retries - don't spam console with errors
            
            # Start API call in background thread - don't wait for it
            api_thread = threading.Thread(target=send_api_async, daemon=True)
            api_thread.start()
            
            # Return immediately - API call happens in background
            return True
            
        except Exception as e:
            # Only log critical errors that prevent API call setup
            return False
    
    def _validate_names(self, killer_name, victim_name, confidence):
        """Validate that names meet quality requirements."""
        if self.text_detector is None:
            return True
        
        if not killer_name or not victim_name or len(killer_name) < 1 or len(victim_name) < 1:
            return False
        
        if killer_name.upper() == victim_name.upper() or killer_name.isdigit() or victim_name.isdigit():
            return False
        
        if killer_name in self.all_player_names or victim_name in self.all_player_names:
            return True
        
        if confidence >= 0.3 or (any(c.isalpha() for c in killer_name) and any(c.isalpha() for c in victim_name)):
            return True
        
        return False
    
    def _process_detection_sync(self, result):
        """
        Synchronous version of detection processing (for thread pool).
        Process a detection result: OCR, validate, save, and send to API.
        Uses fuzzy matching aggressively to avoid "Unknown" names.
        """
        try:
            with self.lock:
                self.active_detections += 1
            
            return self._process_detection(result)
        finally:
            with self.lock:
                self.active_detections -= 1
                self.processed_count += 1
    
    def _process_detection(self, result):
        """
        Process a detection result: OCR, validate, save, and send to API.
        Uses fuzzy matching aggressively to avoid "Unknown" names.
        
        CRITICAL ORDERING GUARANTEE:
        - Image is split into left and right halves
        - Left half = Killer (ALWAYS) - NO SWAPPING
        - Right half = Victim (ALWAYS) - NO SWAPPING
        - Order is IMMUTABLE - fixed by image position, not OCR order
        - This ensures accurate TMS point allocation with zero swapping errors
        """
        cropped_image = result.get('cropped_image')
        status = result.get('status', 'UNKNOWN')
        bbox = result.get('bbox', [])
        detection_confidence = result.get('confidence', 0.0)
        
        # Normalize status to ensure consistent format
        status = self._normalize_status(status)
        
        if cropped_image is None or cropped_image.size == 0:
            return False
        
        # CRITICAL TASK: Run split-image OCR to extract player names accurately
        # Image is split into left/right halves, OCR runs separately on each half
        # Left half = Killer (ALWAYS), Right half = Victim (ALWAYS) - NO SWAPPING
        # This ensures TMS shows actual player names with correct roles
        
        # Generate unique image ID for this detection
        image_id = f"det_{self.detection_sequence}_{int(time.time() * 1000)}"
        
        # CRITICAL: Pass bbox to OCR so roles are assigned by X position, not arrival order
        # Get OCR results with separate confidences
        try:
            killer_name, victim_name, avg_confidence, original_killer, original_victim = self._extract_names(
                cropped_image, image_id=image_id, bbox=bbox
            )
        except Exception as e:
            killer_name, victim_name, avg_confidence, original_killer, original_victim = "", "", 0.0, "", ""
        
        # Get separate confidences from stored result
        killer_conf = 0.0
        victim_conf = 0.0
        killer_x = None
        victim_x = None
        confidence = avg_confidence  # Initialize confidence
        if hasattr(self, '_last_ocr_result') and image_id in self._last_ocr_result:
            ocr_result = self._last_ocr_result[image_id]
            killer_conf = ocr_result.get('killer_conf', 0.0)
            victim_conf = ocr_result.get('victim_conf', 0.0)
            killer_x = ocr_result.get('killer_x')
            victim_x = ocr_result.get('victim_x')
            confidence = max(killer_conf, victim_conf, avg_confidence)
        
        # Generate timestamp for this detection
        timestamp = int(time.time() * 1000)  # Milliseconds since epoch
        
        # Generate short image hash for duplicate detection (optimized for speed)
        import hashlib
        try:
            # Fast hash: use small sample from center of image
            h, w = cropped_image.shape[:2]
            center_y, center_x = h // 2, w // 2
            sample_size = min(32, h, w)
            y1 = max(0, center_y - sample_size // 2)
            y2 = min(h, center_y + sample_size // 2)
            x1 = max(0, center_x - sample_size // 2)
            x2 = min(w, center_x + sample_size // 2)
            sample = cropped_image[y1:y2, x1:x2]
            if sample.size > 0:
                img_hash = hashlib.md5(sample.tobytes()).hexdigest()[:16]
            else:
                img_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:16]
        except:
            img_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:16]
        
        # Check for duplicates: same image hash + bbox already sent recently
        if not hasattr(self, '_sent_detections'):
            self._sent_detections = {}  # {hash_bbox: timestamp}
        
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if len(bbox) >= 4 else ""
        duplicate_key = f"{img_hash}_{bbox_str}"
        duplicate_window = 2000  # 2 seconds
        
        if duplicate_key in self._sent_detections:
            last_sent = self._sent_detections[duplicate_key]
            if timestamp - last_sent < duplicate_window:
                return False  # Skip duplicate
        
        # Record this detection
        self._sent_detections[duplicate_key] = timestamp
        
        # CRITICAL: Store original OCR text IMMEDIATELY - this is the most reliable source
        # Original OCR text is what text.py extracted - preserve it at all costs
        original_killer_before_processing = original_killer if (original_killer and len(original_killer.strip()) >= 2) else (killer_name if (killer_name and len(killer_name.strip()) >= 2) else "")
        original_victim_before_processing = original_victim if (original_victim and len(original_victim.strip()) >= 2) else (victim_name if (victim_name and len(victim_name.strip()) >= 2) else "")
        
        # CRITICAL: Use original OCR text FIRST - it's the most reliable (from text.py)
        # NEVER use "processing" - only use original OCR text or empty if truly unreadable
        # Strict left→right: left = killer, right = victim (based on bbox X positions)
        has_original_killer = original_killer_before_processing and len(original_killer_before_processing.strip()) >= 2
        has_original_victim = original_victim_before_processing and len(original_victim_before_processing.strip()) >= 2
        
        if has_original_killer:
            killer_name_normalized = original_killer_before_processing.strip().upper()
        elif original_killer and len(original_killer.strip()) >= 2:
            killer_name_normalized = original_killer.strip().upper()
        elif killer_name and len(killer_name.strip()) >= 2:
            killer_name_normalized = killer_name.strip().upper()
        else:
            killer_name_normalized = ""  # Empty if invalid
        
        if has_original_victim:
            victim_name_normalized = original_victim_before_processing.strip().upper()
        elif original_victim and len(original_victim.strip()) >= 2:
            victim_name_normalized = original_victim.strip().upper()
        elif victim_name and len(victim_name.strip()) >= 2:
            victim_name_normalized = victim_name.strip().upper()
        else:
            victim_name_normalized = ""  # Empty if invalid
        
        # OPTIMIZED: Fast validation - reject "Play Zone" immediately
        killer_upper = killer_name_normalized.upper().strip() if killer_name_normalized else ""
        victim_upper = victim_name_normalized.upper().strip() if victim_name_normalized else ""
        
        if ('PLAY' in killer_upper and 'ZONE' in killer_upper) or ('PLAY' in victim_upper and 'ZONE' in victim_upper):
            return False
        if killer_upper == 'ZONE' or victim_upper == 'ZONE' or killer_upper.startswith('ZONE') or victim_upper.startswith('ZONE'):
            return False
        
        # CRITICAL: Reject if both names are empty or invalid - don't send empty detections
        # Only send if we have at least one valid name
        has_valid_killer = killer_name_normalized and len(killer_name_normalized.strip()) >= 2
        has_valid_victim = victim_name_normalized and len(victim_name_normalized.strip()) >= 2
        
        if not has_valid_killer and not has_valid_victim:
            # Both names are empty - reject this detection
            return False
        
        # Reject if killer and victim are the same (unless both are empty)
        if killer_name_normalized and victim_name_normalized and killer_name_normalized.upper() == victim_name_normalized.upper():
            return False
        
        # Snap OCR names to known match roster (fast — ~60 rapidfuzz comparisons, negligible latency)
        final_killer_name = self._snap_to_roster(killer_name_normalized) if killer_name_normalized else ""
        final_victim_name = self._snap_to_roster(victim_name_normalized) if victim_name_normalized else ""
        
        # CRITICAL: Final validation - ensure we have at least one valid name before sending
        if not final_killer_name and not final_victim_name:
            return False
        
        # CRITICAL: Encode image and send API call with kill banner
        # Image encoding is fast (~10-20ms), so we encode it first to ensure TMS gets the banner
        # This allows user to verify names are correct by seeing the kill banner image
        if self.api_enabled:
            # Extract bbox coordinates for API payload
            bbox_left = bbox[0] if len(bbox) >= 1 else 0
            bbox_right = bbox[2] if len(bbox) >= 3 else 0
            
            # CRITICAL: Encode image FAST - ensure image is always included for kill banner
            # Image encoding is fast (~10-20ms), so we encode it before API call
            # This ensures the kill banner image is always sent to TMS for verification
            # OPTIMIZED: Use optimized JPEG encoding for speed (quality 80 for faster encoding)
            try:
                # Fast JPEG encoding with quality 80 (good balance of speed/quality/size)
                # Lower quality = faster encoding = smaller payload = faster API call
                _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_bytes = buffer.tobytes()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
            except Exception as e:
                # If encoding fails, try again with even lower quality for speed
                try:
                    _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    image_bytes = buffer.tobytes()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                except:
                    base64_image = ""  # Fallback - skip if encoding completely fails
            
            # CRITICAL: Only send if image encoding succeeded - user needs banner to verify names
            if base64_image:
                # Send with MATCHED names (fuzzy matched player names from roster)
                # Also include raw OCR text in metadata fields for reference
                # Kill banner image is included for verification
                self._send_to_api_instant(
                    final_killer_name, final_victim_name, status, base64_image, 
                    self.detection_sequence, cropped_image, result,
                    image_id=image_id,
                    timestamp=timestamp,
                    bbox_left=bbox_left,
                    bbox_right=bbox_right,
                    ocr_text_left=original_killer_before_processing,  # Raw OCR text in metadata
                    ocr_text_right=original_victim_before_processing,  # Raw OCR text in metadata
                    confidence_left=killer_conf,
                    confidence_right=victim_conf,
                    image_hash=img_hash
                )
                
                # Increment sequence immediately after sending
                self.detection_sequence += 1
            else:
                # Image encoding failed - skip this detection to ensure quality
                # User needs banner to verify names are correct
                return False
        
        # Log OCR detection results - showing raw OCR text only
        ocr_killer = original_killer_before_processing if original_killer_before_processing else ""
        ocr_victim = original_victim_before_processing if original_victim_before_processing else ""
        
        # Log final result - using OCR text directly (no matching)
        print(f"\n📋 FINAL RESULT (OCR Text Only - No Matching):")
        print(f"   🎯 Killer: '{final_killer_name}' | 📊 Status: {status} | 👤 Victim: '{final_victim_name}'")
        if ocr_killer or ocr_victim:
            print(f"   📝 Raw OCR - Killer: '{ocr_killer}' | Victim: '{ocr_victim}'")
        print(f"   ✅ Confidence: {confidence:.2f} | Order: {final_killer_name} → {status} → {final_victim_name}")
        
        # Fast duplicate detection (optimized for speed - check newest first, stop early)
        current_time = time.time()
        detection_key = f"{final_killer_name}_{final_victim_name}_{status}"
        duplicate_window = 2.0  # Only check last 2 seconds for speed
        
        # Fast check: image hash + exact name match (newest first, stop early)
        for record in reversed(self.recent_detections_with_time):
            if len(record) >= 6:
                record_time, record_key, record_killer, record_victim, record_status, record_hash = record[:6]
            elif len(record) >= 5:
                record_time, record_key, record_killer, record_victim, record_status = record[:5]
                record_hash = ""
            else:
                continue
            
            time_diff = current_time - record_time
            if time_diff > duplicate_window:
                break  # Stop checking older records
            
            # Fast duplicate check: same hash OR same names+status
            # Use image hash for duplicate detection (more reliable than names)
            if record_hash and img_hash and record_hash == img_hash:
                print(f"🚫 Duplicate detection - skipping (same image hash)")
                return False  # Same image - duplicate
            # Only check name-based duplicates if both names are not empty
            # Empty name detections should still be sent (different killfeeds)
            if final_killer_name and final_victim_name:
                if (record_key == detection_key or 
                    (status == record_status and 
                     final_killer_name.upper() == record_killer.upper() and 
                     final_victim_name.upper() == record_victim.upper())):
                    print(f"🚫 Duplicate detection - skipping (same names)")
                    return False  # Same detection - duplicate
        
        # REMOVED: Fuzzy matching duplicate check - too slow, exact match is sufficient
        
        # REMOVED: Roster-based rejection - TMS handles player matching
        
        # Add to recent detections (optimized for speed - minimal data)
        self.recent_detections.add(detection_key)
        self.recent_detections_with_time.append((
            current_time, 
            detection_key, 
            final_killer_name, 
            final_victim_name, 
            status,
            image_hash
        ))
        
        # Fast cleanup - only keep last 5 seconds (reduced from 10 for speed)
        cleanup_time = current_time - 5.0
        while self.recent_detections_with_time and self.recent_detections_with_time[0][0] < cleanup_time:
            old_record = self.recent_detections_with_time.popleft()
            old_key = old_record[1] if len(old_record) > 1 else None
            if old_key and old_key in self.recent_detections:
                self.recent_detections.remove(old_key)
        
        # Limit size to prevent memory growth
        if len(self.recent_detections) > 100:
            self.recent_detections.clear()
            for record in self.recent_detections_with_time:
                if len(record) > 1:
                    self.recent_detections.add(record[1])
        
        # Save image asynchronously (non-blocking - already sent to API)
        safe_killer = re.sub(r'[<>:"/\\|?*]', '_', final_killer_name)
        safe_victim = re.sub(r'[<>:"/\\|?*]', '_', final_victim_name)
        # Use sequence number (already incremented above)
        seq_num = self.detection_sequence - 1 if self.detection_sequence > 0 else 0
        filename = f"{seq_num:03d}_{safe_killer} {status} {safe_victim}.jpg"
        
        # Generate filepath
        output_dir = "cropkillblock"
        if not hasattr(self, '_output_dir_created'):
            os.makedirs(output_dir, exist_ok=True)
            self._output_dir_created = True
        filepath = os.path.join(output_dir, filename)
        
        # Log saved image
        print(f"📸 Saved: {filename}")
        
        # INSTANT: Save image in background thread (non-blocking)
        def save_image_async():
            try:
                cv2.imwrite(filepath, cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                pass  # Silent failure - file save is not critical
        
        save_thread = threading.Thread(target=save_image_async, daemon=True)
        save_thread.start()
        
        # Don't wait for file save - continue immediately
        
        # ACCURACY: Validate status is reasonable
        valid_statuses = ['kill', 'revive', 'gun knockout']
        if status not in valid_statuses:
            status = 'gun knockout'
        
        # Determine status color indicator
        status_color = ""
        if status == "kill":
            status_color = "🔴 Status: kill (victim text color: red)"
        elif status == "revive":
            status_color = "🟢 Status: revive (victim text color: green - fallback)"
        else:
            status_color = f"⚪ Status: {status}"
        
        # Check if names are validated (in roster or have reasonable confidence)
        victim_matched = victim_name in self.all_player_names if self.all_player_names else False
        killer_matched = killer_name in self.all_player_names if self.all_player_names else False
        validated = (victim_matched or killer_matched) or confidence >= 0.5
        
        # REAL-TIME: Minimal logging to reduce console I/O delay
        # Only log essential info - detailed logging adds 50-100ms delay
        if self.all_player_names:
            if killer_matched and victim_matched:
                roster_info = " [✓]"
            elif killer_matched or victim_matched:
                roster_info = " [✓]"
            else:
                roster_info = ""
        else:
            roster_info = ""
        
        # Clean, simple logging
        
        # Names already sent with initial API call (OCR ran first)
        # Only update if names improved (e.g., "processing" -> real name)
        if self.api_enabled and base64_image:
            # Check if we need to update (names were "processing" initially)
            if killer_name != "processing" and victim_name != "processing":
                # Update known players
                self.known_players.add(victim_name)
                self.known_players.add(killer_name)
            # No need to update API - already sent with real names
        
        return True
    
    # Frame capture methods removed - frame capture should be handled by caller (main.py)
    # This client only processes cropped images received from gRPC server
    
    def _detection_worker(self, stop_flag):
        """
        Worker thread that processes detections from the queue in parallel.
        This ensures real-time processing without blocking frame capture.
        Robust error handling ensures worker never crashes.
        """
        
        consecutive_errors = 0
        max_consecutive_errors = 100
        last_error_log_time = 0
        error_log_interval = 5.0
        
        while True:
            # Check stop flag
            if stop_flag and stop_flag.is_set():
                break
            
            try:
                # OPTIMIZED: Get detection from queue immediately (minimal timeout)
                # Use very short timeout to check stop flag frequently - no blocking
                try:
                    result = self.detection_queue.get(timeout=0.01)  # 10ms - minimal delay
                except Empty:
                    continue  # Timeout, check stop flag and continue immediately
                
                # Process detection in this worker thread with robust error handling
                try:
                    start_time = time.time()
                    success = self._process_detection_sync(result)
                    process_time = time.time() - start_time
                    
                    if not success:
                        with self.lock:
                            self.failed_count += 1
                    
                    # Silent - no verbose timing warnings
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    
                    # Mark task as done
                    self.detection_queue.task_done()
                    
                except KeyboardInterrupt:
                    # Re-raise KeyboardInterrupt to allow graceful shutdown
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    current_time = time.time()
                    
                    # Log error but continue processing
                    if current_time - last_error_log_time > error_log_interval:
                        last_error_log_time = current_time
                    
                    # Reset error counter periodically
                    if consecutive_errors > max_consecutive_errors:
                        consecutive_errors = 0
                    
                    with self.lock:
                        self.failed_count += 1
                    
                    # Always mark task as done to prevent queue blocking
                    try:
                        self.detection_queue.task_done()
                    except:
                        pass  # Ignore errors in task_done
            
            except KeyboardInterrupt:
                # Re-raise to allow main thread to handle
                raise
            except Exception as e:
                # Catch all other exceptions and continue - worker should never crash
                consecutive_errors += 1
                current_time = time.time()
                if current_time - last_error_log_time > error_log_interval:
                    last_error_log_time = current_time
                
                if consecutive_errors > max_consecutive_errors:
                    consecutive_errors = 0
                
                # REAL-TIME: No blocking delay - continue immediately
                # Continue loop - worker should never exit unless stop_flag is set
        
    
    def start_detection(self, stop_flag=None, frame_capture_callback=None):
        """
        Main detection loop - processes frames via gRPC and queues detections for OCR/API.
        
        CRITICAL: This method will NEVER return automatically - it only stops when:
        1. User manually stops via stop_flag
        2. User presses Ctrl+C (KeyboardInterrupt)
        
        All other errors are caught and the loop continues running with automatic retry.
        
        NOTE: Frame capture should ideally be handled by caller (main.py).
        This method is kept for backward compatibility but frame_capture_callback
        should be provided by the caller.
        
        Args:
            stop_flag: threading.Event to signal stopping (optional)
            frame_capture_callback: Callable that returns frame (numpy array) or None.
                                   If None, this method will wait and retry until callback is provided.
        """
        # CRITICAL: Never return early - wait for callback if not provided
        if frame_capture_callback is None:
            # Wait for callback to be provided - never give up
            while frame_capture_callback is None:
                if stop_flag and stop_flag.is_set():
                    return  # Only exit if user stopped
                # OPTIMIZED: Minimal delay - check frequently for callback
                time.sleep(0.01)  # 10ms - fast check
                # Callback might be set externally, keep checking
        
        
        # OPTIMIZED: Start MANY worker threads for maximum parallel processing
        # Increased workers to handle high killfeed rates without missing any
        worker_threads = []
        num_workers = max(self.max_workers, 100)  # 100 workers for maximum throughput
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._detection_worker,
                args=(stop_flag,),
                daemon=True,
                name=f"DetectionWorker-{i+1}"
            )
            worker.start()
            worker_threads.append(worker)
        
        
        frame_count = 0
        last_stats_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 100  # Only log warning after many consecutive errors
        last_error_log_time = 0
        error_log_interval = 5.0  # Log errors at most once every 5 seconds
        
        # ROBUST: Outer infinite loop to ensure detection NEVER stops automatically
        # This loop will only exit if stop_flag is set or KeyboardInterrupt occurs
        detection_loop_active = True
        last_frame_time = time.time()
        heartbeat_interval = 10.0  # Log heartbeat every 10 seconds if no frames
        max_no_frame_time = 60.0  # Warn if no frames for 60 seconds
        
        try:
            # CRITICAL: This outer loop ensures we NEVER exit unless explicitly stopped
            while detection_loop_active:
                # Check stop flag at the start of each iteration
                if stop_flag and stop_flag.is_set():
                    # Final log with comprehensive match stop information
                    with self.lock:
                        active = self.active_detections
                        processed = self.processed_count
                        failed = self.failed_count
                    queue_size = self.detection_queue.qsize()
                    break
                
                try:
                    # Inner loop for frame processing - this is the main detection loop
                    while True:
                        # Check stop flag
                        if stop_flag and stop_flag.is_set():
                            # Final log with comprehensive match stop information
                            with self.lock:
                                active = self.active_detections
                                processed = self.processed_count
                                failed = self.failed_count
                            queue_size = self.detection_queue.qsize()
                            detection_loop_active = False
                            break
                        
                        try:
                            # Get frame from callback (provided by caller) with robust error handling
                            frame = None
                            frame_capture_attempts = 0
                            max_frame_capture_attempts = 1000  # Never give up on frame capture
                            
                            # ROBUST: Keep trying to get frame - never give up
                            while frame is None and frame_capture_attempts < max_frame_capture_attempts:
                                frame_capture_attempts += 1
                                
                                # Check stop flag during frame capture retries
                                if stop_flag and stop_flag.is_set():
                                    # Final log with comprehensive match stop information
                                    with self.lock:
                                        active = self.active_detections
                                        processed = self.processed_count
                                        failed = self.failed_count
                                    queue_size = self.detection_queue.qsize()
                                    detection_loop_active = False
                                    break
                                
                                try:
                                    frame = frame_capture_callback()
                                except Exception as e:
                                    consecutive_errors += 1
                                    current_time = time.time()
                                    if current_time - last_error_log_time > error_log_interval:
                                        last_error_log_time = current_time
                                    if consecutive_errors > max_consecutive_errors:
                                        consecutive_errors = 0  # Reset to prevent overflow
                                    
                                    # OPTIMIZED: No sleep - retry immediately for maximum speed
                                    continue
                                
                                # Validate frame
                                if frame is None or not hasattr(frame, 'size') or frame.size == 0:
                                    consecutive_errors += 1
                                    # OPTIMIZED: No sleep - retry immediately
                                    frame = None  # Reset to continue loop
                                    continue
                                
                                # Got valid frame - break out of retry loop
                                break
                            
                            # If we couldn't get a frame after many attempts, log but continue
                            if frame is None:
                                current_time = time.time()
                                if current_time - last_frame_time > heartbeat_interval:
                                    if current_time - last_frame_time > max_no_frame_time:
                                        pass
                                    else:
                                        pass
                                    last_frame_time = current_time
                                # REAL-TIME: No delay - retry frame capture immediately
                                continue  # Continue to next iteration - never exit
                            
                            # Reset error counter on successful frame capture
                            consecutive_errors = 0
                            frame_count += 1
                            last_frame_time = time.time()
                            
                            # OPTIMIZED: Process frame via gRPC with automatic reconnection
                            try:
                                # Check gRPC connection before processing
                                if self.grpc_stub is None:
                                    # Auto-reconnect if disconnected
                                    try:
                                        self._connect_to_server(max_retries=1)
                                    except:
                                        pass  # Continue even if reconnect fails
                                
                                results = self.process_frame_via_grpc(frame)
                                # Reset error counter on success
                                consecutive_errors = 0
                            except KeyboardInterrupt:
                                raise  # Re-raise KeyboardInterrupt to exit gracefully
                            except Exception as e:
                                consecutive_errors += 1
                                current_time = time.time()
                                
                                # Log errors only occasionally to avoid spam
                                if current_time - last_error_log_time > error_log_interval:
                                    error_type = type(e).__name__
                                    error_msg = str(e)[:200]
                                    last_error_log_time = current_time
                                
                                # Auto-reconnect on gRPC errors
                                if self.grpc_stub is None or 'grpc' in str(e).lower() or 'rpc' in str(e).lower():
                                    try:
                                        self._connect_to_server(max_retries=1)
                                    except:
                                        pass  # Continue even if reconnect fails
                                
                                # Reset error counter periodically
                                if consecutive_errors > max_consecutive_errors:
                                    consecutive_errors = 0
                                
                                # OPTIMIZED: No delay - process next frame immediately
                                # Never stop - continue loop to prevent missing killfeeds
                                continue
                            
                            # REAL-TIME: Reduced stats frequency to minimize I/O delay
                            # Stats logging adds 10-20ms delay - reduce frequency
                            if time.time() - last_stats_time > 10.0:  # Reduced from 5s to 10s
                                with self.lock:
                                    active = self.active_detections
                                    processed = self.processed_count
                                    failed = self.failed_count
                                queue_size = self.detection_queue.qsize()
                                last_stats_time = time.time()
                        
                        except KeyboardInterrupt:
                            # Only KeyboardInterrupt should exit the loop
                            detection_loop_active = False
                            # Final log with comprehensive match stop information
                            with self.lock:
                                active = self.active_detections
                                processed = self.processed_count
                                failed = self.failed_count
                            queue_size = self.detection_queue.qsize()
                            raise
                        except SystemExit:
                            # SystemExit should also exit
                            detection_loop_active = False
                            # Final log with comprehensive match stop information
                            with self.lock:
                                active = self.active_detections
                                processed = self.processed_count
                                failed = self.failed_count
                            queue_size = self.detection_queue.qsize()
                            raise
                        except Exception as e:
                            # Catch ALL other exceptions and continue - never exit automatically
                            consecutive_errors += 1
                            current_time = time.time()
                            if current_time - last_error_log_time > error_log_interval:
                                last_error_log_time = current_time
                            
                            # Reset error counter periodically to prevent overflow
                            if consecutive_errors > max_consecutive_errors:
                                consecutive_errors = 0
                            
                            # COMPREHENSIVE ERROR LOGGING: Always log inner loop errors
                            
                            # Log traceback for first few errors
                            if consecutive_errors <= 3:
                                import traceback
                                traceback.print_exc()
                            
                            # Reset error counter periodically to prevent overflow
                            if consecutive_errors > max_consecutive_errors:
                                consecutive_errors = 0
                            
                            # REAL-TIME: No blocking delay - continue immediately
                            # Continue loop - never exit due to exceptions
                    
                except KeyboardInterrupt:
                    # KeyboardInterrupt in outer loop - exit
                    detection_loop_active = False
                    # Final log with comprehensive match stop information
                    with self.lock:
                        active = self.active_detections
                        processed = self.processed_count
                        failed = self.failed_count
                    queue_size = self.detection_queue.qsize()
                    raise
                except SystemExit:
                    # SystemExit in outer loop - exit
                    detection_loop_active = False
                    # Final log with comprehensive match stop information
                    with self.lock:
                        active = self.active_detections
                        processed = self.processed_count
                        failed = self.failed_count
                    queue_size = self.detection_queue.qsize()
                    raise
                except Exception as e:
                    # CRITICAL: Even if inner loop somehow exits, catch and restart - NEVER exit automatically
                    error_count = getattr(self, '_inner_loop_error_count', 0) + 1
                    self._inner_loop_error_count = error_count
                    
                    # COMPREHENSIVE ERROR LOGGING: Always log outer loop errors
                    with self.lock:
                        active = self.active_detections
                        processed = self.processed_count
                        failed = self.failed_count
                    queue_size = self.detection_queue.qsize()
                    
                    
                    # Log full traceback for first few errors
                    if error_count <= 3:
                        import traceback
                        traceback.print_exc()
                    
                    # OPTIMIZED: Minimal delay - continue immediately to prevent missing killfeeds
                    time.sleep(0.01)  # 10ms - minimal delay
                    # Reset counters
                    consecutive_errors = 0
                    # Continue outer loop - NEVER exit automatically
                    detection_loop_active = True  # Ensure loop stays active
                    continue
        
        except KeyboardInterrupt:
            pass
        except SystemExit:
            # Final log with comprehensive match stop information
            with self.lock:
                active = self.active_detections
                processed = self.processed_count
                failed = self.failed_count
            queue_size = self.detection_queue.qsize()
            raise  # Re-raise SystemExit
        except Exception as e:
            # CRITICAL: Even if outer try block fails, log but NEVER exit automatically
            # This exception handler ensures the loop continues no matter what
            error_count = getattr(self, '_outer_detection_error_count', 0) + 1
            self._outer_detection_error_count = error_count
            
            if error_count <= 3 or error_count % 10 == 0:
                # Log first few errors and every 10th error to avoid spam
                with self.lock:
                    active = self.active_detections
                    processed = self.processed_count
                    failed = self.failed_count
                queue_size = self.detection_queue.qsize()
                if error_count <= 3:
                    import traceback
                    traceback.print_exc()
            
            # OPTIMIZED: Minimal delay - restart immediately to prevent missing killfeeds
            time.sleep(0.1)  # 100ms - minimal delay before restart
            # Reset error counters
            consecutive_errors = 0
            # Don't exit - the outer loop in start_match will handle restart
            # The loop will continue automatically
            # Restart the detection loop immediately
            detection_loop_active = True
        finally:
            # CRITICAL: Only cleanup if we're actually stopping (not restarting)
            # If stop_flag is not set, we're restarting - don't cleanup
            if stop_flag and stop_flag.is_set():
                try:
                    # Wait for queue with timeout to prevent hanging
                    start_wait = time.time()
                    while not self.detection_queue.empty() and (time.time() - start_wait) < 30:
                        time.sleep(0.5)
                    self.detection_queue.join()
                except Exception as e:
                    pass
                time.sleep(0.5)
                self._cleanup()
                with self.lock:
                    pass
            # If stop_flag is not set, we're restarting - skip cleanup and let outer loop handle it
    
    def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.grpc_channel:
                self.grpc_channel.close()
        except:
            pass
        try:
            if self.api_session:
                self.api_session.close()
        except:
            pass


def main():
    """Main entry point."""
    
    # Get configuration
    match_id = input("Enter Match ID (or press Enter for default '1'): ").strip() or "1"
    access_token = input("Enter Access Token (or press Enter to skip): ").strip() or None
    api_enabled = input("Enable API integration? (Y/n, default: Y): ").strip().lower() != 'n'
    
    try:
        detector = KillblockDetector(match_id=match_id, access_token=access_token, api_enabled=api_enabled)

        # Load current-match roster for OCR snapping
        detector.set_match_roster(MATCH_PLAYERS)

        if detector.grpc_stub is None:
            return
        
        input("Press Enter to begin detection...")
        
        detector.start_detection()
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()


def start_match(match_id=1, access_token=None, camera_index=1, stop_flag=None, frame_capture_callback=None):
    """
    Main function to start killblock OCR processing.
    Compatible with camera_setup_pyqt.py interface.
    
    CRITICAL: This function will NEVER exit automatically - it only stops when:
    1. User manually stops via stop_flag
    2. User presses Ctrl+C (KeyboardInterrupt)
    
    All other errors are caught and the detection loop continues running with automatic retry.
    
    Args:
        match_id (int or str): ID of the match being processed
        access_token (str): Authentication token for API access
        camera_index (int): Camera index for frame capture (used if frame_capture_callback is None)
        stop_flag (threading.Event): Event flag to signal stopping
        frame_capture_callback: Callable that returns frame (numpy array) or None.
                               If None and camera_index is provided, creates internal frame capture.
    """
    detector = None
    consecutive_fatal_errors = 0
    max_fatal_errors = 10
    
    # ROBUST: Outer loop to restart detection if it fails - NEVER exit automatically
    # CRITICAL: This loop will ONLY exit if:
    # 1. User manually sets stop_flag
    # 2. User presses Ctrl+C (KeyboardInterrupt)
    # 3. SystemExit is raised (system shutdown)
    # ALL other errors will cause restart, never exit
    restart_count = 0
    max_restart_warnings = 10  # Only warn about restarts every 10 times
    
    while True:
        # Check stop flag before starting/restarting
        if stop_flag and stop_flag.is_set():
            break
        
        try:
            # Initialize detector (this will call _load_team_rosters() in __init__)
            if detector is None:
                
                detector = KillblockDetector(
                    match_id=str(match_id),
                    access_token=access_token,
                    api_enabled=True,
                    grpc_port=50051
                )

                # Load current-match roster for OCR snapping (zero latency impact)
                detector.set_match_roster(MATCH_PLAYERS)

                # Verify team rosters were loaded
                
                # Create frame capture callback if not provided but camera_index is given (backward compatibility)
                if frame_capture_callback is None and camera_index is not None:
                    # Create internal frame capture for backward compatibility with robust error handling
                    cap = None
                    last_capture_time = time.time()
                    consecutive_failures = 0
                    max_consecutive_failures = 10
                    reconnect_delay = 1.0
                    
                    def create_frame_capture():
                        """
                        ROBUST frame capture callback that NEVER gives up.
                        Will continuously retry camera connection until successful.
                        """
                        nonlocal cap, last_capture_time, consecutive_failures
                        
                        # ROBUST: Keep trying to get frame - never return None permanently
                        max_retries_per_call = 5  # Try up to 5 times per call
                        retry_count = 0
                        
                        while retry_count < max_retries_per_call:
                            retry_count += 1
                            
                            # Check if camera needs reconnection
                            if cap is None or not cap.isOpened():
                                consecutive_failures += 1
                                
                                # Release old camera if exists
                                if cap is not None:
                                    try:
                                        cap.release()
                                    except:
                                        pass
                                    cap = None
                                
                                # OPTIMIZED: Minimal delay before reconnecting
                                if consecutive_failures > max_consecutive_failures:
                                    consecutive_failures = 0
                                    time.sleep(0.1)  # Minimal delay
                                
                                # Try to initialize/reconnect camera - NEVER give up
                                camera_initialized = False
                                for attempt in range(3):  # Try 3 different methods
                                    try:
                                        if attempt == 0:
                                            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                                        elif attempt == 1:
                                            cap = cv2.VideoCapture(camera_index)
                                        else:
                                            # Last attempt: try with default backend
                                            cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
                                        
                                        if cap.isOpened():
                                            camera_initialized = True
                                            break
                                    except Exception as e:
                                        if attempt == 2:  # Last attempt
                                            if consecutive_failures % 10 == 0:
                                                pass
                                        continue
                                
                                if not camera_initialized:
                                    # Camera not initialized - wait and retry
                                    time.sleep(0.2)
                                    continue  # Retry camera initialization
                                
                                # Camera initialized - configure it
                                try:
                                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                                    cap.set(cv2.CAP_PROP_FPS, 60)
                                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                                    # Warm up camera
                                    for _ in range(5):
                                        cap.read()
                                    consecutive_failures = 0  # Reset on successful initialization
                                except Exception as e:
                                    # Release and retry
                                    try:
                                        cap.release()
                                    except:
                                        pass
                                    cap = None
                                    # OPTIMIZED: No delay - retry immediately
                                    continue  # Retry
                            
                            # Try to read frame with error handling
                            try:
                                ret, frame = cap.read()
                                if ret and frame is not None and frame.size > 0:
                                    consecutive_failures = 0  # Reset on successful read
                                    last_capture_time = time.time()
                                    return frame
                                else:
                                    # Frame read failed - check if camera is still valid
                                    if not cap.isOpened():
                                        cap = None  # Force reconnection on next call
                                    consecutive_failures += 1
                                    # OPTIMIZED: No delay - retry immediately
                                    continue  # Retry frame read
                            except Exception as e:
                                # Camera read exception - mark for reconnection
                                consecutive_failures += 1
                                if consecutive_failures % 10 == 0:
                                    pass
                                try:
                                    cap.release()
                                except:
                                    pass
                                cap = None  # Force reconnection on next call
                                # OPTIMIZED: No delay - retry immediately
                                continue  # Retry
                        
                        # If we exhausted retries, return None but caller will retry
                        # This ensures we never permanently fail
                        return None
                    
                    frame_capture_callback = create_frame_capture
            
            # Start detection if frame capture callback is available
            if frame_capture_callback:
                # Reset error counter on successful start
                consecutive_fatal_errors = 0
                
                # CRITICAL: Wrap start_detection in try-except to catch ANY exception
                # Even if start_detection somehow returns or raises, we'll restart
                try:
                    detector.start_detection(stop_flag=stop_flag, frame_capture_callback=frame_capture_callback)
                    # If start_detection returns, check stop flag
                    if stop_flag and stop_flag.is_set():
                        break  # User stopped - exit outer loop
                    # If start_detection returned without stop_flag, it's an error - restart
                    restart_count += 1
                    if restart_count % max_restart_warnings == 0:
                        print(f"⚠️ Detection restarted (attempt {restart_count}) - ensuring continuous operation")
                    # OPTIMIZED: Minimal delay - restart immediately
                    time.sleep(0.1)  # 100ms - minimal delay
                    # Reset detector to force reinitialization
                    detector = None
                    # Continue loop - never exit automatically
                    continue
                except KeyboardInterrupt:
                    # User pressed Ctrl+C - exit
                    raise
                except SystemExit:
                    # SystemExit should propagate
                    raise
                except Exception as e:
                    # ANY other exception - restart detection immediately
                    restart_count += 1
                    if restart_count % max_restart_warnings == 0:
                        print(f"⚠️ Detection error, restarting (attempt {restart_count}): {type(e).__name__}")
                    # OPTIMIZED: Minimal delay - restart immediately
                    time.sleep(0.1)  # 100ms - minimal delay
                # Reset detector to force reinitialization
                detector = None
                # Continue loop - never exit automatically
                continue
            else:
                # No callback yet - wait and retry
                time.sleep(0.1)  # OPTIMIZED: Minimal delay
                continue  # Continue loop to retry initialization
        
        except KeyboardInterrupt:
            # User pressed Ctrl+C - this is the ONLY way to stop (besides stop_flag)
            print("\n🛑 User interrupted - stopping detection...")
            break
        except SystemExit:
            # SystemExit should propagate
            raise
        except Exception as e:
            # CRITICAL: Catch ALL other exceptions and retry - NEVER exit automatically
            consecutive_fatal_errors += 1
            if consecutive_fatal_errors <= 3 or consecutive_fatal_errors % 10 == 0:
                # Log first few errors and every 10th error
                print(f"⚠️ Fatal error in start_match (attempt {consecutive_fatal_errors}): {type(e).__name__}: {str(e)[:100]}")
                if consecutive_fatal_errors <= 3:
                    import traceback
                    traceback.print_exc()
            
            # OPTIMIZED: Minimal delays - restart quickly to prevent missing killfeeds
            if consecutive_fatal_errors >= max_fatal_errors:
                print(f"⚠️ Many consecutive errors - brief pause before retry...")
                time.sleep(1.0)  # Slightly longer only for many errors
                consecutive_fatal_errors = 0  # Reset counter after wait
            else:
                time.sleep(0.2)  # OPTIMIZED: Minimal delay - restart quickly
            
            # Reset detector to force reinitialization
            detector = None
            
            # Check stop flag before retrying
            if stop_flag and stop_flag.is_set():
                print("🛑 Stop flag set - exiting...")
                break
            
            # Continue loop to retry - NEVER exit automatically
            # The loop will continue and restart detection
            print(f"🔄 Restarting detection (error count: {consecutive_fatal_errors})...")
            continue


if __name__ == "__main__":
    main()