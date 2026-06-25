"""
Killfeed Detection Module

Uses gRPC to communicate with remote YOLO + PaddleOCR server for detection.
No local AI models required - all inference is done server-side.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional
import cv2
import numpy as np
import time
import base64
import requests
import aiohttp
import asyncio
import threading
import json
import logging
import glob
import sys
import grpc
import re
import hashlib
from queue import Queue, Empty, Full, PriorityQueue
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from rapidfuzz import fuzz, process
from datetime import datetime
from score_ai.detection import killfeed_pb2, killfeed_pb2_grpc
from score_ai.detection.game_configs import get_game_config, is_killfeed_detection

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Global variables to store team rosters and current match info for OCR matching
TEAM_ROSTERS = {}
CURRENT_MATCH_ID = None
CURRENT_ACCESS_TOKEN = None

# Import from the same package
from score_ai.detection.killfeed_detections import KillfeedDetections
from collections import deque
from score_ai.utils.helpers import get_resource_path
from score_ai.core.config_manager import config
from score_ai.core.logging_config import get_logger

# Module-level logger
logger = get_logger(__name__)

# Optional torch for GPU acceleration (will be imported lazily if available)
torch = None
TORCH_AVAILABLE = False

# Disable all logging for maximum performance
def setup_comprehensive_logger():
    """Set up minimal logging for killfeed detection"""
    # Create a dummy logger that does nothing
    logger = logging.getLogger("killfeed_detection")
    logger.setLevel(logging.CRITICAL)  # Only critical errors
    logger.handlers.clear()
    return logger

# Initialize minimal logger
detection_logger = setup_comprehensive_logger()

# Result file for killfeed detections (local log; API is also called to post killfeed)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GAME_LOGS_DIR = os.path.join(_PROJECT_ROOT, 'game_logs')
RESULT_FILE = os.path.join(GAME_LOGS_DIR, 'result.txt')

# Compact killfeed line (gRPC screenshot pipeline): player: killer + gun + victim
KILLFEED_TABLE_LOG_PATH = os.path.join(GAME_LOGS_DIR, 'killfeed_table.log')
_killfeed_table_lock = threading.Lock()

# Timing report: processing time vs wait-before-API-post (for "in order + immediately" tuning)
KILLFEED_TIMING_REPORT_PATH = os.path.join(GAME_LOGS_DIR, 'killfeed_timing_report.csv')
_killfeed_timing_report_lock = threading.Lock()
_KILLFEED_TIMING_HEADER = "frame_number,processing_time_s,wait_before_post_s,api_post_s,killer,victim,weapon,posted_at"

def _append_killfeed_timing_report(
    frame_number,
    processing_time_s,
    wait_before_post_s,
    api_post_s,
    killer_name,
    victim_name,
    weapon_used,
    posted_at_iso,
):
    """Append one row to killfeed_timing_report.csv (thread-safe)."""
    try:
        os.makedirs(GAME_LOGS_DIR, exist_ok=True)
        with _killfeed_timing_report_lock:
            write_header = not os.path.exists(KILLFEED_TIMING_REPORT_PATH) or os.path.getsize(KILLFEED_TIMING_REPORT_PATH) == 0
            with open(KILLFEED_TIMING_REPORT_PATH, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write(_KILLFEED_TIMING_HEADER + "\n")
                # Escape CSV: wrap in quotes if contains comma
                def esc(s):
                    s = str(s) if s is not None else ""
                    return f'"{s}"' if "," in s or '"' in s else s
                row = ",".join([
                    esc(frame_number),
                    esc(processing_time_s),
                    esc(wait_before_post_s),
                    esc(api_post_s),
                    esc(killer_name),
                    esc(victim_name),
                    esc(weapon_used),
                    esc(posted_at_iso),
                ])
                f.write(row + "\n")
    except Exception as e:
        logger.debug(f"Could not write killfeed timing report: {e}")


# --- gRPC: reuse one channel+stub (creating a new channel per frame is very slow) ---
_grpc_state_lock = threading.Lock()
_grpc_channel: Any = None
_grpc_stub: Optional[Any] = None
_grpc_cache_key: Optional[tuple] = None


def _grpc_config_cache_key() -> tuple:
    gc = config.get_grpc_config()
    return (gc.get("server", ""), bool(gc.get("use_tls", False)), gc.get("tls_cert_path") or "")


def _grpc_channel_options() -> list:
    return [
        ("grpc.max_send_message_length", 32 * 1024 * 1024),
        ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ("grpc.enable_http_proxy", 0),
        ("grpc.keepalive_time_ms", 10_000),
        ("grpc.keepalive_timeout_ms", 5_000),
    ]


def _event_type_to_class_name(event_type: int) -> str:
    """Map killfeed event enum to a class-like name for downstream compatibility."""
    mapping = {
        int(killfeed_pb2.EVENT_TYPE_KILL): "kill",
        int(killfeed_pb2.EVENT_TYPE_KNOCK): "gun-knockout",
        int(killfeed_pb2.EVENT_TYPE_REVIVE): "revive",
        int(killfeed_pb2.EVENT_TYPE_UNSPECIFIED): "unspecified",
    }
    return mapping.get(int(event_type), "unknown")


def _event_type_to_ui_label(event_type: int) -> str:
    """Map killfeed event enum to frontend-facing label text."""
    mapping = {
        int(killfeed_pb2.EVENT_TYPE_KILL): "Kill",
        int(killfeed_pb2.EVENT_TYPE_KNOCK): "gun-knockout",
        int(killfeed_pb2.EVENT_TYPE_REVIVE): "R",
        int(killfeed_pb2.EVENT_TYPE_UNSPECIFIED): "Unknown",
    }
    return mapping.get(int(event_type), "Unknown")


def _normalize_weapon_for_tms_api(weapon_used: Optional[str]) -> str:
    """
    Canonical weapon/event label for TMS killfeed API (weaponUsed).
    Maps legacy bare knock labels to gun-knockout so the tournament backend and UI stay consistent.
    """
    if weapon_used is None:
        return ""
    w = str(weapon_used).strip()
    wl = w.lower()
    if wl == "knock":
        return "gun-knockout"
    # OCR inference returns bare "knockout" for generic gun knocks (not head-knockout, etc.)
    if wl == "knockout":
        return "gun-knockout"
    return w


def _combined_text_from_row(row) -> str:
    """Build a combined text string compatible with existing OCR parsing flow."""
    killer = (getattr(row, "killer", "") or "").strip()
    victim = (getattr(row, "victim", "") or "").strip()
    if killer and victim:
        event_type = int(getattr(row, "event_type", int(killfeed_pb2.EVENT_TYPE_UNSPECIFIED)))
        if event_type == int(killfeed_pb2.EVENT_TYPE_KNOCK):
            return f"{killer} knocked {victim}"
        if event_type == int(killfeed_pb2.EVENT_TYPE_REVIVE):
            return f"{killer} revived {victim}"
        return f"{killer} killed {victim}"
    return killer or victim or ""


def reset_grpc_killfeed_channel() -> None:
    """Close cached gRPC channel (e.g. after server restart or on error)."""
    global _grpc_channel, _grpc_stub, _grpc_cache_key
    with _grpc_state_lock:
        if _grpc_channel is not None:
            try:
                _grpc_channel.close()
            except Exception:
                pass
        _grpc_channel = None
        _grpc_stub = None
        _grpc_cache_key = None
        logger.info("[gRPC CONNECTION] Channel reset - will reconnect on next request")


def test_grpc_connection() -> bool:
    """
    Test the gRPC connection to the AI server.
    Returns True if connection is successful, False otherwise.
    """
    try:
        print(f"\n{'='*70}", flush=True)
        print(f"[CONNECTION TEST] Testing connection to AI Server...", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        # Get the stub (creates connection if needed)
        stub = get_grpc_predict_stub()
        
        # Create a minimal test request with a 1x1 white JPEG
        test_image = np.ones((1, 1, 3), dtype=np.uint8) * 255
        ok_enc, enc_buf = cv2.imencode(".jpg", test_image)
        if not ok_enc:
            print(f"[CONNECTION TEST] ✗ FAILED: Could not encode test image", flush=True)
            logger.error("[CONNECTION TEST] Could not encode test image")
            return False
        
        test_image_bytes = enc_buf.tobytes()
        
        # Create test request
        request = killfeed_pb2.InferRequest(image=test_image_bytes)
        
        # Try to call the service
        grpc_config = config.get_grpc_config()
        api_key = grpc_config.get("api_key", "")
        metadata = (("api-key", api_key),) if api_key else ()
        
        print(f"[CONNECTION TEST] Sending test request to gRPC server...", flush=True)
        logger.info("[CONNECTION TEST] Sending test request to gRPC server")
        
        response = stub.Infer(request, metadata=metadata, timeout=10.0)
        
        # If we got a response, connection is successful
        print(f"[CONNECTION TEST] ✓ SUCCESS: Connected to AI Server!", flush=True)
        print(f"[CONNECTION TEST] Server responded with {len(response.rows)} rows", flush=True)
        print(f"[CONNECTION TEST] Ready to capture and process frames!", flush=True)
        print(f"{'='*70}\n", flush=True)
        logger.info("[CONNECTION TEST] Successfully connected and tested AI server")
        return True
        
    except grpc.RpcError as e:
        print(f"[CONNECTION TEST] ✗ FAILED: gRPC Error", flush=True)
        print(f"[CONNECTION TEST] Error Code: {e.code()}", flush=True)
        print(f"[CONNECTION TEST] Error Details: {e.details()}", flush=True)
        print(f"[CONNECTION TEST] The AI server may be offline or unreachable.", flush=True)
        print(f"{'='*70}\n", flush=True)
        logger.error(f"[CONNECTION TEST] gRPC error: {e.code()} - {e.details()}")
        reset_grpc_killfeed_channel()
        return False
    except Exception as e:
        print(f"[CONNECTION TEST] ✗ FAILED: {str(e)}", flush=True)
        print(f"[CONNECTION TEST] Please check if the AI server is running and accessible.", flush=True)
        print(f"{'='*70}\n", flush=True)
        logger.error(f"[CONNECTION TEST] Connection test failed: {e}")
        reset_grpc_killfeed_channel()
        return False


def get_grpc_predict_stub():
    """
    Return a shared KillfeedStub. Thread-safe; recreates on grpc.* config change.
    """
    global _grpc_channel, _grpc_stub, _grpc_cache_key
    key = _grpc_config_cache_key()
    with _grpc_state_lock:
        if _grpc_stub is not None and _grpc_cache_key == key:
            return _grpc_stub
        if _grpc_channel is not None:
            try:
                _grpc_channel.close()
            except Exception:
                pass
            _grpc_channel = None
        grpc_config = config.get_grpc_config()
        grpc_addr = grpc_config["server"]
        use_tls = grpc_config.get("use_tls", False)
        tls_cert_path = grpc_config.get("tls_cert_path", "")
        opts = _grpc_channel_options()
        use_comp = bool(config.get("detection.grpc_use_compression", False))
        comp = None
        if use_comp:
            try:
                comp = grpc.Compression.Gzip
            except AttributeError:
                comp = None
        
        # Log connection attempt
        print(f"\n{'='*70}", flush=True)
        print(f"[gRPC CONNECTION] Attempting to connect to AI Server...", flush=True)
        print(f"[gRPC CONNECTION] Server Address: {grpc_addr}", flush=True)
        print(f"[gRPC CONNECTION] TLS Enabled: {use_tls}", flush=True)
        print(f"{'='*70}\n", flush=True)
        logger.info(f"[gRPC CONNECTION] Connecting to AI server at {grpc_addr}")
        
        if use_tls:
            if tls_cert_path and os.path.exists(tls_cert_path):
                with open(tls_cert_path, "rb") as f:
                    creds = grpc.ssl_channel_credentials(f.read())
            else:
                creds = grpc.ssl_channel_credentials()
            ch = (
                grpc.secure_channel(grpc_addr, creds, options=opts, compression=comp)
                if comp is not None
                else grpc.secure_channel(grpc_addr, creds, options=opts)
            )
        else:
            ch = (
                grpc.insecure_channel(grpc_addr, options=opts, compression=comp)
                if comp is not None
                else grpc.insecure_channel(grpc_addr, options=opts)
            )
        _grpc_channel = ch
        _grpc_stub = killfeed_pb2_grpc.KillfeedStub(ch)
        _grpc_cache_key = key
        
        # Log successful connection
        print(f"[gRPC CONNECTION] ✓ Connected to AI Server successfully!", flush=True)
        print(f"[gRPC CONNECTION] Ready to receive and process frames from this desktop.", flush=True)
        print(f"{'='*70}\n", flush=True)
        logger.info(f"[gRPC CONNECTION] Successfully created gRPC stub for {grpc_addr}")
        
        return _grpc_stub


def _append_killfeed_table_line(
    killer_name: str,
    weapon_used: str,
    victim_name: str,
    frame_number: int,
    timestamp: str,
) -> None:
    """One line per posted killfeed: player: killer + gun + victim (gRPC OCR pipeline)."""
    try:
        os.makedirs(GAME_LOGS_DIR, exist_ok=True)
        line = (
            f"{timestamp} | Frame #{frame_number} | "
            f"player: {killer_name} + {weapon_used} + {victim_name}\n"
        )
        with _killfeed_table_lock:
            with open(KILLFEED_TABLE_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line)
    except Exception as e:
        logger.debug(f"Could not write killfeed_table.log: {e}")


def save_killfeed_to_result(
    killer_name: str,
    victim_name: str,
    weapon_used: str,
    sift_weapon: str,
    frame_number: int,
    timestamp: str,
    grpc_response_data: Optional[dict[str, Any]] = None
) -> None:
    """
    Save killfeed detection result to result.txt instead of API.
    
    Args:
        killer_name: Name of the player who got the kill.
        victim_name: Name of the player who was killed.
        weapon_used: Weapon used for the kill.
        sift_weapon: Weapon detected via SIFT matching.
        frame_number: Frame number when kill was detected.
        timestamp: Timestamp of the detection.
        grpc_response_data: Optional gRPC response data.
    """
    try:
        os.makedirs(GAME_LOGS_DIR, exist_ok=True)
        with open(RESULT_FILE, 'a', encoding='utf-8') as f:
            result_line = (
                f"{timestamp} | "
                f"Frame #{frame_number} | "
                f"{killer_name} --[{weapon_used}]--> {victim_name} | "
                f"SIFT: {sift_weapon}\n"
            )
            f.write(result_line)
            
            # Add gRPC response data if provided
            if grpc_response_data:
                f.write(f"  [gRPC] Response Time: {grpc_response_data.get('time_ms', 0):.2f}ms | ")
                f.write(f"Total Detections: {grpc_response_data.get('total_detections', 0)} | ")
                f.write(f"Kill Blocks: {grpc_response_data.get('kill_blocks', 0)}\n")
                f.write(f"  [gRPC] Detections: ")
                detections_str = ", ".join([
                    f"{d['class_name']}({d['confidence']:.3f})" 
                    for d in grpc_response_data.get('detections', [])
                ])
                f.write(f"{detections_str}\n")
        _append_killfeed_table_line(
            killer_name, weapon_used, victim_name, frame_number, timestamp
        )
        logger.info(f"[RESULT] Saved: {killer_name} --[{weapon_used}]--> {victim_name}")
    except Exception as e:
        logger.error(f"[RESULT] ERROR saving: {e}")


# Check if GPU is available using PyTorch instead of OpenCV CUDA
def check_opencv_gpu():
    """Check if GPU is available for image processing acceleration using PyTorch (optional)."""
    global torch, TORCH_AVAILABLE
    try:
        # Import torch only when needed (optional dependency)
        if torch is None:
            import torch as torch_module
            torch = torch_module
            TORCH_AVAILABLE = True
        
        # Check if PyTorch can access GPU
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU acceleration available: {device_name}")
            logger.info(f"CUDA device count: {device_count}")
            torch.cuda.set_device(0)  # Use the first GPU
            return True
        else:
            logger.info("GPU acceleration not available (using CPU for SIFT)")
            return False
    except ImportError:
        logger.debug("PyTorch not installed - using CPU for processing (this is fine)")
        return False
    except Exception as e:
        logger.debug(f"GPU check skipped: {e}")
        return False


# Global flag for GPU usage
USE_GPU = check_opencv_gpu()


def get_torch_device():
    """Get PyTorch device safely (optional - returns 'cpu' if torch not available)"""
    global torch, TORCH_AVAILABLE
    # Import torch if not already imported
    if torch is None:
        try:
            import torch as torch_module
            torch = torch_module
            TORCH_AVAILABLE = True
        except ImportError:
            return "cpu"
        except Exception:
            return "cpu"
    
    if not TORCH_AVAILABLE:
        return "cpu"
    
    return torch.device("cuda" if USE_GPU else "cpu")

# Global cache for SIFT features
sift_template_cache = {}

# Global queue for recent killfeeds (with thread-safe lock)
duplicate_check_count = config.get('processing.duplicate_check_count', 5)
recent_killfeeds = deque(maxlen=duplicate_check_count)
recent_killfeeds_lock = threading.Lock()
# Set to track killfeeds currently being processed (to prevent duplicates from reaching worker pool)
processing_killfeeds = set()
processing_killfeeds_lock = threading.Lock()

# Track duplicate statistics
duplicate_stats = {
    'total_processed': 0,
    'duplicates_rejected': 0,
    'api_saves_attempted': 0,
    'api_saves_successful': 0
}

# Stage validation system
stage_validation_count = 4  # Number of consistent detections needed to establish a stage
recent_stages = deque(maxlen=stage_validation_count)  # Track recent stage detections
current_validated_stage = None  # The currently validated stage


def cache_sift_templates():
    """
    Precompute and cache SIFT keypoints and descriptors for all templates in single-template.
    """
    sift_features = config.get('detection.sift_features', 400)
    sift = cv2.SIFT_create(nfeatures=sift_features)
    weapon_folder = os.path.join(os.path.dirname(__file__), "single-template")
    
    if not os.path.exists(weapon_folder):
        return
    
    template_count = 0
    
    for subdir, dirs, files in os.walk(weapon_folder):
        for file in files:
            template_image_path = os.path.join(subdir, file)
            template_image = cv2.imread(template_image_path)
            if template_image is None:
                continue
                
            gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            image_resize_factor = config.get('processing.image_resize_factor', 0.75)
            gray_template = cv2.resize(gray_template, (0, 0), fx=image_resize_factor, fy=image_resize_factor)
            keypoints, descriptors = sift.detectAndCompute(gray_template, None)
            
            if descriptors is not None:
                template_name = os.path.splitext(file)[0].split("-")[0]
                
                # Store as a list in case of multiple templates per weapon
                if template_name not in sift_template_cache:
                    sift_template_cache[template_name] = []
                sift_template_cache[template_name].append(
                    (descriptors, template_image_path)
                )
                template_count += 1


# Call this at startup
cache_sift_templates()


# ======= STAGE DETECTION FUNCTIONS =======
def levenshtein(a: str, b: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            add = prev[j] + 1
            delete = cur[j-1] + 1
            change = prev[j-1] + (0 if ca == cb else 1)
            cur[j] = min(add, delete, change)
        prev = cur
    return prev[lb]


# Correction map applied ONLY when candidate token has NO digits
DIGIT_CORRECTIONS = {
    "O": "0", "o": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1",
    "Z": "2",
    "S": "5", "s": "5",
    "G": "6",
    "T": "7",
    "B": "8", "b": "8",
    "g": "9", "q": "9",
    "a": ""  # drop stray 'a' prefixes
}


def correct_digits_fallback(token: str) -> str:
    """Apply digit corrections only when token contains no digits."""
    if re.search(r"\d", token):
        return token
    return "".join(DIGIT_CORRECTIONS.get(ch, ch) for ch in token)


def extract_stage_from_frame(frame):
    """
    Extract stage number from a game frame using OCR.
    
    Args:
        frame (numpy.ndarray): Input frame/image as numpy array
        
    Returns:
        int or None: Detected stage number, or None if not found
    """
    detection_logger.debug("Starting stage detection from frame")
    stage_start_time = time.time()
    
    try:
        if frame is None:
            detection_logger.warning("Frame is None for stage detection")
            return None
            
        h, w = frame.shape[:2]
        # ROI for stage detection (top-right area)
        y1 = int(h * 0.05)
        y2 = int(h * 0.35)
        x1 = int(w * 0.70)
        x2 = int(w * 1.00)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            detection_logger.warning("Empty crop for stage detection")
            return None

        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Quick preprocessing options for speed
        preprocess_list = [
            ("thresh_127", cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]),
            ("otsu_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        ]

        # Try text detection using gRPC server
        try:
            # Use the existing OCR instance if available in the calling context
            # For now, we'll use a simplified approach with cv2 text detection
            for name, proc in preprocess_list:
                # Scale up for better OCR
                scaled = cv2.resize(proc, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                
                # Simple text extraction using basic image processing
                # Look for text-like regions
                contours, _ = cv2.findContours(scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours that might contain text
                text_regions = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter by size - text should be reasonably sized
                    if 10 < w < 200 and 10 < h < 50:
                        aspect_ratio = w / h
                        if 0.5 < aspect_ratio < 8:  # Reasonable aspect ratio for text
                            text_regions.append((x, y, w, h))
                
                # If we have text regions, we found potential stage text
                if text_regions:
                    detection_logger.debug(f"Found {len(text_regions)} potential text regions for stage detection")
                    # For speed, just return a placeholder for now
                    # In a real implementation, you'd use OCR on these regions
                    # But since we want to keep it fast, we'll use pattern matching
                    
                    # Look for stage-like patterns in the image
                    # This is a simplified approach - in practice you'd use OCR here
                    stage_num = detect_stage_pattern(scaled)
                    if stage_num is not None:
                        stage_time = time.time() - stage_start_time
                        detection_logger.info(f"Stage detection successful: {stage_num} (took {stage_time:.4f}s)")
                        return stage_num
                        
        except Exception as e:
            detection_logger.error(f"Error in stage detection OCR: {e}")
            
        stage_time = time.time() - stage_start_time
        detection_logger.debug(f"Stage detection failed (took {stage_time:.4f}s)")
        return None
        
    except Exception as e:
        stage_time = time.time() - stage_start_time
        detection_logger.error(f"Exception in stage detection: {e} (took {stage_time:.4f}s)")
        return None


def detect_stage_pattern(image):
    """
    Simplified stage pattern detection for speed.
    This is a placeholder - in production you'd use proper OCR here.
    
    Args:
        image (numpy.ndarray): Preprocessed image
        
    Returns:
        int or None: Detected stage number or None
    """
    try:
        # For now, return None to keep it fast
        # In a full implementation, you would:
        # 1. Use template matching for "Stage" text
        # 2. Look for digit patterns near the "Stage" text
        # 3. Apply OCR only to small regions containing potential stage numbers
        
        # Placeholder - you can implement actual OCR here if needed
        # For maximum speed, you might want to use template matching instead
        return None
        
    except Exception as e:
        detection_logger.error(f"Error in stage pattern detection: {e}")
        return None


# DEPRECATED: Stage detection is now disabled since OCR comes from gRPC server
def extract_stage_with_ocr(frame, ocr, match_name, frame_counter):
    """
    DEPRECATED: Stage detection via separate OCR is no longer used.
    OCR is now handled by the gRPC server.
    
    Extract stage number using OCR (fast version without logging/saving).
    
    Args:
        frame (numpy.ndarray): Input frame
        ocr: OCR instance (unused - handled by gRPC server)
        match_name (str): Match name for folder organization (unused in fast version)
        frame_counter (int): Frame counter for unique naming (unused in fast version)
        
    Returns:
        int or None: Detected stage number or None
    """
    # Stage detection is disabled - return None
    return None


# ======= END STAGE DETECTION FUNCTIONS =======


def validate_stage_detection(detected_stage):
    """
    Validate stage detection to prevent backwards progression.
    
    Args:
        detected_stage (int or None): The stage number detected in current frame
        
    Returns:
        int or None: Validated stage number or None if invalid
    """
    global current_validated_stage, recent_stages
    
    # If no stage detected, return None
    if detected_stage is None:
        return None
    
    # Add current detection to recent stages
    recent_stages.append(detected_stage)
    
    # If we don't have enough samples yet, return the detected stage
    if len(recent_stages) < stage_validation_count:
        return detected_stage
    
    # Check if we have consistent stage detections
    unique_stages = set(recent_stages)
    
    # If all recent detections are the same stage, validate it
    if len(unique_stages) == 1:
        consistent_stage = list(unique_stages)[0]
        
        # If this is a new validated stage
        if current_validated_stage is None:
            current_validated_stage = consistent_stage
            return consistent_stage
        
        # If this is the same as current validated stage, return it
        if consistent_stage == current_validated_stage:
            return consistent_stage
        
        # If this is a higher stage (progression), update and return it
        if consistent_stage > current_validated_stage:
            current_validated_stage = consistent_stage
            return consistent_stage
        
        # If this is a lower stage (regression), return None (invalid)
        if consistent_stage < current_validated_stage:
            return None
    
    # If stages are inconsistent, return None
    return None


# Log killfeed ROI mode once (avoid per-frame noise)
_killfeed_roi_mode_logged: bool = False


def _killfeed_grpc_image_and_map_scales(
    frame: "np.ndarray",
    resized_w: int,
    resized_h: int,
) -> tuple["np.ndarray", int, int, int, int, float, float]:
    """
    Build the BGR image to resize+JPEG for gRPC, and the scale factors to map
    gRPC bboxes (in resized_w x resized_h space) to full frame coordinates.

    When detection.killfeed_roi_enabled is true and killfeed_roi is set, only that
    rectangle is sent to the model (faster, higher resolution on the strip).

    Returns:
        bgr_in: image to pass to cv2.resize(..., (resized_w, resized_h))
        roi_ox, roi_oy, roi_w, roi_h: ROI in full-frame pixels (offset + size)
        scale_x, scale_y: multipliers from resized coords to *full* frame
          (i.e. roi_w/resized_w and roi_h/resized_h when using ROI, else orig_w/resized_w)
    """
    global _killfeed_roi_mode_logged
    orig_h, orig_w = int(frame.shape[0]), int(frame.shape[1])
    if orig_w < 1 or orig_h < 1:
        raise ValueError("empty frame")

    use_roi = bool(config.get("detection.killfeed_roi_enabled", False))
    roi_spec = config.get("detection.killfeed_roi", None)
    min_side = int(config.get("detection.killfeed_roi_min_size_px", 32))

    if not use_roi or not roi_spec:
        bgr_in = frame
        if use_roi and not roi_spec and not _killfeed_roi_mode_logged:
            logger.warning(
                "detection.killfeed_roi_enabled is true but killfeed_roi is missing; using full frame"
            )
            _killfeed_roi_mode_logged = True
        scales_x = orig_w / float(resized_w)
        scales_y = orig_h / float(resized_h)
        return bgr_in, 0, 0, orig_w, orig_h, scales_x, scales_y

    if not isinstance(roi_spec, (list, tuple)) or len(roi_spec) != 4:
        logger.warning("detection.killfeed_roi must be [x1, y1, x2, y2] in 0..1; using full frame")
        bgr_in = frame
        return bgr_in, 0, 0, orig_w, orig_h, orig_w / float(resized_w), orig_h / float(resized_h)

    try:
        nx1, ny1, nx2, ny2 = (float(roi_spec[i]) for i in range(4))
    except (TypeError, ValueError, IndexError):
        logger.warning("detection.killfeed_roi values invalid; using full frame")
        return frame, 0, 0, orig_w, orig_h, orig_w / float(resized_w), orig_h / float(resized_h)

    x1 = int(max(0.0, min(1.0, nx1)) * orig_w)
    y1 = int(max(0.0, min(1.0, ny1)) * orig_h)
    x2 = int(max(0.0, min(1.0, nx2)) * orig_w)
    y2 = int(max(0.0, min(1.0, ny2)) * orig_h)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(orig_w - 1, x1))
    y1 = max(0, min(orig_h - 1, y1))
    x2 = max(x1 + 1, min(orig_w, x2))
    y2 = max(y1 + 1, min(orig_h, y2))

    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w < min_side or roi_h < min_side:
        logger.warning(
            f"detection.killfeed_roi too small ({roi_w}x{roi_h} px < min {min_side}); using full frame"
        )
        return frame, 0, 0, orig_w, orig_h, orig_w / float(resized_w), orig_h / float(resized_h)

    bgr_in = frame[y1:y2, x1:x2]
    if bgr_in.size == 0:
        logger.warning("detection.killfeed ROI crop is empty; using full frame")
        return frame, 0, 0, orig_w, orig_h, orig_w / float(resized_w), orig_h / float(resized_h)

    scales_x = roi_w / float(resized_w)
    scales_y = roi_h / float(resized_h)
    if not _killfeed_roi_mode_logged:
        logger.info(
            f"Killfeed gRPC ROI crop: offset=({x1}, {y1}) size={roi_w}x{roi_h} on {orig_w}x{orig_h} frame"
        )
        _killfeed_roi_mode_logged = True
    return bgr_in, x1, y1, roi_w, roi_h, scales_x, scales_y


def process_frame(
    frame,
    ocr,
    killfeed_detector,
    TEAM_ROSTERS,
    Match_name,
    match_id,
    access_token,
    USE_GPU,
    device,
    frame_counter,
    position,
    return_killfeed_data=False,
):
    confidence_threshold = config.get('detection.confidence_threshold', 0.20)

    # Create game_logs folder structure
    game_logs_folder = GAME_LOGS_DIR
    match_folder = os.path.join(game_logs_folder, Match_name)
    kill_block_folder = os.path.join(match_folder, "kill_block")
    os.makedirs(kill_block_folder, exist_ok=True)

    try:
        # --- gRPC YOLO PREDICTION ---
        # Full-res geometry from BGR frame; avoid PIL for resize/encode (faster).
        # Optional ROI: only the killfeed strip is resized+sent (see detection.killfeed_roi_*).
        orig_h, orig_w = frame.shape[0], frame.shape[1]
        resized_w = int(config.get("detection.grpc_input_width", 1280))
        resized_h = int(config.get("detection.grpc_input_height", 720))
        resized_w = max(320, min(4096, resized_w))
        resized_h = max(240, min(4096, resized_h))
        bgr_in, _roi_ox, _roi_oy, _roi_w, _roi_h, scale_x, scale_y = _killfeed_grpc_image_and_map_scales(
            frame, resized_w, resized_h
        )
        resized_bgr = cv2.resize(
            bgr_in, (resized_w, resized_h), interpolation=cv2.INTER_AREA
        )
        jpg_q = int(config.get("detection.grpc_jpeg_quality", 90))
        jpg_q = max(70, min(100, jpg_q))
        enc_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            jpg_q,
            int(cv2.IMWRITE_JPEG_OPTIMIZE),
            1,
        ]
        ok_enc, enc_buf = cv2.imencode(".jpg", resized_bgr, enc_params)
        if not ok_enc:
            logger.error("[gRPC] JPEG encode failed for killfeed frame")
            return [] if return_killfeed_data else position
        image_bytes = enc_buf.tobytes()

        stub = get_grpc_predict_stub()
        grpc_config = config.get_grpc_config()
        api_key = grpc_config["api_key"]
        detection_config = config.get_detection_config()
        game = detection_config.get("game", "") or ""
        request_kwargs = {"image": image_bytes}
        conf_override = detection_config.get("grpc_conf")
        if conf_override is None and config.get("detection.grpc_send_confidence_threshold", False):
            conf_override = detection_config.get("confidence_threshold", confidence_threshold)
        if conf_override is not None:
            try:
                request_kwargs["conf"] = float(conf_override)
            except Exception:
                pass
        min_kb_conf = detection_config.get("min_killblock_conf")
        if min_kb_conf is not None:
            try:
                request_kwargs["min_killblock_conf"] = float(min_kb_conf)
            except Exception:
                pass
        if game:
            # Keep this read for compatibility with existing game config flow/logging.
            _ = get_game_config(game)
        request = killfeed_pb2.InferRequest(**request_kwargs)
        metadata = (("api-key", api_key),) if api_key else ()
        grpc_start_time = time.time()
        
        # Log that frame is being sent
        print(f"[FRAME #{frame_counter:06d}] Capturing and sending frame to AI Server...", flush=True)
        logger.debug(f"[FRAME SEND] Frame #{frame_counter} - Image: {resized_w}x{resized_h}, JPEG: {len(image_bytes)} bytes")
        # Always overwrite the last-sent sample file so user can verify the exact image being
        # transmitted. This runs for every frame send (success or failure).
        try:
            sample_dir = os.path.join(GAME_LOGS_DIR, 'sample_frames')
            os.makedirs(sample_dir, exist_ok=True)
            sample_path = os.path.join(sample_dir, "last_sent_frame.jpg")
            cv2.imwrite(sample_path, resized_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])
            sha1 = hashlib.sha1(image_bytes).hexdigest() if image_bytes else ""
            print(f"[FRAME #{frame_counter:06d}] 🔁 WROTE last_sent_frame: {sample_path} (bytes={len(image_bytes) if image_bytes else 0}, sha1={sha1})", flush=True)
        except Exception as e:
            print(f"[FRAME #{frame_counter:06d}] 🔍 Failed to write last_sent_frame: {e}", flush=True)

        try:
            response = stub.Infer(request, metadata=metadata)
            grpc_time = time.time() - grpc_start_time
            response_rows = list(getattr(response, "rows", []) or [])
            
            # Log successful reception from server
            print(f"[FRAME #{frame_counter:06d}] ✓ Frame received and processed by AI Server (took {grpc_time:.3f}s)", flush=True)
            logger.info(f"[FRAME RECEIVED] Frame #{frame_counter} processed successfully in {grpc_time:.3f}s")

            # Print raw server response summary for every frame
            detection_count = len(response_rows)
            ocr_count = 0
            for r in response_rows:
                if (getattr(r, "killer", "") or "").strip():
                    ocr_count += 1
                if (getattr(r, "victim", "") or "").strip():
                    ocr_count += 1
            print(
                f"[FRAME #{frame_counter:06d}] 📡 RAW SERVER RESPONSE: "
                f"detections={detection_count}, ocr_items={ocr_count}, ocr_time_ms={getattr(response, 'rows_pipeline_ms', 0):.2f}",
                flush=True,
            )

            # Print detailed detection results to terminal
            if response_rows:
                print(f"[FRAME #{frame_counter:06d}] 📊 DETECTIONS RECEIVED: {detection_count} object(s) detected", flush=True)
                for idx, det in enumerate(response_rows):
                    ocr_text = _combined_text_from_row(det)
                    bbox = list(getattr(det, "bbox_xyxy", []) or [])
                    if len(bbox) != 4:
                        bbox = [0, 0, 0, 0]
                    ocr_summary = f" OCR: '{ocr_text}'" if ocr_text else " (no OCR text)"
                    ui_event = _event_type_to_ui_label(det.event_type)
                    det_class = (getattr(det, "detection_class", "") or "").strip()
                    print(
                        f"  [{idx+1}] {ui_event}/{det_class or _event_type_to_class_name(det.event_type)} "
                        f"(conf: {det.confidence:.3f}) "
                        f"bbox:[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"
                        f"{ocr_summary}",
                        flush=True,
                    )
            else:
                print(f"[FRAME #{frame_counter:06d}] 📊 No detections in this frame", flush=True)
                # Save a sample of the image we sent for offline inspection (helpful for debugging)
                try:
                    sample_dir = os.path.join(GAME_LOGS_DIR, 'sample_frames')
                    os.makedirs(sample_dir, exist_ok=True)
                    sample_path = os.path.join(sample_dir, "last_sent_frame.jpg")
                    # Save the resized image that was sent to the server
                    cv2.imwrite(sample_path, resized_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])
                    sha1 = hashlib.sha1(image_bytes).hexdigest() if image_bytes else ""
                    print(f"[FRAME #{frame_counter:06d}] 🔍 SAVED SAMPLE FRAME: {sample_path} (bytes={len(image_bytes) if image_bytes else 0}, sha1={sha1})", flush=True)
                except Exception as e:
                    print(f"[FRAME #{frame_counter:06d}] 🔍 Failed to save sample frame: {e}", flush=True)
            
        except Exception as e:
            print(f"[FRAME #{frame_counter:06d}] ✗ ERROR sending to AI Server: {e}", flush=True)
            logger.error(f"[gRPC] Error calling killfeed detection service: {e}")
            reset_grpc_killfeed_channel()
            return [] if return_killfeed_data else position

        # Optional: verbose OCR table (default off — printing is slow on hot paths)
        if response_rows and config.get("detection.log_grpc_ocr_table", False):
            rows = []
            for d in response_rows:
                ocr_cnt = int(bool((getattr(d, "killer", "") or "").strip())) + int(bool((getattr(d, "victim", "") or "").strip()))
                combined = _combined_text_from_row(d)
                rows.append((str(frame_counter), _event_type_to_ui_label(d.event_type), str(ocr_cnt), combined[:50]))
            col_widths = [10, 14, 12, 50]
            sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
            header = "| " + "Frame".ljust(col_widths[0]) + " | " + "Class".ljust(col_widths[1]) + " | " + "OCR Texts".ljust(col_widths[2]) + " | " + "Combined".ljust(col_widths[3]) + " |"
            print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | [gRPC] OCR per detection", flush=True)
            print(sep, flush=True)
            print(header, flush=True)
            print(sep, flush=True)
            for r in rows:
                row = "| " + r[0].ljust(col_widths[0]) + " | " + r[1].ljust(col_widths[1]) + " | " + r[2].ljust(col_widths[2]) + " | " + (r[3][:col_widths[3]]).ljust(col_widths[3]) + " |"
                print(row, flush=True)
            print(sep, flush=True)
        
        # Map bboxes from gRPC resized space to full frame (scale_x/scale_y from ROI or full image)
        detections = []
        for det in response_rows:
            bbox = list(getattr(det, "bbox_xyxy", []) or [])  # [x1, y1, x2, y2] in resized space
            if len(bbox) != 4:
                continue
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            # Clamp to image bounds
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
            # Crop from full-res BGR frame (no PIL; matches previous geometry)
            if y2 <= y1 or x2 <= x1:
                continue
            crop_np = frame[y1:y2, x1:x2].copy()
            if crop_np.size == 0:
                continue
            
            # Adapt Infer row names to existing OCR text-box flow.
            ocr_text_boxes = []
            killer_txt = (getattr(det, "killer", "") or "").strip()
            victim_txt = (getattr(det, "victim", "") or "").strip()
            if killer_txt:
                ocr_text_boxes.append((x1, y1, x2, y2, killer_txt, float(det.confidence)))
            if victim_txt:
                ocr_text_boxes.append((x1, y1, x2, y2, victim_txt, float(det.confidence)))
            
            detections.append({
                "class_id": int(det.event_type),
                "class_name": (getattr(det, "detection_class", "") or "").strip() or _event_type_to_class_name(det.event_type),
                "confidence": det.confidence,
                "bbox": [x1, y1, x2, y2],
                "crop": crop_np,
                "combined_text": _combined_text_from_row(det),  # Combined OCR-like text
                "ocr_text_boxes": ocr_text_boxes,    # Individual OCR detections
            })

        # Now, process detections as before, but use the new detections list
        kill_blocks = []
        other_objects = []
        for i, detection in enumerate(detections):
            confidence = detection.get("confidence", 0)
            class_id = detection.get("class_id")
            class_name = detection.get("class_name")
            combined_text = (detection.get("combined_text", "") or "").strip()
            ocr_text_boxes = detection.get("ocr_text_boxes", []) or []
            bbox = detection.get("bbox")
            if confidence >= confidence_threshold:
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                    continue
                detection_data = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                }
                killfeed_detector.add_detection(detection_data)
                # Game-specific killfeed detection (BGMI vs FreeFire)
                looks_like_killfeed_text = bool(combined_text) or bool(ocr_text_boxes)
                if is_killfeed_detection(class_id, class_name, game) or looks_like_killfeed_text:
                    # Include OCR data, class_name, and event_type enum from gRPC
                    kill_blocks.append((x1, y1, x2, y2, detection["crop"], ocr_text_boxes, combined_text, class_name or "", class_id))
                else:
                    other_objects.append((x1, y1, x2, y2, class_name))

        kill_blocks.sort(key=lambda x: x[1])

        if not kill_blocks:
            if response_rows:
                print(f"[FRAME #{frame_counter:06d}] 🔍 Analysis: {len(response_rows)} detections received, but 0 matched killfeed pattern (game={game})", flush=True)
            if return_killfeed_data:
                return []
            return position
        
        # Print killfeed detection summary
        print(f"[FRAME #{frame_counter:06d}] 🎯 KILLFEED FOUND: {len(kill_blocks)} kill block(s) detected", flush=True)
        for kb_idx, kb in enumerate(kill_blocks):
            x1, y1, x2, y2, crop, ocr_boxes, combined_text, class_name, event_type_id = kb
            event_type_str = _event_type_to_class_name(event_type_id)
            print(f"  [{kb_idx+1}] Kill Block at [{x1},{y1},{x2},{y2}] event:{event_type_str} OCR:'{combined_text}'", flush=True)

        # Store gRPC response data for logging
        grpc_response_data = {
            'time_ms': grpc_time * 1000,
            'ocr_time_ms': getattr(response, 'rows_pipeline_ms', 0.0),
            'total_detections': len(response_rows),
            'kill_blocks': len(kill_blocks),
            'other_objects': len(other_objects),
            'detections': [
                {
                    'class_name': (getattr(det, "detection_class", "") or "").strip() or _event_type_to_class_name(det.event_type),
                    'event_label': _event_type_to_ui_label(det.event_type),
                    'class_id': int(det.event_type),
                    'confidence': det.confidence,
                    'bbox': list(getattr(det, 'bbox_xyxy', []) or []),
                    'combined_text': _combined_text_from_row(det),
                    'ocr_texts': [
                        {'text': (getattr(det, "killer", "") or "").strip(), 'confidence': float(det.confidence)}
                    ] + (
                        [{'text': (getattr(det, "victim", "") or "").strip(), 'confidence': float(det.confidence)}]
                        if (getattr(det, "victim", "") or "").strip() else []
                    )
                }
                for det in response_rows
            ]
        }
        
        # Log detailed gRPC response when killfeed is detected
        logger.info(f"[gRPC] KILLFEED DETECTED - Frame #{frame_counter}")
        logger.info(f"[gRPC] Response: {grpc_response_data['time_ms']:.2f}ms, "
                   f"OCR: {grpc_response_data['ocr_time_ms']:.2f}ms, "
                   f"Detections: {grpc_response_data['total_detections']}, "
                   f"Kill Blocks: {grpc_response_data['kill_blocks']}")
        
        for idx, det_info in enumerate(grpc_response_data['detections']):
            bbox = det_info['bbox']
            logger.debug(f"  [{idx}] {det_info['class_name']} (conf: {det_info['confidence']:.4f}) "
                        f"bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            if det_info['combined_text']:
                logger.debug(f"       OCR: '{det_info['combined_text']}'")

        # === OCR FROM gRPC (no separate OCR call needed) ===
        # OCR is now done by the gRPC server and included in detections
        # Stage detection is disabled since OCR is from gRPC
        detected_stage = None
        
        # Validate stage detection to prevent backwards progression
        validated_stage = validate_stage_detection(detected_stage)

        # Initialize killfeed list if returning data
        killfeed_list = [] if return_killfeed_data else None

        # Process each kill block
        for kb_index, kb_data in enumerate(kill_blocks):
            x1_kb, y1_kb, x2_kb, y2_kb, kill_block_crop, ocr_text_boxes, combined_text = (
                kb_data[0], kb_data[1], kb_data[2], kb_data[3], kb_data[4], kb_data[5], kb_data[6]
            )
            # Get the actual event_type enum from gRPC response
            event_type_id = kb_data[8] if len(kb_data) > 8 else int(killfeed_pb2.EVENT_TYPE_UNSPECIFIED)
            # Convert event_type enum to string (kill, gun-knockout, revive, etc.)
            event_type_str = _event_type_to_class_name(event_type_id)
            try:
                if y2_kb > frame.shape[0] or x2_kb > frame.shape[1]:
                    continue
                if kill_block_crop is None or kill_block_crop.size == 0:
                    continue
                kb_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # Save kill block image in colored format (not grayscale)
                
                # OCR text from gRPC response (server handles OCR now)
                detected_text_boxes = ocr_text_boxes
                logger.debug(f"[gRPC OCR] Kill Block {kb_index}: combined_text='{combined_text}', text_boxes={len(detected_text_boxes)}")
                
                # Save kill block image
                kill_block_path = os.path.join(
                    kill_block_folder,
                    f"frame_{frame_counter}_{kb_index}_{kb_timestamp}_kill_block.jpg",
                )
                cv2.imwrite(
                    kill_block_path,
                    kill_block_crop,
                    [cv2.IMWRITE_JPEG_QUALITY, config.get('processing.image_quality', 85)],
                )
                
                if not detected_text_boxes:
                    detected_text_boxes = []
                else:
                    detected_text_boxes.sort(key=lambda box: box[0])
                # Extract killer/victim names only (game-specific parsing)
                killer_name, victim_name, _ = extract_kill_info(
                    detected_text_boxes,
                    other_objects,
                    x1_kb,
                    y1_kb,
                    x2_kb,
                    y2_kb,
                    combined_text=combined_text or "",
                    game=game,
                )
                killer_name, victim_name = _fix_identical_killer_victim(
                    killer_name,
                    victim_name,
                    combined_text or "",
                    detected_text_boxes,
                    event_type_str,
                )
                # Use raw gRPC event type directly - no processing or modification
                # Send mapped event labels: kill, revive, gun-knockout, etc.
                weapon_used = event_type_str
                weapon_for_api = _normalize_weapon_for_tms_api(weapon_used)

                # SIFT verification
                sift_result, sift_weapon_name = verify_weapon_with_sift(
                    kill_block_folder, weapon_used
                )
                # Get player teams
                killer_name = get_player_team(killer_name, TEAM_ROSTERS)
                victim_name = get_player_team(victim_name, TEAM_ROSTERS)
                killer_name, victim_name = _fix_identical_killer_victim(
                    killer_name,
                    victim_name,
                    combined_text or "",
                    detected_text_boxes,
                    weapon_used,
                )
                if _is_invalid_same_killer_victim(killer_name, victim_name, weapon_used):
                    try:
                        os.remove(kill_block_path)
                    except Exception:
                        pass
                    continue

                # Don't post when victim is unknown; killer may still be unknown (e.g. single OCR line)
                def _is_unknown(s):
                    return not s or str(s).strip().lower() == "unknown"
                if _is_unknown(victim_name):
                    logger.debug(f"Skipping kill block - no victim name (killer={killer_name!r}, victim={victim_name!r})")
                    try:
                        os.remove(kill_block_path)
                    except Exception:
                        pass
                    continue

                # If return_killfeed_data is True, collect data for worker pool
                # (duplicate check deferred to process_killfeed_worker to avoid double-gating)
                if return_killfeed_data:
                    killfeed_data = {
                        'frame_number': frame_counter,
                        'killer_name': killer_name,
                        'weapon_used': weapon_for_api,
                        'victim_name': victim_name,
                        'kill_block_path': kill_block_path,
                        'sift_result': sift_result,
                        'sift_weapon_name': sift_weapon_name,
                        'frame_timestamp': kb_timestamp,
                        'detected_stage': detected_stage,
                        'match_name': Match_name,
                        'match_id': match_id,
                        'access_token': access_token,
                        'position': position,
                    }
                    killfeed_list.append(killfeed_data)
                    # Print extracted kill info to terminal
                    print(f"[FRAME #{frame_counter:06d}] 💥 EXTRACTED: '{killer_name}' --[{weapon_for_api}]--> '{victim_name}'", flush=True)
                else:
                    # Sequential path: check duplicates here before posting
                    if _is_duplicate_killfeed(killer_name, victim_name, weapon_for_api):
                        duplicate_stats['duplicates_rejected'] += 1
                        logger.debug(f"Skipping duplicate killfeed: {killer_name} -> {victim_name} ({weapon_for_api})")
                        try:
                            os.remove(kill_block_path)
                        except Exception:
                            pass
                        continue

                    # Save to result.txt (local log)
                    save_killfeed_to_result(
                        killer_name=killer_name,
                        victim_name=victim_name,
                        weapon_used=weapon_for_api,
                        sift_weapon=sift_weapon_name,
                        frame_number=frame_counter,
                        timestamp=kb_timestamp,
                        grpc_response_data=grpc_response_data,
                    )
                    # Post killfeed to API via queue (weaponUsed = actual weapon from pipeline, not detection class)
                    killfeed_data = {
                        'frame_number': frame_counter,
                        'killer_name': killer_name,
                        'weapon_used': weapon_for_api,
                        'victim_name': victim_name,
                        'kill_block_path': kill_block_path,
                        'sift_result': sift_result,
                        'sift_weapon_name': sift_weapon_name,
                        'frame_timestamp': kb_timestamp,
                        'detected_stage': detected_stage,
                        'match_name': Match_name,
                        'match_id': match_id,
                        'access_token': access_token,
                        'position': position,
                    }
                    _submit_api_call_async(killfeed_data)
                    position += 1
            except Exception:
                continue
    except Exception:
        if return_killfeed_data:
            return []
        return position
    
    if return_killfeed_data:
        return killfeed_list
    
    return position


def frame_capture_worker(camera_index, frame_queue, stop_flag, frame_capture_interval):
    """
    Worker thread that continuously captures frames from camera and puts them in queue.
    ZERO FRAME DROPS - uses blocking queue operations to ensure all frames are captured.
    Supports alternate frame processing to improve performance.

    Args:
        camera_index (int): Camera index to capture from
        frame_queue (Queue): Queue to put captured frames
        stop_flag (threading.Event): Event to signal stopping
        frame_capture_interval (float): Interval between frame captures
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Could not open camera with index {camera_index}.")
        return

    camera_config = config.get_camera_config()
    
    # Try to set camera properties (may fail for virtual cameras like OBS)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['default_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['default_height'])
        # Set buffer size to 1 to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error as e:
        logger.warning(f"Could not set camera properties (this is normal for virtual cameras): {e}")
    except Exception as e:
        logger.warning(f"Could not set camera properties: {e}")
    
    last_capture_time = time.time()
    frame_counter = 0
    dropped_frames = 0
    processed_frame_counter = 0  # Counter for frames actually processed
    
    # Get alternate frame processing setting
    process_alternate_frames = config.get('detection.process_alternate_frames', False)
    # When gRPC lags, prefer fresh frames over backlog (reduces stale killfeed misses)
    drop_oldest_when_full = bool(
        config.get('detection.frame_queue_drop_oldest_when_full', True)
    )

    logger.info(f"Frame capture thread started for camera {camera_index}")
    if process_alternate_frames:
        logger.info("Alternate frame processing enabled - processing every other frame")
    if drop_oldest_when_full:
        logger.info(
            "Frame queue mode: drop-oldest when full (lower latency for fast killfeeds)"
        )
    
    while not stop_flag.is_set():
        try:
            current_time = time.time()
            time_since_last_capture = current_time - last_capture_time
            
            if time_since_last_capture >= frame_capture_interval:
                last_capture_time = current_time
                
                ret, frame = cap.read()
                
                if ret:
                    # Always increment frame counter for tracking
                    frame_counter += 1
                    
                    # Check if we should process this frame (alternate frame logic)
                    should_process_frame = True
                    if process_alternate_frames:
                        # Process only even-numbered frames (0, 2, 4, 6, ...)
                        # This effectively processes every other frame
                        should_process_frame = (frame_counter % 2 == 0)
                    
                    if should_process_frame:
                        # Put frame in queue with metadata
                        frame_data = {
                            'frame': frame.copy(),  # Make a copy to avoid memory issues
                            'timestamp': current_time,
                            'frame_number': processed_frame_counter  # Use processed counter for frame numbering
                        }
                        if drop_oldest_when_full:
                            try:
                                frame_queue.put_nowait(frame_data)
                            except Full:
                                try:
                                    frame_queue.get_nowait()
                                    dropped_frames += 1
                                except Empty:
                                    pass
                                try:
                                    frame_queue.put_nowait(frame_data)
                                except Full:
                                    dropped_frames += 1
                            processed_frame_counter += 1
                            if processed_frame_counter % 200 == 0 and dropped_frames:
                                logger.info(
                                    f"Capture: {processed_frame_counter} frames enqueued, "
                                    f"oldest dropped: {dropped_frames} (keeps feed fresh under gRPC load)"
                                )
                        else:
                            # Blocking put: no intentional drop, but capture may stall if queue is full
                            try:
                                frame_queue.put(frame_data, block=True, timeout=1.0)
                                processed_frame_counter += 1
                                if processed_frame_counter % 100 == 0:
                                    logger.info(
                                        f"Processed {processed_frame_counter} frames "
                                        f"(total: {frame_counter}), dropped: {dropped_frames}"
                                    )
                            except Exception as e:
                                logger.critical(
                                    f"Cannot put frame in queue after timeout: {e} "
                                    f"(queue {frame_queue.qsize()}/{frame_queue.maxsize})"
                                )
                                dropped_frames += 1
                                processed_frame_counter += 1
                    else:
                        # Frame was skipped due to alternate frame processing
                        # No need to do anything, just continue
                        pass
                else:
                    time.sleep(0.01)  # Brief sleep if no frame available
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                
        except Exception as e:
            logger.error(f"Error in frame capture worker: {e}")
            time.sleep(0.1)
    
    logger.info(f"Frame capture thread stopping... Processed: {processed_frame_counter}, Total: {frame_counter}, Dropped: {dropped_frames}")
    cap.release()


# Global queue and worker thread for API posting (completely separate from main processing)
_api_queue = None
_api_worker_thread = None
_api_stop_flag = None
_api_queue_lock = threading.Lock()


def _api_queue_worker(stop_flag):
    """
    Dedicated worker thread that processes API calls from the queue.
    Completely independent from main processing - doesn't affect it at all.
    """
    logger.info("API queue worker thread started")
    print("[API Worker] Thread started - will process queued killfeeds (base64 + POST)", flush=True)
    api_call_count = 0

    while not stop_flag.is_set():
        try:
            # Get API call from queue with timeout
            try:
                killfeed_data = _api_queue.get(timeout=1.0)
            except Empty:
                continue  # Timeout, check stop flag and continue

            api_call_count += 1
            frame_number = killfeed_data.get('frame_number', 'unknown')
            print(f"[API Worker] Got item #{api_call_count} frame #{frame_number} -> calling POST (base64 + HTTP)", flush=True)
            logger.info(f"[API Queue] Processing #{api_call_count} frame {frame_number}")

            # Process the API call
            try:
                # Run the async API call (pass timing fields for killfeed_timing_report)
                asyncio.run(
                    save_to_mongodb_async(
                        killfeed_data['killer_name'],
                        killfeed_data['weapon_used'],
                        killfeed_data['victim_name'],
                        killfeed_data['kill_block_path'],
                        (killfeed_data['sift_result'], killfeed_data['sift_weapon_name']),
                        killfeed_data['position'],
                        killfeed_data['match_id'],
                        killfeed_data['access_token'],
                        killfeed_data['frame_timestamp'],
                        killfeed_data['detected_stage'],
                        killfeed_data['match_name'],
                        processing_start_time=killfeed_data.get('processing_start_time'),
                        processing_ready_time=killfeed_data.get('processing_ready_time'),
                        frame_number=killfeed_data.get('frame_number'),
                    )
                )
                logger.debug(f"[API Queue] Completed API call #{api_call_count} for frame {frame_number}")
            except Exception as e:
                logger.error(f"[API Queue] Error processing API call for frame {frame_number}: {e}")
                print(f"[API Worker] ERROR posting frame #{frame_number}: {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                # Mark task as done
                _api_queue.task_done()

        except Exception as e:
            logger.error(f"[API Queue] Error in API queue worker: {e}")
            print(f"[API Worker] Outer error: {e}", flush=True)
            time.sleep(0.1)

    logger.info(f"API queue worker thread stopping. Processed {api_call_count} API calls.")
    print(f"[API Worker] Stopping. Processed {api_call_count} API calls.", flush=True)


def _init_api_queue():
    """Initialize the API queue and worker thread."""
    global _api_queue, _api_worker_thread, _api_stop_flag

    with _api_queue_lock:
        if _api_queue is None:
            # Create queue for API calls (unbounded - API calls won't block main processing)
            _api_queue = Queue()
            _api_stop_flag = threading.Event()

            # Start dedicated worker thread for API calls
            _api_worker_thread = threading.Thread(
                target=_api_queue_worker,
                args=(_api_stop_flag,),
                daemon=True,
                name="api_queue_worker"
            )
            _api_worker_thread.start()
            logger.info("API queue and worker thread initialized")
            print("[API Queue] Initialized - worker thread is running and will POST killfeeds with base64", flush=True)


def _ensure_api_queue_ready():
    """
    Ensure the API queue has a live worker. If the worker from a previous run is dead,
    reset state and re-init so new capture runs actually post killfeeds.
    """
    global _api_queue, _api_worker_thread, _api_stop_flag
    with _api_queue_lock:
        if _api_worker_thread is not None and not _api_worker_thread.is_alive():
            # Previous run stopped; worker is dead. Reset so _init_api_queue starts a fresh worker.
            _api_queue = None
            _api_worker_thread = None
            _api_stop_flag = None
            logger.info("API queue worker was dead - re-initializing for new capture run")
            print("[API Queue] Previous worker stopped - re-initializing for this run", flush=True)
    _init_api_queue()


def _submit_api_call_async(killfeed_data):
    """
    Submit API call to the separate API queue (non-blocking, fire-and-forget).
    This doesn't affect main processing at all - just puts it in a queue.
    """
    global _api_queue, _api_worker_thread

    # If queue exists but worker is dead (e.g. previous run stopped), re-init so items are consumed
    with _api_queue_lock:
        if _api_queue is not None and (_api_worker_thread is None or not _api_worker_thread.is_alive()):
            _api_queue = None
    if _api_queue is None:
        _init_api_queue()

    # Put in queue (non-blocking, returns immediately)
    try:
        _api_queue.put_nowait(killfeed_data)
        print(f"[DEBUG] API queue: submitted killfeed (killer={killfeed_data.get('killer_name')} -> victim={killfeed_data.get('victim_name')})")
    except Exception as e:
        # If queue is full (shouldn't happen with unbounded queue), log and continue
        logger.warning(f"[API Queue] Could not queue API call: {e}")


def _is_duplicate_killfeed(killer_name, victim_name, weapon_used):
    """
    Check if a killfeed is a duplicate based on killer, victim, and weapon.
    Thread-safe check that happens BEFORE worker pool submission.
    Uses a separate 'processing' set to prevent duplicates from reaching worker pool.
    
    Args:
        killer_name (str): Name of the killer
        victim_name (str): Name of the victim
        weapon_used (str): Weapon used
        
    Returns:
        bool: True if duplicate, False if new
    """
    # Create event key (same as used in save_to_mongodb_async); normalize knock → gun-knockout
    w = _normalize_weapon_for_tms_api(weapon_used)
    event_key = f"{killer_name}|{victim_name}|{w}"
    
    with processing_killfeeds_lock:
        # Check both processing set and recent_killfeeds (thread-safe)
        with recent_killfeeds_lock:
            if event_key in recent_killfeeds or event_key in processing_killfeeds:
                return True  # Duplicate found
        
        # Add to processing set (will be moved to recent_killfeeds after API call)
        processing_killfeeds.add(event_key)
        return False  # Not a duplicate


def process_killfeed_worker(killfeed_data):
    """
    Worker function to process a single killfeed and post it to API.
    This runs in parallel with other workers. API posting is non-blocking.
    Skips when victim unknown or when duplicate.
    
    Args:
        killfeed_data (dict): Killfeed data dictionary
        
    Returns:
        tuple: (frame_number, result_position, success)
    """
    frame_number = killfeed_data['frame_number']
    killer_name = killfeed_data.get('killer_name')
    victim_name = killfeed_data.get('victim_name')
    weapon_used = _normalize_weapon_for_tms_api(killfeed_data.get('weapon_used', ''))
    killfeed_data['weapon_used'] = weapon_used

    def _is_unknown(s):
        return not s or str(s).strip().lower() == "unknown"

    if _is_unknown(victim_name):
        logger.debug(f"Skipping killfeed (no victim): frame #{frame_number} killer={killer_name!r} victim={victim_name!r}")
        return (frame_number, killfeed_data['position'], False)

    # Don't post duplicates (no API call, no base64 upload)
    if _is_duplicate_killfeed(killer_name, victim_name, weapon_used):
        duplicate_stats['duplicates_rejected'] += 1
        logger.info(
            f"[SKIP] Duplicate killfeed - not posting (no base64): frame #{frame_number} "
            f"{killer_name} -> {victim_name} ({weapon_used})"
        )
        return (frame_number, killfeed_data['position'], False)

    try:
        # Save to result.txt immediately (fast, synchronous)
        save_killfeed_to_result(
            killer_name=killfeed_data['killer_name'],
            victim_name=killfeed_data['victim_name'],
            weapon_used=killfeed_data['weapon_used'],
            sift_weapon=killfeed_data['sift_weapon_name'],
            frame_number=frame_number,
            timestamp=killfeed_data['frame_timestamp'],
        )
        
        # Submit API call to background async worker (non-blocking, fire-and-forget)
        _submit_api_call_async(killfeed_data)
        
        # Return immediately without waiting for API call
        return (frame_number, killfeed_data['position'], True)
    except Exception as e:
        logger.error(f"Error processing killfeed for frame {frame_number}: {e}")
        import traceback
        traceback.print_exc()
        return (frame_number, killfeed_data['position'], False)


def ordered_killfeed_poster(futures_dict, futures_dict_lock, stop_flag, num_workers=5):
    """
    Thread that posts killfeeds in order by frame number.
    Monitors futures from worker pool and posts results in frame number order.
    
    Args:
        futures_dict (dict): Dictionary mapping frame_number -> list of futures
        stop_flag (threading.Event): Event to signal stopping
        num_workers (int): Number of workers (for logging)
    """
    logger.info(f"Ordered killfeed poster started (monitoring {num_workers} workers)")
    
    # Dictionary to store results by frame number
    pending_results = {}  # {frame_number: [(position, success), ...]}
    next_expected_frame = None  # Track the next frame number we expect
    completed_frames = set()  # Track which frames have all their killfeeds processed
    
    while not stop_flag.is_set() or pending_results or (futures_dict_lock and len(futures_dict) > 0):
        try:
            # Check all pending futures (thread-safe)
            with futures_dict_lock:
                frames_to_check = list(futures_dict.keys())
            
            if not frames_to_check and not pending_results:
                if stop_flag.is_set():
                    break
                time.sleep(0.01)
                continue
            
            for frame_number in frames_to_check:
                with futures_dict_lock:
                    if frame_number not in futures_dict:
                        continue
                    futures = futures_dict[frame_number]
                
                completed_count = 0
                
                for future in futures:
                    if future.done():
                        try:
                            result_frame_num, position, success = future.result()
                            if frame_number not in pending_results:
                                pending_results[frame_number] = []
                            pending_results[frame_number].append((position, success))
                            completed_count += 1
                        except Exception as e:
                            logger.error(f"Error getting result for frame {frame_number}: {e}")
                            if frame_number not in pending_results:
                                pending_results[frame_number] = []
                            pending_results[frame_number].append((None, False))
                            completed_count += 1
                
                # If all futures for this frame are done, remove from dict
                if completed_count == len(futures):
                    completed_frames.add(frame_number)
                    with futures_dict_lock:
                        if frame_number in futures_dict:
                            del futures_dict[frame_number]
            
            # Post results in order
            if next_expected_frame is None and pending_results:
                next_expected_frame = min(pending_results.keys())
            
            if next_expected_frame is not None:
                while next_expected_frame in pending_results:
                    results = pending_results.pop(next_expected_frame)
                    for position, success in results:
                        if success:
                            logger.info(f"Posted killfeed for frame #{next_expected_frame} (position: {position})")
                        else:
                            logger.error(f"Failed to post killfeed for frame #{next_expected_frame}")
                    next_expected_frame += 1
                    
                    # Find next expected frame if there's a gap
                    if next_expected_frame not in pending_results and pending_results:
                        next_expected_frame = min(pending_results.keys())
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
                
        except Exception as e:
            logger.error(f"Error in ordered killfeed poster: {e}")
            time.sleep(0.1)
    
    # Post any remaining results
    if next_expected_frame is None and pending_results:
        next_expected_frame = min(pending_results.keys())
    
    while pending_results:
        if next_expected_frame in pending_results:
            results = pending_results.pop(next_expected_frame)
            for position, success in results:
                if success:
                    logger.info(f"Posted remaining killfeed for frame #{next_expected_frame}")
            next_expected_frame += 1
        else:
            # Find next frame
            if pending_results:
                next_expected_frame = min(pending_results.keys())
            else:
                break
    
    logger.info(f"Ordered killfeed poster stopped. Last posted frame: {next_expected_frame - 1 if next_expected_frame else 'N/A'}")


def frame_processing_worker(frame_queue, ocr, killfeed_detector, TEAM_ROSTERS, Match_name, 
                          match_id, access_token, USE_GPU, device, stop_flag, 
                          worker_pool=None, futures_dict=None, futures_dict_lock=None):
    """
    Worker thread that processes frames from the queue.
    
    Args:
        frame_queue (Queue): Queue containing captured frames
        ocr: OCR instance
        killfeed_detector: Killfeed detection instance
        TEAM_ROSTERS: Team roster data
        Match_name: Name of the match
        match_id: Match ID
        access_token: API access token
        USE_GPU: GPU usage flag
        device: Torch device
        stop_flag (threading.Event): Event to signal stopping
    """
    position = 1
    processed_count = 0
    
    logger.info("Frame processing thread started")
    
    # Determine if using worker pool (for parallel processing)
    use_worker_pool = worker_pool is not None
    
    while not stop_flag.is_set():
        try:
            # Get frame from queue with timeout
            try:
                frame_data = frame_queue.get(timeout=1.0)
            except Empty:
                continue  # Timeout, check stop flag and continue
            
            frame = frame_data['frame']
            frame_number = frame_data['frame_number']
            capture_time = frame_data.get('timestamp') or time.time()

            if use_worker_pool:
                # Process frame and get killfeed data (don't post yet)
                try:
                    killfeed_list = process_frame(
                        frame,
                        ocr,
                        killfeed_detector,
                        TEAM_ROSTERS,
                        Match_name,
                        match_id,
                        access_token,
                        USE_GPU,
                        device,
                        frame_number,
                        position,
                        return_killfeed_data=True,
                    )
                    processing_ready_time = time.time()
                    # Ensure killfeed_list is always a list
                    if not isinstance(killfeed_list, list):
                        logger.warning(f"process_frame returned {type(killfeed_list)} instead of list for frame {frame_number}")
                        killfeed_list = []
                except Exception as e:
                    logger.error(f"Error processing frame {frame_number}: {e}")
                    import traceback
                    traceback.print_exc()
                    killfeed_list = []
                    processing_ready_time = time.time()
                
                # Submit ALL killfeeds to worker pool (no duplicate filtering - post everything detected)
                if killfeed_list:
                    print(f"[DEBUG] Frame #{frame_number}: {len(killfeed_list)} killfeed(s) detected, queuing for API")
                    for killfeed_data in killfeed_list:
                        killfeed_data['position'] = position
                        killfeed_data['processing_start_time'] = capture_time
                        killfeed_data['processing_ready_time'] = processing_ready_time
                        worker_pool.submit(process_killfeed_worker, killfeed_data)
                        position += 1
            else:
                # Sequential processing: process and post immediately
                position = process_frame(
                    frame,
                    ocr,
                    killfeed_detector,
                    TEAM_ROSTERS,
                    Match_name,
                    match_id,
                    access_token,
                    USE_GPU,
                    device,
                    frame_number,
                    position,
                    return_killfeed_data=False,
                )
            
            processed_count += 1
            
            # Clear the frame from memory
            del frame
            
        except Exception as e:
            print(f"Error in frame processing worker: {e}")
            time.sleep(0.1)
    
    print(f"Frame processing thread stopping. Processed {processed_count} frames.")


def obs_frame_capture(match_id=1, access_token=None, camera_index=1, stop_flag=None):
    """
    Main function to capture frames from OBS and detect kill events in game footage.
    Uses threading to separate frame capture from processing for better performance.

    Args:
        match_id (int): ID of the match being processed
        access_token (str): Authentication token for API access
        camera_index (int): Index of the camera to use for capture
        stop_flag (threading.Event): Event flag to signal stopping the capture
    """
    match_data = fetch_match_data(match_id, access_token)
    if not match_data:
        print(
            "[match] ABORT: fetch_match_data returned no data — killfeed will NOT be sent to TMS / backend.\n"
            "  Fix: same api.backend_url as your web app, valid Bearer token, and a match id that exists on that API.",
            flush=True,
        )
        return

    Match_name = match_data.get("collectionName")

    # Test connection to AI server before starting capture
    print("\n" + "="*70)
    print("INITIALIZING CONNECTION TO AI SERVER")
    print("="*70)
    connection_ok = test_grpc_connection()
    if not connection_ok:
        print("\n" + "="*70)
        print("⚠️  WARNING: Could not connect to AI Server!")
        print("="*70)
        print("\nPlease make sure:")
        print("  1. The AI server application is running on the other laptop")
        print("  2. The server IP address is correct in config.json")
        print("  3. Both computers are connected to the same network")
        print("  4. There are no firewalls blocking the connection")
        print("="*70 + "\n")
        return
    
    print("\n" + "="*70)
    print("✅ CONNECTION VERIFIED - Ready to start capturing frames!")
    print("="*70 + "\n")

    # OCR is handled by gRPC server - no local initialization needed
    ocr = None  # Placeholder for compatibility, actual OCR done via gRPC
    print("✅ OCR handled by gRPC server - no local initialization needed")
    
    killfeed_detector = KillfeedDetections(Match_name)
    game_logger = setup_logger(Match_name)
    TEAM_ROSTERS = load_team_rosters_from_api(match_id, access_token)
    TEAM_ROSTERS = TEAM_ROSTERS.get("teams", {})
    device = get_torch_device()

    # Create stop flag if not provided
    if stop_flag is None:
        stop_flag = threading.Event()

    # Configuration (defaults favor fast live matches — override in config.json if CPU/server struggles)
    frame_capture_interval = float(
        config.get('detection.frame_capture_interval', 0.12)
    )
    frame_capture_interval = max(0.05, min(3.0, frame_capture_interval))
    max_queue_size = int(config.get('detection.max_queue_size', 100))
    max_queue_size = max(5, min(500, max_queue_size))

    # Create frame queue
    frame_queue = Queue(maxsize=max_queue_size)

    # Parallel killfeed API post workers (gRPC is separate, see num_processing_threads)
    num_workers = int(config.get('detection.killfeed_api_pool_workers', 5))
    num_workers = max(1, min(32, num_workers))
    worker_pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="killfeed_worker")
    
    # Ensure API queue has a live worker (re-inits if previous run's worker stopped)
    _ensure_api_queue_ready()

    # Start frame capture thread
    capture_thread = threading.Thread(
        target=frame_capture_worker,
        args=(camera_index, frame_queue, stop_flag, frame_capture_interval),
        daemon=True
    )
    
    # Multiple threads drain the frame queue in parallel (each frame may call gRPC)
    num_processing_threads = int(
        config.get('detection.num_grpc_worker_threads', 4)
    )
    num_processing_threads = max(1, min(8, num_processing_threads))
    processing_threads = []
    for i in range(num_processing_threads):
        pt = threading.Thread(
            target=frame_processing_worker,
            args=(frame_queue, ocr, killfeed_detector, TEAM_ROSTERS, Match_name, 
                  match_id, access_token, USE_GPU, device, stop_flag,
                  worker_pool, None, None),
            daemon=True,
            name=f"grpc_worker_{i}"
        )
        processing_threads.append(pt)
    
    print(
        f"Starting capture: interval={frame_capture_interval}s, queue={max_queue_size}, "
        f"{num_processing_threads} gRPC worker thread(s), {num_workers} API pool worker(s)...",
        flush=True,
    )
    capture_thread.start()
    for pt in processing_threads:
        pt.start()
    
    try:
        # Main thread handles monitoring
        while not stop_flag.is_set():
            try:
                # Monitor queue size
                queue_size = frame_queue.qsize()
                if queue_size > max_queue_size * 0.8:  # 80% full
                    print(f"Warning: Frame queue is {queue_size}/{max_queue_size} full")
                
                time.sleep(0.1)  # Brief sleep to prevent busy waiting
                
            except KeyboardInterrupt:
                print("Interrupted by user (Ctrl+C), stopping capture...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(0.1)
    
    finally:
        # Signal all threads to stop
        print("Signaling threads to stop...")
        stop_flag.set()
        
        # Shutdown worker pool
        print("Shutting down worker pool...")
        worker_pool.shutdown(wait=True)
        
        # Shutdown API queue worker (completely separate queue)
        global _api_stop_flag, _api_worker_thread, _api_queue
        if _api_stop_flag is not None:
            print("Signaling API queue worker to stop...")
            _api_stop_flag.set()
            if _api_worker_thread is not None and _api_worker_thread.is_alive():
                _api_worker_thread.join(timeout=10)
                if _api_queue is not None:
                    remaining = _api_queue.qsize()
                    if remaining > 0:
                        print(f"⚠️ API queue has {remaining} pending calls (will be lost on shutdown)")
        
        # Wait for threads to finish
        capture_thread.join(timeout=5)
        for pt in processing_threads:
            pt.join(timeout=5)
        
        # Clean up
    print("Shutting down...")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass  # GUI not available, ignore
    print("Shutdown complete.")


def fetch_match_data(match_id, access_token=None):
    """
    Fetches match data from the API.

    Args:
        match_id: Match id (int or string UUID) as used in the backend query string
        access_token (str): Authorization token for the API

    Returns:
        dict: Match data or None if error occurred
    """
    if not access_token:
        access_token = ""

    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

    backend_url = config.get('api.backend_url', 'http://192.168.1.11:5006').rstrip('/')
    endpoint = config.get('endpoints.match_info', 'LeagueMatchData/LeagueMatch/info')
    full_url = f"{backend_url}/{endpoint}?matchId={match_id}"

    try:
        response = requests.get(full_url, headers=headers, timeout=30)
    except requests.RequestException as e:
        logger.error(
            "[match] fetch_match_data network error for matchId=%s: %s | url=%s",
            match_id,
            e,
            full_url,
        )
        print(
            f"[match] ERROR: could not reach backend for match info: {e}\n"
            f"  url: {full_url}\n"
            f"  Check api.backend_url in config.json (must be the same API TMS uses).",
            flush=True,
        )
        return None

    if not response.ok:
        body = (response.text or "")[:500]
        logger.error(
            "[match] fetch_match_data HTTP %s for matchId=%s: %s | url=%s",
            response.status_code,
            match_id,
            body,
            full_url,
        )
        print(
            f"[match] ERROR: match info API returned {response.status_code} for matchId={match_id}\n"
            f"  url: {full_url}\n"
            f"  body: {body}\n"
            f"  If 401/403: log in again or check token. If 404: match id may be wrong for this backend.",
            flush=True,
        )
        return None

    if not response.text:
        logger.error("[match] fetch_match_data empty body for matchId=%s url=%s", match_id, full_url)
        return None

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error("[match] fetch_match_data invalid JSON for matchId=%s: %s", match_id, e)
        return None

    if not data or "data" not in data:
        logger.error(
            "[match] fetch_match_data missing 'data' key for matchId=%s | keys=%s",
            match_id,
            list(data.keys()) if isinstance(data, dict) else type(data),
        )
        print(
            f"[match] ERROR: match info response has no 'data' for matchId={match_id}. "
            f"Killfeed capture will not start.",
            flush=True,
        )
        return None

    return data.get("data")


def setup_logger(match_name):
    """
    Sets up a logger for recording game events.

    Args:
        match_name (str): Name of the match for log file naming

    Returns:
        Logger: Configured logger object
    """
    # Create a dummy logger that does nothing
    game_logger = logging.getLogger("game_stats")
    game_logger.setLevel(logging.CRITICAL)
    game_logger.handlers.clear()
    return game_logger





def load_team_rosters_from_api(match_id, access_token=None):
    """
    Load team roster data from API.

    Args:
        match_id (int): Match ID to get team data for
        access_token (str): Authentication token for API access

    Returns:
        dict: Team roster data in a structured format
    """
    detection_logger.debug(f"Loading team rosters for match ID: {match_id}")
    try:
        # Set up headers with access token if provided
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        detection_logger.debug(f"Team roster request headers: {headers}")

        # Make API request to get team data
        backend_url = config.get('api.backend_url', 'http://192.168.1.11:5006').rstrip('/')
        endpoint = config.get('endpoints.team_players', 'LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players')
        full_url = f"{backend_url}/{endpoint}?matchId={match_id}"
        detection_logger.debug(f"Making team roster request to: {full_url}")
        
        api_start = time.time()
        response = requests.get(full_url, headers=headers)
        api_time = time.time() - api_start
        detection_logger.debug(f"Team roster API request took: {api_time:.4f}s")
        
        response.raise_for_status()
        detection_logger.debug(f"Team roster response status: {response.status_code}")

        # Parse response
        data = response.json()
        detection_logger.debug(f"Team roster response data: {data}")

        # Check if response is valid
        if not data or not data.get("success") or not data.get("data"):
            detection_logger.error(f"Invalid team data response: {data}")
            return {"teams": {}}

        # Format data into team roster structure
        formatted_teams_data = {"teams": {}}

        # Process each team
        for team_idx, team in enumerate(data.get("data", [])):
            team_name = team.get("team_name")
            detection_logger.debug(f"Processing team {team_idx}: {team_name}")
            
            if not team_name:
                detection_logger.warning(f"Team {team_idx} has no team_name, skipping")
                continue

            # Get player data
            players = {
                key: value
                for key, value in team.items()
                if key.startswith("player") and value
            }
            formatted_teams_data["teams"][team_name] = players
            detection_logger.debug(f"Team {team_name} players: {players}")

        detection_logger.info(f"Retrieved {len(formatted_teams_data['teams'])} teams from API")
        detection_logger.debug(f"Full team roster data: {formatted_teams_data}")
        return formatted_teams_data

    except requests.exceptions.RequestException as e:
        detection_logger.error(f"Error fetching team roster from API: {e}")
    except json.JSONDecodeError as e:
        detection_logger.error(f"Error parsing team roster API response: {e}")
    except Exception as e:
        detection_logger.error(f"Unexpected error in team roster API call: {e}")

    return {"teams": {}}


def fetch_and_update_team_players(match_id, access_token):
    """
    Fetch team players from the API and update the global TEAM_ROSTERS for OCR matching.
    
    Args:
        match_id (str): The match ID
        access_token (str): API access token
        
    Returns:
        dict: Updated team rosters in the format expected by OCR matching
    """
    global TEAM_ROSTERS, CURRENT_MATCH_ID, CURRENT_ACCESS_TOKEN
    
    try:
        print(f"🔄 Fetching team players for match: {match_id}")
        
        # Store current match info globally
        CURRENT_MATCH_ID = match_id
        CURRENT_ACCESS_TOKEN = access_token
        
        # Use the existing API function to get team data
        team_data = load_team_rosters_from_api(match_id, access_token)
        
        if not team_data or 'teams' not in team_data:
            print(f"❌ No team data received from API")
            return {}
        
        print(f"📊 Received team data: {len(team_data['teams'])} teams")
        
        # Convert API format to OCR matching format
        formatted_teams = {}
        
        for team_name, players in team_data['teams'].items():
            # Players are already in the correct format from load_team_rosters_from_api
            formatted_teams[team_name] = players
            
            print(f"   🏆 Team: {team_name} ({len(players)} players)")
            for player_key, player_name in players.items():
                print(f"      👤 {player_key}: {player_name}")
        
        # Update the global variable
        TEAM_ROSTERS = formatted_teams
        print(f"✅ Successfully updated OCR team rosters with {len(formatted_teams)} teams!")
        print(f"✅ Current match ID: {CURRENT_MATCH_ID}")
        
        return formatted_teams
        
    except Exception as e:
        print(f"❌ Failed to fetch and update team players: {e}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return {}


def get_player_team(player_name, team_rosters):
    """
    Find the team a player belongs to based on name similarity.

    Args:
        player_name (str): Player name to look up
        team_rosters (dict): Dictionary of team rosters

    Returns:
        str: The matching player name from team rosters, or "unknown" if no good match found
    """
    detection_logger.debug(f"Looking up player: '{player_name}'")
    
    # Handle special case player names
    if player_name in ["killer", "Playzone", "unknown"]:
        detection_logger.debug(f"Special case player name: {player_name}")
        return player_name

    # Set similarity thresholds
    processing_config = config.get_processing_config()
    HIGH_SIMILARITY_THRESHOLD = processing_config['similarity_threshold_high']  # 90% or higher is a definite match
    LOW_SIMILARITY_THRESHOLD = processing_config['similarity_threshold_low']  # 50% is minimum for consideration
    
    detection_logger.debug(f"Using similarity thresholds: high={HIGH_SIMILARITY_THRESHOLD}, low={LOW_SIMILARITY_THRESHOLD}")

    player_name_lower = player_name.lower()

    # Collect all player names
    all_players = []
    for team, team_data in team_rosters.items():
        player_values = [
            team_data[f"player{i}"]
            for i in range(1, 9)
            if f"player{i}" in team_data and team_data[f"player{i}"]
        ]
        all_players.extend([(p, team) for p in player_values if p])

    detection_logger.debug(f"Total players to match against: {len(all_players)}")
    if not all_players:
        logger.warning(f"[Similarity] No team roster - skipping match for '{player_name}'")
        return player_name

    # First pass: check for high-confidence matches (90%+)
    for player, team in all_players:
        similarity = fuzz.ratio(player_name_lower, player.lower()) / 100.0

        # If very close match found (≥90%), return immediately
        if similarity >= HIGH_SIMILARITY_THRESHOLD:
            msg = f"[Similarity] '{player_name}' -> '{player}' (high: {similarity:.0%})"
            logger.info(msg)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO     | [-] | score_ai.detection.killblocks | {msg}", flush=True)
            return player

    # Second pass: find all matches above 60% threshold
    matches = []
    for player, team in all_players:
        similarity = fuzz.ratio(player_name_lower, player.lower()) / 100.0
        if similarity >= LOW_SIMILARITY_THRESHOLD:
            matches.append((player, team, similarity))

    # Sort matches by similarity score (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    detection_logger.debug(f"Found {len(matches)} matches above threshold")

    # If we have matches above the threshold, return the best one
    if matches:
        best_player, best_team, best_score = matches[0]
        msg = f"[Similarity] '{player_name}' -> '{best_player}' (team={best_team}, score: {best_score:.0%})"
        logger.info(msg)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO     | [-] | score_ai.detection.killblocks | {msg}", flush=True)
        return best_player

    # Use RapidFuzz's process.extractOne as a fallback method
    if all_players:
        best_match_process = process.extractOne(
            player_name_lower, [p[0].lower() for p in all_players], scorer=fuzz.ratio
        )

        if (
            best_match_process
            and best_match_process[1] >= LOW_SIMILARITY_THRESHOLD * 100
        ):
            best_idx = best_match_process[2]
            best_player_name = all_players[best_idx][0]
            sim = best_match_process[1] / 100
            msg = f"[Similarity] '{player_name}' -> '{best_player_name}' (fallback: {sim:.0%})"
            logger.info(msg)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO     | [-] | score_ai.detection.killblocks | {msg}", flush=True)
            return best_player_name

    # No roster match: keep the incoming name (usually raw OCR) — do not replace with "unknown"
    # so logs/API show the detected string; scoring can still use team data when it matches.
    msg = f"[Similarity] No roster match for '{player_name}' — keeping raw name ({len(all_players)} roster players)"
    logger.info(msg)
    detection_logger.info(msg)
    return player_name


# COMMENTED OUT - OCR now handled by gRPC server
# def call_ocr_api(image_array):
#     """
#     Call the OCR API with an image array (numpy array).
#     
#     Args:
#         image_array (numpy.ndarray): Image as numpy array (grayscale or BGR)
#         
#     Returns:
#         list: OCR results in PaddleOCR format, or None if error
#         Format: [[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], (text, confidence)], ...]
#     """
#     pass  # OCR is now handled by gRPC server

def _deprecated_call_ocr_api(image_array):
    """DEPRECATED: OCR is now handled by the gRPC server. This function is kept for reference."""
    try:
        # Get OCR API configuration
        ocr_api_url = config.get('ocr.api_url', 'http://localhost:8000/ocr')
        ocr_api_key = config.get('ocr.api_key', 'your-secret-api-key-change-me')
        
        # Convert numpy array to image bytes (PNG format)
        if len(image_array.shape) == 2:
            # Grayscale
            _, img_encoded = cv2.imencode('.png', image_array)
        else:
            # BGR
            _, img_encoded = cv2.imencode('.png', image_array)
        
        img_bytes = img_encoded.tobytes()
        
        # Prepare request
        headers = {
            'X-API-Key': ocr_api_key
        }
        
        files = {
            'file': ('image.png', img_bytes, 'image/png')
        }
        
        # Make API call
        print(f"🌐 [OCR API] Calling OCR API at {ocr_api_url}...")
        api_start = time.time()
        response = requests.post(ocr_api_url, headers=headers, files=files, timeout=10)
        api_time = time.time() - api_start
        print(f"🌐 [OCR API] API call completed in {api_time*1000:.1f}ms")
        
        if response.status_code != 200:
            print(f"❌ OCR API error: Status {response.status_code}, Response: {response.text}")
            return None
        
        # Parse response
        result_data = response.json()
        
        if not result_data.get('success', False):
            print(f"❌ OCR API returned success=false: {result_data}")
            return None
        
        detections = result_data.get('detections', [])
        processing_time = result_data.get('processing_time_ms', 0)
        
        print(f"✅ OCR API: {len(detections)} detections in {processing_time:.1f}ms (API call: {api_time*1000:.1f}ms)")
        
        # Convert API response format to PaddleOCR format
        # PaddleOCR format: [[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], (text, confidence)], ...]
        ocr_result = []
        for detection in detections:
            text = detection.get('text', '')
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bounding_box', {})
            
            x1 = bbox.get('x1', 0)
            y1 = bbox.get('y1', 0)
            x2 = bbox.get('x2', 0)
            y2 = bbox.get('y2', 0)
            
            # Create polygon in PaddleOCR format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            # PaddleOCR format: [polygon, (text, confidence)]
            ocr_result.append([polygon, (text, confidence)])
        
        # Wrap in line format (PaddleOCR returns list of lines, each line has words)
        # For simplicity, put all detections in one line
        if ocr_result:
            return [ocr_result]
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ OCR API request error: {e}")
        return None
    except Exception as e:
        print(f"❌ OCR API error: {e}")
        import traceback
        traceback.print_exc()
        return None


# COMMENTED OUT - OCR now handled by gRPC server
# def combined_ocr_detection(frame, ocr, kill_block_coords):
#     """DEPRECATED: OCR is now handled by gRPC server."""
#     pass

def _deprecated_combined_ocr_detection(frame, ocr, kill_block_coords):
    """
    DEPRECATED: OCR is now handled by the gRPC server.
    This function is kept for reference only.
    
    Perform OCR only on ROI regions (kill blocks + stage region) to extract both killfeed text and stage text.
    This eliminates the need for two separate OCR calls while keeping results independent.
    
    Args:
        frame (numpy.ndarray): Full frame image
        ocr (PaddleOCR): OCR model instance
        kill_block_coords (list): List of kill block coordinates [(x1, y1, x2, y2), ...]
        
    Returns:
        tuple: (killfeed_text_boxes_dict, stage_number)
        killfeed_text_boxes_dict: {kill_block_index: [(x1, y1, x2, y2, text), ...], ...}
        stage_number: int or None
    """
    # If OCR isn't available or failed to initialize, skip gracefully
    if ocr is None:
        print(f"❌ OCR is None - skipping OCR detection")
        return {}, None
    try:
        # Initialize results
        killfeed_text_boxes_dict = {}
        stage_text_combined = ""
        
        # Stage OCR only if enabled to save CPU cycles when disabled
        if config.get('detection.enable_stage_detection', False):
            # Define stage region (top-right area)
            h, w = frame.shape[:2]
            stage_y1 = int(h * 0.05)
            stage_y2 = int(h * 0.35)
            stage_x1 = int(w * 0.70)
            stage_x2 = int(w * 1.00)
            
            # Crop stage region
            stage_crop = frame[stage_y1:stage_y2, stage_x1:stage_x2]
            
            # Convert stage crop to grayscale
            if len(stage_crop.shape) == 3:
                gray_stage = cv2.cvtColor(stage_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_stage = stage_crop
            
            # Fast preprocessing
            thresh_stage = cv2.threshold(gray_stage, 127, 255, cv2.THRESH_BINARY)[1]
            scaled_stage = cv2.resize(thresh_stage, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            
            # OCR stage region using API (DEPRECATED)
            stage_result = _deprecated_call_ocr_api(scaled_stage)
            
            # Process stage OCR results
            if stage_result:
                for line in stage_result:
                    if line is None:
                        continue
                    for word_info in line:
                        if not word_info or len(word_info) < 2:
                            continue
                        try:
                            text, confidence = word_info[1][0], word_info[1][1]
                            if confidence > 0.3 and len(text) >= 4:  # Filter out text shorter than 4 characters
                                stage_text_combined += text.strip() + " "
                        except Exception:
                            continue
        
        # Process each kill block region
        for kb_idx, (kb_x1, kb_y1, kb_x2, kb_y2) in enumerate(kill_block_coords):
            # Crop kill block region
            kill_block_crop = frame[kb_y1:kb_y2, kb_x1:kb_x2]
            
            if kill_block_crop.size == 0:
                continue
                
            # Convert to grayscale
            if len(kill_block_crop.shape) == 3:
                gray_kb = cv2.cvtColor(kill_block_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_kb = kill_block_crop
            
            # OCR kill block region using API (DEPRECATED)
            print(f"🔍 Processing Kill Block {kb_idx} at ({kb_x1}, {kb_y1}, {kb_x2}, {kb_y2})")
            print(f"📏 Kill block crop size: {kill_block_crop.shape}")
            print(f"📏 Grayscale size: {gray_kb.shape}")
            
            # Call deprecated OCR API
            kb_result = _deprecated_call_ocr_api(gray_kb)
            print(f"🎯 OCR Result for KB {kb_idx}: {kb_result}")
            
            # Process kill block OCR results
            if kb_result:
                print(f"✅ KB {kb_idx} has OCR results: {len(kb_result)} lines")
                killfeed_text_boxes_dict[kb_idx] = []
                for line_idx, line in enumerate(kb_result):
                    if line is None:
                        print(f"⚠️ KB {kb_idx} line {line_idx} is None")
                        continue
                    print(f"📝 KB {kb_idx} line {line_idx}: {line}")
                    for word_idx, word_info in enumerate(line):
                        if not word_info or len(word_info) < 2:
                            print(f"⚠️ KB {kb_idx} word {word_idx} invalid: {word_info}")
                            continue
                        try:
                            text, confidence = word_info[1][0], word_info[1][1]
                            print(f"🔤 KB {kb_idx} word {word_idx}: '{text}' (conf: {confidence:.3f})")
                            
                            if confidence > 0.3 and len(text) >= 4:
                                # Get coordinates in kill block local space
                                x1, y1 = map(int, word_info[0][0])
                                x2, y2 = map(int, word_info[0][2])
                                clean_text = text.strip()
                                
                                if clean_text:
                                    killfeed_text_boxes_dict[kb_idx].append((x1, y1, x2, y2, clean_text))
                                    print(f"✅ KB {kb_idx} added text: '{clean_text}' at ({x1},{y1},{x2},{y2})")
                            else:
                                print(f"❌ KB {kb_idx} filtered out: '{text}' (conf: {confidence:.3f}, len: {len(text)})")
                        except Exception as e:
                            print(f"❌ KB {kb_idx} word {word_idx} error: {e}")
                            continue
            else:
                print(f"❌ KB {kb_idx} has NO OCR results")
        

        
        # Process stage text to extract stage number (using exact same logic as extract_stage_with_ocr)
        stage_number = None
        if stage_text_combined.strip():
            raw_text = stage_text_combined.strip()

            
            # Handle Roman numerals in stage text (e.g., "Stage I" -> "Stage 1")
            corrected_text = raw_text
            corrected_text = re.sub(r'Stage\s+I\b', 'Stage 1', corrected_text, flags=re.IGNORECASE)
            corrected_text = re.sub(r'Stege\s+I\b', 'Stage 1', corrected_text, flags=re.IGNORECASE)
            

            
            # Pattern 1: "Stage" followed by a number (same word or immediately after)
            stage_with_number = re.search(r"Stage\s*(\d{1,2})", corrected_text, re.IGNORECASE)
            if stage_with_number:
                stage_val = int(stage_with_number.group(1))
                if 1 <= stage_val <= 14:
                    stage_number = stage_val

            # Pattern 1b: "Stege" (OCR error for "Stage") followed by a number
            if stage_number is None:
                stege_with_number = re.search(r"Stege\s*(\d{1,2})", corrected_text, re.IGNORECASE)
                if stege_with_number:
                    stage_val = int(stege_with_number.group(1))
                    if 1 <= stage_val <= 14:
                        stage_number = stage_val

            # Pattern 2: "Stage" followed by a number in the next word
            if stage_number is None:
                stage_followed_by_number = re.search(r"Stage\s+(\d{1,2})", corrected_text, re.IGNORECASE)
                if stage_followed_by_number:
                    stage_val = int(stage_followed_by_number.group(1))
                    if 1 <= stage_val <= 14:
                        stage_number = stage_val

            # Pattern 3: Word that contains both "Stage" and digits
            if stage_number is None:
                words = re.findall(r"\S+", corrected_text)
                for word in words:
                    # Check if word contains "stage" AND actual digits
                    if re.search(r"stage", word, re.IGNORECASE) and re.search(r"\d", word):
                        digits = re.findall(r"\d{1,2}", word)
                        if digits:
                            stage_val = int(digits[0])
                            if 1 <= stage_val <= 14:
                                stage_number = stage_val

            # Pattern 4: "Stage" found but no digits - default to Stage 1
            if stage_number is None and re.search(r"Stage", corrected_text, re.IGNORECASE):
                stage_number = 1
                
            # FALLBACK: Apply digit corrections only if all patterns failed
            if stage_number is None:
                corrected_text_with_digits = raw_text
                for old_char, new_char in DIGIT_CORRECTIONS.items():
                    corrected_text_with_digits = corrected_text_with_digits.replace(old_char, new_char)
                
                # Try patterns again with corrected text
                stage_with_number = re.search(r"Stage\s*(\d{1,2})", corrected_text_with_digits, re.IGNORECASE)
                if stage_with_number:
                    stage_val = int(stage_with_number.group(1))
                    if 1 <= stage_val <= 14:
                        stage_number = stage_val
                        
                # Try other patterns with corrected text if still None
                if stage_number is None:
                    stege_with_number = re.search(r"Stege\s*(\d{1,2})", corrected_text_with_digits, re.IGNORECASE)
                    if stege_with_number:
                        stage_val = int(stege_with_number.group(1))
                        if 1 <= stage_val <= 14:
                            stage_number = stage_val
        
        return killfeed_text_boxes_dict, stage_number
        
    except Exception as e:
        return {}, None


# COMMENTED OUT - OCR now handled by gRPC server
# def text_detection(kill_block_region, ocr):
#     """DEPRECATED: OCR is now handled by gRPC server."""
#     pass

def _deprecated_text_detection(kill_block_region, ocr):
    """
    DEPRECATED: OCR is now handled by the gRPC server.
    Detect text in a kill block image.

    Args:
        kill_block_region (numpy.ndarray): Image region to detect text in
        ocr (PaddleOCR): OCR model instance

    Returns:
        list: List of detected text boxes with coordinates and text
    """
    detection_logger.debug("Starting text detection")
    try:
        ocr_start = time.time()
        # Call OCR API instead of local PaddleOCR
        result = _deprecated_call_ocr_api(kill_block_region)
        ocr_time = time.time() - ocr_start
        detection_logger.debug(f"OCR processing took: {ocr_time:.4f}s")
        
        if not result:
            detection_logger.warning("No OCR result for kill block")
            return []

        final = []
        for line_idx, line in enumerate(result):
            if line is None:
                detection_logger.debug(f"Line {line_idx} is None")
                continue

            for word_idx, word_info in enumerate(line):
                if not word_info or len(word_info) < 2:
                    detection_logger.debug(f"Line {line_idx}, word {word_idx} has invalid info")
                    continue

                try:
                    text, confidence = word_info[1][0], word_info[1][1]
                    detection_logger.debug(f"Line {line_idx}, word {word_idx}: '{text}' (confidence: {confidence})")

                    # Lowered confidence threshold from 0.5 to 0.3
                    # Filter out text shorter than 4 characters
                    if confidence > 0.3 and len(text) >= 4:
                        detection_logger.debug(f"Text '{text}' passed confidence and length checks")

                        # Clean up text if needed
                        original_text = text
                        if " " in text and not text.endswith(" "):
                            text = text.split(" ")[1]
                            detection_logger.debug(f"Cleaned text from '{original_text}' to '{text}'")

                        # Get coordinates
                        x1, y1 = map(int, word_info[0][0])
                        x2, y2 = map(int, word_info[0][2])

                        # Clean text
                        clean_text = text.strip()

                        if clean_text:
                            final.append((x1, y1, x2, y2, clean_text))
                            detection_logger.debug(f"Added text box: '{clean_text}' at ({x1}, {y1}, {x2}, {y2})")
                    else:
                        reason = "confidence too low" if confidence <= 0.3 else "text too short"
                        detection_logger.debug(f"Text '{text}' filtered out - {reason} (confidence: {confidence}, length: {len(text)})")
                except Exception as e:
                    detection_logger.error(f"Error processing word {word_idx} in line {line_idx}: {e}")
                    continue

        detection_logger.info(f"Text detection completed: {len(final)} text boxes found")
        return final
    except Exception as e:
        detection_logger.error(f"OCR exception: {e}")
        return []


def _weapon_hint_for_separator(sep: str) -> str:
    """Map a combined-text separator to a best-effort event/weapon label."""
    s = (sep or "").lower()
    if "reviv" in s:
        return "revived"
    if "knock" in s or "knockout" in s:
        return "knockout"
    if "gunkill" in s or "gun-kill" in s or s.strip() in ("gun kill",):
        return "gunkill"
    return "kill"


def _try_split_combined_text_by_separators(combined_text: str, game: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Case-insensitive split on game separators (longest first). Returns
    (killer, victim, weapon_hint) or (None, None, None) if no separator matches.
    """
    if not combined_text or not str(combined_text).strip():
        return None, None, None
    ct = combined_text.strip()
    ct_l = ct.lower()
    game_cfg = get_game_config(game)
    separators: list = list(game_cfg.get("combined_text_separators", []))
    for sep in sorted(separators, key=len, reverse=True):
        if not sep or not str(sep).strip():
            continue
        s = str(sep).lower()
        idx = ct_l.find(s)
        if idx < 0:
            continue
        left = ct[:idx].strip()
        right = ct[idx + len(sep) :].strip()
        if not left and not right:
            continue
        w = _weapon_hint_for_separator(sep)
        return (left or None, right or None, w)
    return None, None, None


def infer_event_type_from_ocr_string(text: str) -> Optional[str]:
    """
    When the pipeline has no model weapon label, infer an event/weapon string from
    the full OCR line (order: specific phrases before generic "kill").

    Returns None if not enough signal (caller may still fall back to "kill" elsewhere).
    """
    if not text or not str(text).strip():
        return None
    t = re.sub(r"\s+", " ", str(text).lower().strip())
    if "reviv" in t:
        return "revived"
    if "gunkill" in t or re.search(r"\bgun[\s-]kill\b", t):
        return "gunkill"
    if "molotov" in t or "molly" in t:
        if "knock" in t:
            return "molotov-knockout"
        return "molotov-kill"
    if "grenade" in t or t.startswith("frag ") or " frag" in t:
        if "knock" in t:
            return "grenade-knockout"
        return "grenade-kill"
    if "playzone" in t or re.search(r"\bblue zone\b", t) or re.search(r"\bzone damage\b", t):
        if "knock" in t:
            return "playzone-knockout"
        return "playzone-kill"
    if re.search(r"\b(run over|roadkill|car ?blast|car-blast|carblow|car blow|vehicle)\b", t) or (
        "vehicle" in t and ("knock" in t or "kill" in t or "killed" in t)
    ):
        if "knock" in t:
            return "car-knockout"
        return "car-kill"
    if re.search(r"\b(knocked out|knockout)\b", t) or " knocked " in t or t.rstrip().endswith(" knock"):
        if "head" in t or "headshot" in t or " head-" in t:
            return "head-knockout"
        return "knockout"
    if re.search(r"\b(knock|knocked|knocks)\b", t) and "gunkill" not in t and "gunk" not in t:
        return "knockout"
    if "head" in t and "knock" in t:
        return "head-knockout"
    if "head" in t and "kill" in t:
        return "head-kill"
    if re.search(r"\b(killed|eliminate|shot down|eliminated)\b", t) or t.rstrip().endswith(" kill"):
        return "kill"
    if " kill" in t or t.startswith("kill ") or "killed" in t:
        return "kill"
    return None


def _refine_weapon_from_full_ocr(
    weapon_used: str,
    combined_text: str,
    text_boxes: list,
) -> str:
    """Prefer inferred event from full OCR if weapon is still unknown or a generic 'kill' but text disagrees."""
    w = (weapon_used or "").strip()
    w_lower = w.lower()
    box_text = " ".join(str(b[4]) for b in text_boxes if len(b) > 4) if text_boxes else ""
    blob = f"{combined_text or ''} {box_text}"
    if not str(blob).strip():
        return weapon_used
    inf = infer_event_type_from_ocr_string(blob)
    if not inf:
        return weapon_used
    if w_lower == "unknown" or w_lower == "":
        return inf
    if w_lower == "kill" and inf != "kill":
        return inf
    return weapon_used


def _is_plausible_event_weapon(weapon_used: str) -> bool:
    """
    Heuristic: strings that already describe an event type, or the legacy tokens.
    Normal gun names (e.g. M416) return False; caller may still post when killer+victim are known.
    """
    w = (weapon_used or "").lower()
    if not w or w == "unknown":
        return False
    markers = (
        "kill",
        "knockout",
        "revived",
        "revive",
        "gunkill",
        "grenade",
        "molotov",
        "self-knockout",
        "playzone",
        "head-knockout",
        "head-kill",
        "pan-kill",
        "pan-knockout",
        "knife-kill",
        "knife-knockout",
        "car-kill",
        "car-knockout",
        "car-blast",
        "carblow",
        "gun-knockout",
        "gun-kill",
    )
    return any(m in w for m in markers)


def _select_victim_ocr_text_box(
    text_boxes: list,
    victim_name: str,
    killer_name: str,
) -> Optional[tuple]:
    """
    Choose the OCR box that corresponds to the **victim** name for color sampling.
    Free Fire-style killfeeds: names are LTR, victim is usually the right-hand box.
    When OCR text is available, the box whose string best fuzzy-matches the victim name wins.
    """
    cands = _victim_ocr_text_box_candidates(text_boxes, victim_name, killer_name)
    return cands[0] if cands else None


def _victim_ocr_text_box_candidates(
    text_boxes: list,
    victim_name: str,
    killer_name: str,
) -> list:
    """
    Unique victim-likely OCR boxes in try order: best fuzzy, rightmost, then 2nd LTR.
    """
    if not text_boxes:
        return []
    by_x = sorted(text_boxes, key=lambda b: float(b[0]))
    by_right = sorted(text_boxes, key=lambda b: float(b[0]), reverse=True)
    out: list = []
    seen: set = set()

    def _add(b) -> None:
        if b is None or len(b) < 4:
            return
        key = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        if key in seen:
            return
        seen.add(key)
        out.append(b)

    vn = (victim_name or "").strip().lower()
    if vn and vn != "unknown":
        best = None
        best_score = -1.0
        for b in by_x:
            if len(b) < 5:
                continue
            t = str(b[4]).strip().lower()
            if not t:
                continue
            try:
                score = float(fuzz.ratio(vn, t))
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best = b
        if best is not None and best_score >= 45.0:
            _add(best)
    if by_right:
        _add(by_right[0])
    if len(by_x) >= 2:
        _add(by_x[1])
    if not out and by_x:
        _add(by_x[0])
    return out


def _inset_ocr_box(
    x1: int, y1: int, x2: int, y2: int,
    inset_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    """Shrink bbox toward center to skip anti-alias / outline (misleading hue)."""
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 6:
        return x1, y1, x2, y2
    dx = max(1, int(w * inset_ratio))
    dy = max(1, int(h * inset_ratio))
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def _victim_bright_text_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Foreground (glyph) mask: top fraction by luminance, so medians are not washed out by
    dark panel/background.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.array([], dtype=bool)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    frac = float(config.get("detection.victim_color_brightest_fraction", 0.35))
    frac = max(0.12, min(0.5, frac))
    thr = float(np.percentile(gray, 100.0 * (1.0 - frac)))
    m = gray >= max(thr, 18.0)
    if int(np.count_nonzero(m)) < 4:
        m = gray > 15
    if int(np.count_nonzero(m)) < 1:
        m = np.ones(gray.shape, dtype=bool)
    return m


def _victim_text_color_scores(roi_bgr: np.ndarray) -> tuple[float, float, float]:
    """
    Return (revived_score, kill_score, knock_score) in ~[0, 1] from BGR+HSV.
    - Samples **brightest pixels** (text) only.
    - **Knock (white)** is strongly penalized when saturation indicates colored text
      (avoids classifying red/green kills as white knock).
    - Rev / kill get extra weight from BGR channel dominance.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    tmask = _victim_bright_text_mask(roi_bgr)
    if tmask.size == 0 or not np.any(tmask):
        return 0.0, 0.0, 0.0

    h_ = hsv[:, :, 0].astype(np.float32)[tmask]
    s_raw = hsv[:, :, 1].astype(np.float32)[tmask] / 255.0
    v_ = hsv[:, :, 2].astype(np.float32)[tmask] / 255.0
    b_ = roi_bgr[:, :, 0].astype(np.float32)[tmask]
    g_ = roi_bgr[:, :, 1].astype(np.float32)[tmask]
    r_ = roi_bgr[:, :, 2].astype(np.float32)[tmask]

    h_m = float(np.median(h_)) if h_.size else 0.0
    s_m = float(np.median(s_raw)) if s_raw.size else 0.0
    v_m = float(np.median(v_)) if v_.size else 0.0
    b_m, g_m, r_m = float(np.median(b_)), float(np.median(g_)), float(np.median(r_))

    h_lo = float(config.get("detection.victim_color_green_hue_lo", 30))
    h_hi = float(config.get("detection.victim_color_green_hue_hi", 100))

    # Green / revive
    if h_lo <= h_m <= h_hi:
        hue_green = 1.0
    else:
        dist = min(abs(h_m - 60.0), 40.0)
        hue_green = max(0.0, 1.0 - dist / 45.0)
    g_domin = max(0.0, (g_m - max(r_m, b_m)) / 90.0)
    s_rev = (
        0.42 * hue_green
        + 0.22 * min(1.0, s_m * 1.15)
        + 0.32 * min(1.0, g_domin * 1.5)
    ) * min(1.0, v_m * 1.1 + 0.15)
    s_rev = float(np.clip(s_rev, 0.0, 1.0))

    # Red / kill
    if h_m <= 15.0 or h_m >= 172.0:
        hue_red = 1.0
    else:
        hue_red = max(0.0, 1.0 - min(abs(h_m - 0.0), abs(h_m - 180.0)) / 28.0) * 0.55
    r_domin = max(0.0, (r_m - max(g_m, b_m)) / 90.0)
    s_kill = (
        0.42 * hue_red
        + 0.24 * min(1.0, s_m * 1.05)
        + 0.30 * min(1.0, r_domin * 1.5)
    ) * min(1.0, v_m * 1.1 + 0.15)
    s_kill = float(np.clip(s_kill, 0.0, 1.0))

    # White / gray knock: only when *not* clearly chromatic
    spread = float(max(r_m, g_m, b_m) - min(r_m, g_m, b_m))
    neutral = max(0.0, 1.0 - spread / 55.0)
    s_knock = (
        neutral * 0.5
        + (1.0 - min(1.0, s_m * 1.4)) * 0.45
    ) * min(1.0, v_m * 0.9 + 0.2)
    # Strong saturation => not white text — crush knock
    if s_m > float(config.get("detection.victim_color_knock_max_saturation", 0.32)):
        over = (s_m - 0.32) / 0.55
        s_knock *= max(0.0, 1.0 - 0.92 * min(1.0, over))
    # BGR: clear green or red lead => not knock
    if g_m > r_m + 12.0 and g_m > b_m + 12.0:
        s_knock *= 0.15
    if r_m > g_m + 12.0 and r_m > b_m + 12.0:
        s_knock *= 0.15

    s_knock = float(np.clip(s_knock, 0.0, 1.0))

    # If rev and kill are both weak but chroma says something, nudge the stronger channel
    if s_m > 0.2 and s_knock < 0.45:
        if g_m - r_m > 8.0 and g_m > b_m + 5.0:
            s_rev = min(1.0, s_rev + 0.12)
        if r_m - g_m > 8.0 and r_m > b_m + 5.0:
            s_kill = min(1.0, s_kill + 0.12)

    return float(s_rev), float(s_kill), float(s_knock)


def _classify_victim_text_color_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """Pick revived / kill / knockout from 3-way scores; None if too ambiguous."""
    s_rev, s_kill, s_knock = _victim_text_color_scores(roi_bgr)
    margin = float(config.get("detection.victim_color_min_win_margin", 0.10))
    floor = float(config.get("detection.victim_color_min_winner_score", 0.22))
    scores = (("revived", s_rev), ("kill", s_kill), ("knockout", s_knock))
    scores = sorted(scores, key=lambda x: -x[1])
    best_name, best_s = scores[0]
    second_s = scores[1][1]
    if best_s < floor or (best_s - second_s) < margin:
        return None
    return best_name


def infer_event_from_victim_text_color(
    frame_bgr: np.ndarray,
    ocr_box: tuple,
) -> Optional[str]:
    """
    Single-box API (legacy): classify one OCR box on the full BGR frame.
    """
    return infer_event_from_victim_text_color_best_single(frame_bgr, ocr_box)


def infer_event_from_victim_text_color_best(
    frame_bgr: np.ndarray,
    text_boxes: list,
    victim_name: str,
    killer_name: str,
) -> Optional[str]:
    """
    Try several victim-likely OCR boxes; keep the box whose (rev, kill, knock) scores
    win with the best margin. Avoids bailing on the first box (often the killer).
    """
    if frame_bgr is None or not getattr(frame_bgr, "size", 0):
        return None
    candidates = _victim_ocr_text_box_candidates(text_boxes, victim_name, killer_name)
    if not candidates:
        return None
    h, w = frame_bgr.shape[:2]
    margin = float(config.get("detection.victim_color_min_win_margin", 0.10))
    floor = float(config.get("detection.victim_color_min_winner_score", 0.22))
    best_label: Optional[str] = None
    best_winner = -1.0
    for box in candidates:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1, x2, y2 = _inset_ocr_box(
            x1, y1, x2, y2, float(config.get("detection.victim_color_inset", 0.1))
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        s_rev, s_kill, s_knock = _victim_text_color_scores(roi)
        # Tight revive vs kill: use median G vs R on bright text
        if abs(s_rev - s_kill) < 0.12 and s_knock < max(s_rev, s_kill) + 0.02:
            tm = _victim_bright_text_mask(roi)
            if np.any(tm):
                g_med = float(np.median(roi[:, :, 1].astype(np.float32)[tm]))
                r_med = float(np.median(roi[:, :, 2].astype(np.float32)[tm]))
                if g_med - r_med > 7.0:
                    s_rev = min(1.0, s_rev + 0.18)
                    s_kill *= 0.55
                elif r_med - g_med > 7.0:
                    s_kill = min(1.0, s_kill + 0.18)
                    s_rev *= 0.55
        scores = (s_rev, s_kill, s_knock)
        labels = ("revived", "kill", "knockout")
        ordered = sorted(scores, reverse=True)
        top, second = ordered[0], ordered[1]
        if top < floor or (top - second) < margin:
            continue
        w_i = int(np.argmax(scores))
        lbl = labels[w_i]
        if config.get("detection.log_victim_color_scores", False):
            logger.debug(
                f"[victim_color] {x1},{y1},{x2},{y2} rev={s_rev:.2f} kill={s_kill:.2f} "
                f"knock={s_knock:.2f} -> {lbl} (top={top:.2f} margin={top - second:.2f})"
            )
        if top > best_winner:
            best_winner = top
            best_label = lbl
    return best_label


def infer_event_from_victim_text_color_best_single(
    frame_bgr: np.ndarray,
    ocr_box: tuple,
) -> Optional[str]:
    if not ocr_box or len(ocr_box) < 4 or frame_bgr is None or not frame_bgr.size:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = int(ocr_box[0]), int(ocr_box[1]), int(ocr_box[2]), int(ocr_box[3])
    x1, y1, x2, y2 = _inset_ocr_box(x1, y1, x2, y2, float(config.get("detection.victim_color_inset", 0.1)))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return _classify_victim_text_color_from_roi(roi)


def _fix_identical_killer_victim(
    killer_name: str,
    victim_name: str,
    combined_text: str,
    text_boxes: list,
    weapon_used: str,
) -> tuple[str, str]:
    """
    Killer and victim must differ for normal kills/revives/knocks. When OCR/roster maps them
    to the same string, recover from combined_text or left/right name boxes (3+ regions).
    Leaves self-knockout unchanged.
    """
    wu = (weapon_used or "").lower()
    if wu == "self-knockout" or "self-knockout" in wu:
        return killer_name, victim_name
    k = (killer_name or "").strip()
    v = (victim_name or "").strip()
    if not k or not v:
        return killer_name, victim_name
    if k.lower() != v.lower():
        return killer_name, victim_name
    ct = (combined_text or "").strip()
    if ct:
        parts = [p.strip() for p in re.split(r"\s+", ct) if p.strip()]
        if len(parts) >= 2 and parts[0].lower() != parts[1].lower():
            return parts[0], parts[1]
        if len(parts) >= 3 and parts[0].lower() == parts[1].lower():
            return parts[0], parts[2]
    if text_boxes and len(text_boxes) >= 3:
        by_x = sorted(text_boxes, key=lambda b: float(b[0]))
        t0 = str(by_x[0][4]).strip()
        t1 = str(by_x[-1][4]).strip()
        if t0 and t1 and t0.lower() != t1.lower():
            return t0, t1
    return killer_name, victim_name


def _is_invalid_same_killer_victim(killer_name: str, victim_name: str, weapon_used: str) -> bool:
    """True when killer==victim but the event is not an intentional self-knockout."""
    wu = (weapon_used or "").lower()
    if wu == "self-knockout" or "self-knockout" in wu:
        return False
    k = (killer_name or "").strip()
    v = (victim_name or "").strip()
    if not k or not v:
        return False
    return k.lower() == v.lower()


def extract_kill_info(text_boxes, other_objects, x1, y1, x2, y2, combined_text="", game=""):
    """
    Extract killer, victim and weapon info from detected text and objects.
    Game-specific parsing: BGMI uses "Killer killed Victim", FreeFire uses "Killer Victim".

    Args:
        text_boxes (list): List of detected text boxes (x1,y1,x2,y2,text,conf)
        other_objects (list): List of other detected objects
        x1, y1, x2, y2 (int): Coordinates of the kill block
        combined_text (str): Fallback - server's combined OCR string
        game (str): Game identifier for parsing ("bgmi", "freefire")

    Returns:
        tuple: (killer_name, victim_name, weapon_used)
    """
    print(f"🎯 Starting kill info extraction for KB at ({x1}, {y1}, {x2}, {y2})")
    print(f"📦 Text boxes count: {len(text_boxes)}")
    print(f"📝 Combined text: {repr(combined_text[:80]) if combined_text else 'empty'}")
    print(f"🔫 Other objects count: {len(other_objects)}")
    
    detection_logger.debug("Starting kill info extraction")
    killer_name = "unknown"
    victim_name = "unknown"
    weapon_used = "unknown"

    detection_logger.debug(f"Text boxes: {text_boxes}")
    detection_logger.debug(f"Other objects: {other_objects}")
    detection_logger.debug(f"Kill block coordinates: ({x1}, {y1}, {x2}, {y2})")
    
    # Print detailed text boxes info
    for i, text_box in enumerate(text_boxes):
        print(f"📝 Text box {i}: {text_box}")
    
    # Print detailed other objects info  
    for i, obj in enumerate(other_objects):
        print(f"🔫 Object {i}: {obj}")

    # Detect weapons inside the kill block
    detected_weapons = []
    for obj_idx, obj_coords in enumerate(other_objects):
        ox1, oy1, ox2, oy2, label = obj_coords
        is_inside_kb = is_inside((ox1, oy1, ox2, oy2), (x1, y1, x2, y2))
        detection_logger.debug(f"Object {obj_idx} '{label}' at ({ox1}, {oy1}, {ox2}, {oy2}): inside kill block = {is_inside_kb}")
        if is_inside_kb:
            detected_weapons.append(label)

    detection_logger.info(f"Detected weapons inside kill block: {detected_weapons}")

    # Determine weapon used from detected weapons
    if detected_weapons:
        if len(detected_weapons) > 1:
            detection_logger.debug("Multiple weapons detected, checking for special types")
            # Check for special weapon types
            w_lower = [w.lower() for w in detected_weapons]
            has_kill = any("kill" in w for w in w_lower)
            has_knockout = any(
                "knockout" in w
                or "gun-knock" in w
                or w.strip() in ("gun knockout", "gun-knockout", "revived", "gunkill")
                for w in w_lower
            )
            detection_logger.debug(f"Has kill: {has_kill}, Has knockout: {has_knockout}")

            if has_kill and has_knockout:
                # Prefer knockout over kill
                weapon_used = next(
                    w for w in detected_weapons if "knockout" in w.lower()
                )
                detection_logger.debug(f"Chose knockout weapon: {weapon_used}")
            elif has_kill:
                # Use first kill weapon
                weapon_used = next(w for w in detected_weapons if "kill" in w.lower())
                detection_logger.debug(f"Chose kill weapon: {weapon_used}")
        else:
            # Only one weapon detected
            weapon_used = detected_weapons[0]
            detection_logger.debug(f"Single weapon detected: {weapon_used}")

    detection_logger.info(f"Final weapon used: {weapon_used}")

    # Fallback: when no text_boxes but combined_text exists (game-specific parsing)
    if len(text_boxes) == 0 and combined_text and combined_text.strip():
        ct = combined_text.strip()
        game_cfg = get_game_config(game)
        k, v, w_hint = _try_split_combined_text_by_separators(combined_text, game)
        if k and v:
            killer_name, victim_name = k, v
            if weapon_used == "unknown" and w_hint:
                weapon_used = w_hint
            detection_logger.info(
                f"Parsed from combined_text (case-insensitive split): {killer_name} -> {victim_name} ({weapon_used})"
            )
            weapon_used = _refine_weapon_from_full_ocr(weapon_used, combined_text, text_boxes)
            return killer_name, victim_name, weapon_used
        # FreeFire format: "Name1 Name2" (two names, space-separated) - only when enabled for game
        if game_cfg.get("combined_text_space_fallback"):
            parts = [p.strip() for p in ct.split() if p.strip()]
            if len(parts) >= 2 and all(len(p) >= 2 for p in parts[:2]):
                killer_name = parts[0]
                victim_name = parts[1]
                if weapon_used == "unknown":
                    weapon_used = infer_event_type_from_ocr_string(ct) or "kill"
                detection_logger.info(f"Parsed space-separated format (FreeFire): {killer_name} -> {victim_name}")
                weapon_used = _refine_weapon_from_full_ocr(weapon_used, combined_text, text_boxes)
                return killer_name, victim_name, weapon_used
        # Single name (self-knockout or victim only)
        victim_name = ct
        killer_name = "killer" if weapon_used != "self-knockout" else ct
        if "self" in ct.lower() or "knockout" in ct.lower():
            weapon_used = "self-knockout"
            killer_name, victim_name = ct, ct
        elif weapon_used == "unknown":
            weapon_used = infer_event_type_from_ocr_string(ct) or "kill"
        detection_logger.info(f"Parsed single from combined_text: killer={killer_name}, victim={victim_name}")
        weapon_used = _refine_weapon_from_full_ocr(weapon_used, combined_text, text_boxes)
        return killer_name, victim_name, weapon_used

    game_cfg = get_game_config(game)

    # Stable LTR order: left = killer, right = victim (gRPC box order is not always sorted)
    if len(text_boxes) >= 1:
        text_boxes = sorted(text_boxes, key=lambda b: float(b[0]))

    # If combined text has "KillerName VictimName" but OCR returned one big box, split names
    if (
        len(text_boxes) == 1
        and combined_text
        and str(combined_text).strip()
        and game_cfg.get("combined_text_space_fallback")
    ):
        parts = [p.strip() for p in combined_text.split() if p.strip()]
        if len(parts) >= 2 and all(len(p) >= 2 for p in parts[:2]):
            killer_name, victim_name = parts[0], parts[1]
            detection_logger.info(
                f"Split single OCR box using combined_text space words: {killer_name} -> {victim_name}"
            )
            if weapon_used == "unknown":
                weapon_used = infer_event_type_from_ocr_string(combined_text) or "kill"
            weapon_used = _refine_weapon_from_full_ocr(weapon_used, combined_text, text_boxes)
            return killer_name, victim_name, weapon_used

    # Extract player names based on special cases
    if len(text_boxes) == 1:
        detection_logger.debug("Single text box case")
        if weapon_used == "self-knockout":
            # Self-knockout case
            killer_name = text_boxes[0][4]
            victim_name = text_boxes[0][4]
            detection_logger.debug(f"Self-knockout detected: {killer_name}")
        elif weapon_used == "kill":
            # Generic kill with single player name
            killer_name = "killer"
            victim_name = text_boxes[0][4]
            detection_logger.debug(f"Generic kill detected: killer -> {victim_name}")
        else:
            # Single text box but not self-knockout or generic kill
            # This might be a victim name only
            victim_name = text_boxes[0][4]
            detection_logger.debug(f"Single text box (victim only): {victim_name}")
    elif len(text_boxes) > 1:
        # Normal kill: LTR left=killer, right=victim; 3+ boxes often have an icon/word between names
        if len(text_boxes) >= 3:
            killer_name = str(text_boxes[0][4]).strip()
            victim_name = str(text_boxes[-1][4]).strip()
        else:
            killer_name = str(text_boxes[0][4]).strip()
            victim_name = str(text_boxes[1][4]).strip()
        detection_logger.debug(f"Normal kill detected: {killer_name} -> {victim_name}")
    else:
        detection_logger.warning("No text boxes found")

    print(f"🎯 FINAL EXTRACTION RESULT:")
    print(f"   Killer: {killer_name}")
    print(f"   Victim: {victim_name}")
    print(f"   Weapon: {weapon_used}")
    print(f"   Text boxes used: {len(text_boxes)}")
    
    detection_logger.info(f"Extraction result: Killer={killer_name}, Victim={victim_name}, Weapon={weapon_used}")
    weapon_used = _refine_weapon_from_full_ocr(weapon_used, combined_text, text_boxes)

    # Final recovery pass: if one side is still unknown, try to parse the
    # server-supplied combined_text before this detection is dropped.
    if combined_text and combined_text.strip() and (killer_name == "unknown" or victim_name == "unknown"):
        ct = combined_text.strip()
        k, v, w_hint = _try_split_combined_text_by_separators(ct, game)
        if k and v:
            killer_name, victim_name = k, v
            if weapon_used == "unknown" and w_hint:
                weapon_used = w_hint
            weapon_used = _refine_weapon_from_full_ocr(weapon_used, combined_text, text_boxes)
            detection_logger.info(
                f"Recovered from combined_text fallback: {killer_name} -> {victim_name} ({weapon_used})"
            )
            return killer_name, victim_name, weapon_used

        game_cfg = get_game_config(game)
        if game_cfg.get("combined_text_space_fallback"):
            parts = [p.strip() for p in ct.split() if p.strip()]
            if len(parts) >= 2 and all(len(p) >= 2 for p in parts[:2]):
                killer_name = parts[0]
                victim_name = parts[1]
                if weapon_used == "unknown":
                    weapon_used = infer_event_type_from_ocr_string(ct) or "kill"
                weapon_used = _refine_weapon_from_full_ocr(weapon_used, combined_text, text_boxes)
                detection_logger.info(
                    f"Recovered space-separated fallback: {killer_name} -> {victim_name} ({weapon_used})"
                )
                return killer_name, victim_name, weapon_used

    return killer_name, victim_name, weapon_used


def is_inside(inner_box, outer_box):
    """
    Check if one bounding box is inside another.

    Args:
        inner_box (tuple): (x1, y1, x2, y2) of inner box
        outer_box (tuple): (x1, y1, x2, y2) of outer box

    Returns:
        bool: True if inner_box is inside outer_box
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    # Check if boundaries are close enough
    upper_measurement = abs(oy1 - iy1)
    lower_measurement = abs(oy2 - iy2)

    if upper_measurement < 20 and lower_measurement < 20:
        return True

    # Traditional containment check
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def image_to_base64(image_path):
    """
    Convert an image file to base64 encoding.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64-encoded image or None if error
    """
    abs_path = os.path.abspath(image_path) if image_path else ""
    detection_logger.debug(f"Converting image to base64: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            detection_logger.debug(f"Base64 conversion successful, size: {len(base64_data)} chars")
            return base64_data
    except FileNotFoundError as e:
        detection_logger.error(f"Image file not found: {image_path} (absolute: {abs_path})")
        print(f"[Base64] File not found: {abs_path}", flush=True)
        return None
    except Exception as e:
        detection_logger.error(f"Error converting image to base64: {e}")
        return None


async def save_to_mongodb_async(
    killer_name,
    weapon_used,
    victim_name,
    image_path,
    sift_weapon,
    position,
    match_id=1,
    access_token=None,
    frame_timestamp=None,
    detected_stage=None,
    match_name=None,
    processing_start_time=None,
    processing_ready_time=None,
    frame_number=None,
):
    """
    Async version: Save kill event record to API instead of MongoDB and delete the local image.
    Prevents posting duplicate killfeeds by checking the last 5 events.
    Only killer_name, victim_name, and weapon_used (from API) are used for duplicate detection.
    """
    weapon_used = _normalize_weapon_for_tms_api(weapon_used)
    print(f"🎯 KILLFEED SAVE ATTEMPT: {killer_name} -> {victim_name} ({weapon_used})", flush=True)
    print(f"   Image: {image_path}", flush=True)
    
    detection_logger.debug("Starting async save to API")
    try:
        # Track statistics
        duplicate_stats['total_processed'] += 1
        
        # Create a unique key for the event (for logging/stats only - no duplicate filtering)
        event_key = f"{killer_name}|{victim_name}|{weapon_used}"
        detection_logger.debug(f"Event key: {event_key}")

        # No duplicate filtering - post all detected killfeeds to API
        
        detection_logger.debug(f"Added to recent killfeeds. Queue size: {len(recent_killfeeds)}")
        detection_logger.debug(f"Recent killfeeds queue: {list(recent_killfeeds)}")

        # Convert image to base64
        print(f"🔄 Converting image to base64...")
        base64_start = time.time()
        base64_image = image_to_base64(image_path)
        base64_time = time.time() - base64_start
        print(f"✅ Base64 conversion completed in {base64_time:.4f}s")
        detection_logger.debug(f"Base64 conversion took: {base64_time:.4f}s")
        
        if not base64_image:
            print(f"❌ FAILED TO CONVERT IMAGE TO BASE64 - NOT POSTING (image missing or unreadable: {image_path})")
            detection_logger.error(f"Failed to convert image to base64: {image_path}")
            # Remove from processing set on failure
            with processing_killfeeds_lock:
                processing_killfeeds.discard(event_key)
            return position

        # Create payload for API (base64 image is always included in the body)
        sift_weapon_name = sift_weapon[1] if isinstance(sift_weapon, (list, tuple)) and len(sift_weapon) > 1 else None
        image_key = config.get("api.killfeed_image_key", "image")
        image_value = base64_image
        if config.get("api.killfeed_image_data_url", False):
            image_value = f"data:image/jpeg;base64,{base64_image}"
        api_payload = {
            "killerName": killer_name,
            "victimName": victim_name,
            "weaponUsed": weapon_used,
            "imagePath": image_path,
            "siftWeapon": sift_weapon_name,
            "stage": detected_stage if detected_stage is not None else 0,
        }
        api_payload[image_key] = image_value

        # Add timestamp for proper ordering if provided
        if frame_timestamp:
            api_payload["timestamp"] = frame_timestamp

        detection_logger.debug(f"API payload created (image size: {len(base64_image)} chars, stage: {detected_stage})")

        # Set headers with authorization token if provided
        headers = {"accept": "text/plain", "Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        # Make async API request
        backend_url = config.get('api.backend_url', 'http://192.168.1.11:5006').rstrip('/')
        endpoint = config.get('endpoints.killfeed', 'LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed')
        api_url = f"{backend_url}/{endpoint}?matchId={match_id}"
        detection_logger.debug(f"Making async API request to: {api_url}")

        # Track API save attempt
        duplicate_stats['api_saves_attempted'] += 1
        post_log = (
            f"[API] POST killfeed | {api_url} | "
            f"killer={killer_name} | victim={victim_name} | weapon={weapon_used} | "
            f"base64 image in body (key={image_key!r}, {len(image_value)} chars)"
        )
        logger.info(post_log)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO     | [-] | score_ai.detection.killblocks | {post_log}", flush=True)
        print(f"🌐 SENDING HTTP POST with base64 image in JSON body (key={image_key!r}) to: {api_url}", flush=True)
        
        # Timing: how long we waited before calling API (ready -> now) and how long processing took (capture -> ready)
        post_call_time = time.time()
        wait_before_post_s = (post_call_time - processing_ready_time) if processing_ready_time else None
        processing_time_s = (processing_ready_time - processing_start_time) if (processing_start_time and processing_ready_time) else None
        
        api_start = time.time()
        post_start_time = time.time()  # Track total posting time
        
        # Use aiohttp for async HTTP request
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(api_url, headers=headers, json=api_payload) as response:
                    api_time = time.time() - api_start
                    post_total_time = time.time() - post_start_time
                    response_log = (
                        f"[API] POST killfeed response | status={response.status} | "
                        f"time={api_time:.2f}s | killer={killer_name} -> {victim_name} | {weapon_used}"
                    )
                    logger.info(response_log)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | INFO     | [-] | score_ai.detection.killblocks | {response_log}", flush=True)
                    
                    # Log timing to file
                    try:
                        os.makedirs(GAME_LOGS_DIR, exist_ok=True)
                        log_path = os.path.join(GAME_LOGS_DIR, 'killfeed_posting_times.log')
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(f"[{datetime.now().isoformat()}] Frame posting time: {post_total_time:.4f}s | API time: {api_time:.4f}s | Frame: {frame_timestamp if frame_timestamp else 'N/A'}\n")
                    except Exception:
                        pass
                    
                    detection_logger.info(f"Async API save request took: {api_time:.4f}s")
                    detection_logger.info(f"Total killfeed posting time: {post_total_time:.4f}s")
                    detection_logger.debug(f"API response status: {response.status}")

                    # Append to timing report: processing time vs wait-before-post vs API time (for in-order + immediate tuning)
                    _append_killfeed_timing_report(
                        frame_number=frame_number,
                        processing_time_s=round(processing_time_s, 4) if processing_time_s is not None else "",
                        wait_before_post_s=round(wait_before_post_s, 4) if wait_before_post_s is not None else "",
                        api_post_s=round(api_time, 4),
                        killer_name=killer_name,
                        victim_name=victim_name,
                        weapon_used=weapon_used,
                        posted_at_iso=datetime.now().isoformat(),
                    )

                    # Check if request was successful
                    if response.status == 200 or response.status == 201:
                        duplicate_stats['api_saves_successful'] += 1
                        print(f"✅ KILLFEED SAVED SUCCESSFULLY!")
                        print(f"   Event key: {event_key}")
                        print(f"   Response status: {response.status}")
                        print(f"   Save stats: {duplicate_stats['api_saves_successful']}/{duplicate_stats['api_saves_attempted']} successful")
                        
                        detection_logger.info(f"Successfully saved killfeed: {event_key}")
                        detection_logger.info(f"Save stats: {duplicate_stats['api_saves_successful']}/{duplicate_stats['api_saves_attempted']} successful")
                        
                        # Move from processing set to recent_killfeeds (thread-safe)
                        with processing_killfeeds_lock:
                            processing_killfeeds.discard(event_key)
                        with recent_killfeeds_lock:
                            recent_killfeeds.append(event_key)
                        
                        # Delete the image file after successful save
                        try:
                            os.remove(image_path)
                            print(f"🗑️ Cleaned up image after successful save: {image_path}")
                            detection_logger.debug(f"Cleaned up image after successful save: {image_path}")
                        except Exception as e:
                            print(f"❌ Error cleaning up image after save: {e}")
                            detection_logger.error(f"Error cleaning up image after save: {e}")
                        return position + 1
                    else:
                        response_text = await response.text()
                        fail_log = (
                            f"[API] POST killfeed FAILED | status={response.status} | "
                            f"killer={killer_name} -> {victim_name} | response={response_text[:100]}"
                        )
                        logger.error(fail_log)
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR    | [-] | score_ai.detection.killblocks | {fail_log}", flush=True)
                        
                        detection_logger.error(f"API request failed with status {response.status}")
                        detection_logger.error(f"API response text: {response_text}")
                        detection_logger.error(f"Save stats: {duplicate_stats['api_saves_successful']}/{duplicate_stats['api_saves_attempted']} successful")
                        
                        # Remove from processing set on failure (so it can be retried if needed)
                        with processing_killfeeds_lock:
                            processing_killfeeds.discard(event_key)
                        
                        return position
        except Exception as post_err:
            print(f"❌ HTTP POST EXCEPTION: {post_err}", flush=True)
            import traceback
            traceback.print_exc()
            with processing_killfeeds_lock:
                if 'event_key' in locals():
                    processing_killfeeds.discard(event_key)
            raise
    except Exception as e:
        exc_log = (
            f"[API] POST killfeed EXCEPTION | error={e} | "
            f"event_key={event_key if 'event_key' in locals() else 'Unknown'}"
        )
        logger.error(exc_log)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR    | [-] | score_ai.detection.killblocks | {exc_log}", flush=True)
        
        detection_logger.error(f"Exception in save_to_mongodb_async: {e}")
        
        # Remove from processing set on exception
        if 'event_key' in locals():
            with processing_killfeeds_lock:
                processing_killfeeds.discard(event_key)
    finally:
        # Ensure cleanup even if errors occur
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"🗑️ Final cleanup of image: {image_path}")
                detection_logger.debug(f"Final cleanup of image: {image_path}")
            except Exception as e:
                print(f"❌ Error in final image cleanup: {e}")
                detection_logger.error(f"Error in final image cleanup: {e}")
        
        # Log final summary
        print(f"📊 KILLFEED SAVE SUMMARY:")
        print(f"   Total processed: {duplicate_stats['total_processed']}")
        print(f"   Duplicates rejected: {duplicate_stats['duplicates_rejected']}")
        print(f"   API saves attempted: {duplicate_stats['api_saves_attempted']}")
        print(f"   API saves successful: {duplicate_stats['api_saves_successful']}")
        print(f"   Success rate: {(duplicate_stats['api_saves_successful']/max(duplicate_stats['api_saves_attempted'], 1)*100):.1f}%")
        print(f"   Duplicate rate: {(duplicate_stats['duplicates_rejected']/max(duplicate_stats['total_processed'], 1)*100):.1f}%")
        print(f"   ==========================================")
        
    return position


def save_to_mongodb(
    killer_name,
    weapon_used,
    victim_name,
    image_path,
    sift_weapon,
    position,
    match_id=1,
    access_token=None,
    frame_timestamp=None,
    detected_stage=None,
    match_name=None,
):
    """
    Synchronous wrapper for async save function.
    This maintains backward compatibility while allowing async calls.
    """
    print(f"🔄 SYNCHRONOUS WRAPPER CALLED:")
    print(f"   Killer: {killer_name}")
    print(f"   Victim: {victim_name}")
    print(f"   Weapon: {weapon_used}")
    print(f"   Match ID: {match_id}")
    
    # Create a new event loop if one doesn't exist
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async function
    return loop.run_until_complete(
        save_to_mongodb_async(
            killer_name,
            weapon_used,
            victim_name,
            image_path,
            sift_weapon,
            position,
            match_id,
            access_token,
            frame_timestamp,
            detected_stage,
            match_name,
        )
    )


def verify_weapon_with_sift(kill_block_folder, kill_type_from_api):
    """
    Verify weapon detection using SIFT matching against single-template images only.
    Returns only the base weapon name without kill/knockout suffixes.
    Args:
        kill_block_folder (str): Folder containing kill block images
        kill_type_from_api (str): Kill/knockout type from the API (e.g., kill, knockout, head-kill, etc.)
    Returns:
        tuple: (verification_result, detected_weapon_label)
    """
    # Get API weapons from configuration
    api_weapons = config.get_api_weapons()
    if kill_type_from_api in api_weapons:
        # For special weapons like grenade, molotov, etc., return just the base name
        if any(special_weapon in kill_type_from_api.lower() for special_weapon in ['grenade', 'molotov', 'car', 'playzone', 'self', 'knife', 'pan']):
            # Extract base weapon name (remove -kill, -knockout, etc.)
            base_weapon = kill_type_from_api.split('-')[0]
            return True, base_weapon
        return True, kill_type_from_api
    
    try:
        sift = cv2.SIFT_create(nfeatures=400)
        best_weapon = None
        best_score = 0
        
        try:
            latest_kill_block = max(
                glob.glob(os.path.join(kill_block_folder, "*.jpg")),
                key=os.path.getctime,
            )
        except Exception:
            return False, "unknown"
            
        target_image = cv2.imread(latest_kill_block)
        if target_image is None:
            return False, "unknown"
            
        gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        image_resize_factor = config.get('processing.image_resize_factor', 0.75)
        gray_target = cv2.resize(gray_target, (0, 0), fx=image_resize_factor, fy=image_resize_factor)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_target, None)
        
        if descriptors2 is None:
            return False, "unknown"
            
        for template_name, template_list in sift_template_cache.items():
            for descriptors1, template_image_path in template_list:
                try:
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    
                    if descriptors1.dtype != descriptors2.dtype:
                        descriptors1 = descriptors1.astype(np.float32)
                        descriptors2 = descriptors2.astype(np.float32)
                        
                    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                    if not matches:
                        continue
                        
                    good_matches = []
                    sift_match_ratio = config.get('detection.sift_match_ratio', 0.75)
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < sift_match_ratio * n.distance:
                                good_matches.append(m)
                    
                    num_good_matches = len(good_matches)
                    
                    if num_good_matches > best_score:
                        best_score = num_good_matches
                        best_weapon = template_name
                        
                except Exception:
                    continue
        
        sift_min_matches = config.get('detection.sift_min_matches', 10)
        if best_weapon and best_score >= sift_min_matches:
            # Return only the base weapon name without any suffixes
            return True, best_weapon
        else:
            return False, "unknown"
            
    except Exception:
        return False, "unknown"


if __name__ == "__main__":
    obs_frame_capture() 
