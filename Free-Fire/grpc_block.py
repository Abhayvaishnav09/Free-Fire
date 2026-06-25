"""gRPC Server - Killfeed Detection and Status Analysis Service."""
import cv2
import os
import numpy as np
import warnings
import logging
import threading
import time
import subprocess
import sys
import gc
from concurrent import futures
import grpc
from ultralytics import YOLO

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KillfeedDetectionServicer(killfeed_detection_pb2_grpc.KillfeedDetectionServiceServicer):
    """gRPC service implementation for killfeed detection."""
    
    def __init__(self, model_path="best (1).pt"):
        """Initialize the killfeed detection servicer."""
        logger.info("🔧 Initializing Killfeed Detection Servicer...")
        self.model = self._load_model(model_path)
        if self.model is None:
            logger.error("❌ Failed to load YOLO model")
        else:
            logger.info("✅ Killfeed Detection Servicer ready!")
    
    def _load_model(self, model_path):
        """Load YOLO model for killfeed detection."""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Model file not found: {model_path}")
                return None
            logger.info("📦 Loading YOLO model...")
            model = YOLO(model_path)
            logger.info("✅ YOLO model loaded")
            
            if hasattr(model, 'names'):
                class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
                logger.info(f"📋 Available classes: {class_names}")
            
            return model
        except Exception as e:
            logger.error(f"❌ Error loading YOLO model: {e}")
            return None
    
    def ProcessFrame(self, request, context):
        """
        Process a frame to detect killfeeds and determine their status.
        Server continues running even if processing takes a long time.
        
        Args:
            request: ProcessFrameRequest containing frame image bytes
            context: gRPC context
        
        Returns:
            ProcessFrameResponse with detection results
        """
        start_time = time.time()
        frame = None
        frame_array = None
        try:
            # Decode image from bytes
            frame_array = np.frombuffer(request.frame_image, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            # Explicitly delete the frame_array to free memory immediately
            del frame_array
            gc.collect()
            
            if frame is None or frame.size == 0:
                logger.warning("⚠️ Failed to decode frame image")
                return killfeed_detection_pb2.ProcessFrameResponse(
                    success=False,
                    error_message="Failed to decode frame image"
                )
            
            # Process frame for detections (this may take time, but server keeps running)
            logger.debug(f"🔄 Processing frame (size: {frame.shape})...")
            detections = self._process_frame(frame)
            
            # Explicitly delete frame after processing to free memory
            del frame
            gc.collect()
            
            # Build response
            response = killfeed_detection_pb2.ProcessFrameResponse(
                success=True,
                detections=[]
            )
            
            # Limit maximum number of detections to prevent huge responses
            max_detections = 10
            detections = detections[:max_detections]
            
            total_response_size = 0
            max_response_size = 40 * 1024 * 1024  # 40MB limit (below 50MB gRPC limit)
            
            for detection in detections:
                try:
                    cropped_image = detection['cropped_image']
                    
                    # Resize large cropped images to reduce memory usage
                    max_crop_size = 800  # Maximum dimension for cropped images
                    h, w = cropped_image.shape[:2]
                    if max(h, w) > max_crop_size:
                        scale = max_crop_size / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        cropped_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Encode cropped image to JPEG bytes (much smaller than PNG)
                    # Use quality 85 for good balance between size and quality
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                    _, cropped_encoded = cv2.imencode('.jpg', cropped_image, encode_params)
                    cropped_bytes = cropped_encoded.tobytes()
                    
                    # Check if adding this detection would exceed size limit
                    if total_response_size + len(cropped_bytes) > max_response_size:
                        logger.warning(f"⚠️ Response size limit reached, truncating detections at {len(response.detections)}")
                        break
                    
                    # Create bounding box
                    bbox = killfeed_detection_pb2.BoundingBox(
                        x1=detection['bbox'][0],
                        y1=detection['bbox'][1],
                        x2=detection['bbox'][2],
                        y2=detection['bbox'][3]
                    )
                    
                    # Create detection result
                    result = killfeed_detection_pb2.DetectionResult(
                        cropped_image=cropped_bytes,
                        status=detection['status'],
                        bbox=bbox,
                        confidence=detection['confidence'],
                        detection_sources=detection['detection_sources']
                    )
                    
                    response.detections.append(result)
                    total_response_size += len(cropped_bytes)
                    
                    # Explicitly delete intermediate variables to free memory
                    del cropped_image, cropped_encoded, cropped_bytes
                    
                except MemoryError as me:
                    logger.error(f"❌ MemoryError encoding detection result: {me}, stopping...")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Error encoding detection result: {e}, skipping...")
                    continue
            
            # Force garbage collection before returning response
            gc.collect()
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Processed frame: {len(response.detections)} detections in {processing_time:.2f}s")
            return response
            
        except MemoryError as me:
            # Handle MemoryError specifically
            logger.error(f"❌ MemoryError processing frame: {me}")
            # Clean up any remaining references
            if frame is not None:
                del frame
            if frame_array is not None:
                del frame_array
            gc.collect()
            # Return error response but server keeps running
            return killfeed_detection_pb2.ProcessFrameResponse(
                success=False,
                error_message=f"Memory error processing frame: {str(me)}"
            )
        except Exception as e:
            # Log error but don't crash - server continues running
            logger.error(f"❌ Error processing frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Clean up any remaining references
            if frame is not None:
                del frame
            if frame_array is not None:
                del frame_array
            gc.collect()
            # Return error response but server keeps running
            return killfeed_detection_pb2.ProcessFrameResponse(
                success=False,
                error_message=f"Error processing frame: {str(e)}"
            )
    
    def _process_frame(self, full_frame):
        """
        Process a full frame to detect killfeeds and determine their status.
        This method may take time, but server continues running.
        
        Args:
            full_frame: Full captured frame (numpy array, BGR format)
        
        Returns:
            list: List of detection results, each containing:
                - 'cropped_image': Cropped killfeed image
                - 'status': Status (gun knockout/kill/revive)
                - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
                - 'confidence': Detection confidence
                - 'detection_sources': List of detection sources for REVIVED
            Returns empty list if no killfeeds detected
        """
        if self.model is None or full_frame is None or full_frame.size == 0:
            return []
        
        try:
            # Detect killblocks in the frame (may take time, but server keeps running)
            detections = self._detect_killblocks(full_frame)
            if not detections:
                return []
            
            results = []
            for detection in detections:
                try:
                    x1, y1, x2, y2 = detection['bbox']
                    cropped_image = full_frame[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        continue
                    
                    # Use YOLO class for status when a specific class is detected.
                    # Only fall back to color analysis for generic "killblock".
                    yolo_class = detection.get('class_name', 'killblock')
                    status, detection_sources = self._determine_status_from_class(
                        full_frame, cropped_image, yolo_class
                    )
                    
                    results.append({
                        'cropped_image': cropped_image,
                        'status': status,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'detection_sources': detection_sources
                    })
                except Exception as e:
                    # Log error but continue processing other detections
                    logger.warning(f"⚠️ Error processing detection: {e}, continuing...")
                    continue
            
            return results
        except Exception as e:
            # Log error but don't crash - return empty list and server continues
            logger.error(f"⚠️ Error in process_frame: {e}, returning empty results")
            return []
    
    # All row-detection classes from best (1).pt that indicate a killfeed row.
    _ROW_CLASSES = frozenset({
        "player-skill-kill", "player-skill-knock", "falling-knock",
        "grenade-kill", "grenade-knock", "gun-head-kill", "gun-head-knock",
        "gun-kill", "gun-knock", "killblock", "landmine-knock", "player-kill",
        "playzone-kill", "playzone-knock", "revive",
        "smoke-grenade-kill", "smoke-grenade-knock",
    })

    def _detect_killblocks(self, image):
        """Detect killfeed rows using YOLO model (all row classes, not just killblock)."""
        if self.model is None:
            return None
        
        try:
            processed_image = cv2.resize(image, (1280, 720))
            names_map = getattr(self.model, 'names', None)
            if names_map is None:
                return None
            
            try:
                # Use lower confidence to match config (0.20) instead of legacy 0.65
                results = self.model.predict(
                    processed_image, 
                    conf=0.20, 
                    verbose=False, 
                    device='cpu',
                    imgsz=640,  # Fixed size for faster processing
                    half=False  # Disable half precision for CPU
                )
            except Exception as e:
                logger.warning(f"⚠️ YOLO predict error: {e}")
                return None
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return None
            
            detections = []
            orig_size = image.shape[:2]
            
            for box in result.boxes:
                try:
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = names_map.get(class_id, f"class_{class_id}") if isinstance(names_map, dict) else f"class_{class_id}"
                    
                    # Accept ALL row classes from the model, not just killblock
                    if class_name.lower() in self._ROW_CLASSES and confidence >= 0.20:
                        bbox_resized = box.xyxy[0].cpu().numpy()
                        bbox = self._convert_bbox_coords(bbox_resized, orig_size, (1280, 720))
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_name': class_name.lower(),
                        })
                except Exception as e:
                    logger.warning(f"⚠️ Error processing box: {e}")
                    continue
            
            return detections if detections else None
        except Exception as e:
            logger.error(f"⚠️ YOLO detection error: {e}")
            return None
    
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
    
    # YOLO class → gRPC status mapping (mirrors tms_map.py vocabulary).
    _CLASS_TO_STATUS = {
        "gun-kill": "kill",
        "gun-knock": "gun knockout",
        "player-kill": "kill",
        "gun-head-kill": "gun-head-kill",
        "gun-head-knock": "gun-head-knock",
        "player-skill-kill": "player-skill-kill",
        "player-skill-knock": "gun knockout",
        "grenade-kill": "grenade-kill",
        "grenade-knock": "grenade-knock",
        "smoke-grenade-kill": "smoke-grenade-kill",
        "smoke-grenade-knock": "smoke-grenade-knock",
        "playzone-kill": "playzone-kill",
        "playzone-knock": "playzone-knock",
        "falling-knock": "gun knockout",
        "landmine-knock": "grenade-knock",
        "revive": "revive",
    }

    def _determine_status_from_class(self, full_frame, cropped_image, yolo_class):
        """Status from best (1).pt YOLO class only — no color fallback."""
        del full_frame, cropped_image
        from killfeed.yolo_classes import yolo_class_to_tms_status

        status = yolo_class_to_tms_status(yolo_class)
        if status:
            cls = (yolo_class or "").lower()
            logger.info(f"✅ STATUS: {status} (YOLO class={cls})")
            return status, [f"YOLO: {cls}"]
        logger.debug("⚪ No confident YOLO class for status (generic killblock or unknown)")
        return "unknown", ["no_confident_yolo_class"]
    
    def _determine_status(self, full_frame, cropped_image):
        """
        Determine the status of a killfeed (KNOCKED/KILL/REVIVE).
        
        Returns:
            tuple: (status, detection_sources)
                - status: "KNOCKED", "KILL", or "REVIVE"
                - detection_sources: List of sources that detected revive (empty for gun knockout/kill)
        """
        # First, get color analysis to cross-validate revive detection
        color_status = self._analyze_victim_color(cropped_image)
        color_analysis = self._get_detailed_color_analysis(cropped_image)
        
        # PRIORITY: If color analysis strongly indicates KILL, prioritize it over revive
        red_ratio = color_analysis.get('red_ratio', 0.0)
        white_ratio = color_analysis.get('white_ratio', 0.0)
        
        # Strong kill detection - immediately reject revive and return kill
        if color_status == "kill":
            # Additional validation: ensure red signal is strong enough
            # Lower threshold to catch more kill cases (0.008 instead of 0.015)
            if red_ratio >= 0.008 and white_ratio < 0.010:
                logger.info(f"✅ STATUS: kill (strong red signal detected, red: {red_ratio:.4f}, white: {white_ratio:.4f}) - revive rejected")
                return "kill", []
        
        # Check for revive with enhanced validation
        revive_result = self._detect_revive_with_validation(cropped_image, full_frame, color_analysis)
        
        if revive_result['is_revive']:
            # Cross-validate: if color strongly suggests knocked/kill, reject revive
            if color_status in ["gun knockout", "kill"]:
                # Check if color signals are strong enough to override revive
                # STRONGER thresholds for kill - be more aggressive
                
                # For KILL: Lower threshold to catch more red signals
                if color_status == "kill":
                    # Any red signal above 0.008 with low white should reject revive
                    if red_ratio >= 0.008 and white_ratio < 0.010:
                        logger.warning(f"⚠️ Revive rejected - kill detected (red: {red_ratio:.4f}, white: {white_ratio:.4f})")
                        # Fall through to color-based detection
                    else:
                        # Revive confirmed with validation
                        status = "revive"
                        logger.info(f"✅ STATUS: revive (detected via {' + '.join(revive_result['sources'])}, confidence: {revive_result['confidence']:.2f}, validated: {revive_result['validation_passed']})")
                        return status, revive_result['sources']
                else:
                    # For gun knockout: existing logic
                    if (white_ratio > 0.020 and red_ratio < 0.005) or (red_ratio > 0.015 and white_ratio < 0.005):
                        logger.warning(f"⚠️ Revive rejected - strong color signal detected (white: {white_ratio:.4f}, red: {red_ratio:.4f})")
                        # Fall through to color-based detection
                    else:
                        # Revive confirmed with validation
                        status = "revive"
                        logger.info(f"✅ STATUS: revive (detected via {' + '.join(revive_result['sources'])}, confidence: {revive_result['confidence']:.2f}, validated: {revive_result['validation_passed']})")
                        return status, revive_result['sources']
            else:
                # Revive confirmed (no conflicting color signal)
                status = "revive"
                logger.info(f"✅ STATUS: revive (detected via {' + '.join(revive_result['sources'])}, confidence: {revive_result['confidence']:.2f}, validated: {revive_result['validation_passed']})")
                return status, revive_result['sources']
        
        # If not revive, analyze color for gun knockout/kill
        logger.debug("🔍 No revive detected, checking color for gun knockout/kill...")
        status = color_status
        
        if status not in ["gun knockout", "kill", "UNKNOWN"]:
            logger.warning(f"⚠️ Invalid status '{status}', defaulting to gun knockout")
            status = "gun knockout"
        
        if status == "UNKNOWN":
            logger.warning(f"⚠️ Status detection returned UNKNOWN, defaulting to gun knockout")
            status = "gun knockout"
        
        status = self._validate_status_consistency(status)
        color_name = {'gun knockout': 'WHITE', 'kill': 'RED'}.get(status, 'UNKNOWN')
        logger.info(f"✅ STATUS: {status} (victim name text color: {color_name})")
        
        return status, []
    
    def _get_detailed_color_analysis(self, cropped_image):
        """Get detailed color analysis metrics for cross-validation."""
        try:
            if cropped_image is None or cropped_image.size == 0:
                return {'white_ratio': 0.0, 'red_ratio': 0.0, 'green_ratio': 0.0}
            
            text_region = self._extract_victim_text_region(cropped_image)
            if text_region is None:
                return {'white_ratio': 0.0, 'red_ratio': 0.0, 'green_ratio': 0.0}
            
            text_region = self._preprocess_text_region(text_region)
            total_pixels = text_region.shape[0] * text_region.shape[1]
            if total_pixels == 0:
                return {'white_ratio': 0.0, 'red_ratio': 0.0, 'green_ratio': 0.0}
            
            hsv_image = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)
            
            # Red detection
            lower_red1 = np.array([0, 55, 65])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, 55, 65])
            upper_red2 = np.array([180, 255, 255])
            mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # White detection
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 18, 255])
            mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
            
            # Green detection (for revive indicators)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
            
            red_pixels = cv2.countNonZero(mask_red)
            white_pixels = cv2.countNonZero(mask_white)
            green_pixels = cv2.countNonZero(mask_green)
            
            red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0
            green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0
            
            return {
                'white_ratio': white_ratio,
                'red_ratio': red_ratio,
                'green_ratio': green_ratio
            }
        except Exception as e:
            logger.debug(f"⚠️ Color analysis error: {e}")
            return {'white_ratio': 0.0, 'red_ratio': 0.0, 'green_ratio': 0.0}
    
    def _detect_green_indicators(self, cropped_image):
        """Detect green color indicators that are common in revive killfeeds."""
        try:
            if cropped_image is None or cropped_image.size == 0:
                return False
            
            height, width = cropped_image.shape[:2]
            if height < 10 or width < 10:
                return False
            
            # Check multiple regions for green indicators
            hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
            
            # Green color range (broader for revive indicators)
            lower_green1 = np.array([40, 40, 40])
            upper_green1 = np.array([85, 255, 255])
            mask_green = cv2.inRange(hsv_image, lower_green1, upper_green1)
            
            # Also check for bright green (common in revive icons)
            lower_green2 = np.array([50, 100, 100])
            upper_green2 = np.array([75, 255, 255])
            mask_green_bright = cv2.inRange(hsv_image, lower_green2, upper_green2)
            
            mask_combined = cv2.bitwise_or(mask_green, mask_green_bright)
            green_pixels = cv2.countNonZero(mask_combined)
            total_pixels = height * width
            green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Revive killfeeds typically have some green indicators
            # Threshold is lower because green might be in icons, not text
            return green_ratio >= 0.005  # At least 0.5% green pixels
        except Exception as e:
            logger.debug(f"⚠️ Green detection error: {e}")
            return False
    
    def _validate_revive_spatial(self, cropped_image, revive_detections):
        """Validate that revive detections are in appropriate spatial locations."""
        try:
            if not revive_detections or cropped_image is None:
                return False
            
            height, width = cropped_image.shape[:2]
            if height < 30 or width < 30:
                return False
            
            # Revive indicators are typically in specific regions
            # Check if detections are in reasonable locations (not edge cases)
            valid_count = 0
            for detection in revive_detections:
                # If we have bbox info, validate it
                # For now, just check if we have multiple detections
                valid_count += 1
            
            # Require at least one valid detection
            return valid_count > 0
        except Exception as e:
            logger.debug(f"⚠️ Spatial validation error: {e}")
            return False
    
    def _detect_revive_with_validation(self, cropped_image, full_frame, color_analysis):
        """
        Enhanced revive detection with multiple validation layers.
        
        Returns:
            dict: {
                'is_revive': bool,
                'sources': list,
                'confidence': float,
                'validation_passed': bool
            }
        """
        result = {
            'is_revive': False,
            'sources': [],
            'confidence': 0.0,
            'validation_passed': False
        }
        
        # Step 1: YOLO-based detection with higher confidence threshold
        revive_in_crop = self._process_yolo_revive_detection(cropped_image, min_confidence=0.15)
        revive_in_frame = self._process_yolo_revive_detection(full_frame, min_confidence=0.15, resize_to=(1280, 720))
        
        # Step 2: Collect detection sources
        if revive_in_crop:
            result['sources'].append("YOLO killblock crop")
            result['confidence'] += 0.5  # Increased weight
        
        if revive_in_frame:
            result['sources'].append("YOLO full frame")
            result['confidence'] += 0.4  # Increased weight
        
        # Step 3: Require stronger evidence (at least one detection with higher confidence)
        if not result['sources']:
            return result
        
        # Step 4: Color-based validation - STRONGER for kill detection
        white_ratio = color_analysis.get('white_ratio', 0.0)
        red_ratio = color_analysis.get('red_ratio', 0.0)
        green_ratio = color_analysis.get('green_ratio', 0.0)
        
        # Validation 1: Should NOT have strong white/red signals (those indicate knocked/kill)
        # STRONGER thresholds - be more sensitive to red (kill) signals
        has_strong_white = white_ratio > 0.015 and red_ratio < 0.003
        # LOWER threshold for red to catch more kill cases (0.008 instead of 0.012)
        has_strong_red = red_ratio > 0.008 and white_ratio < 0.008
        
        if has_strong_white or has_strong_red:
            logger.debug(f"⚠️ Revive validation failed - strong color signal (white: {white_ratio:.4f}, red: {red_ratio:.4f})")
            return result  # Reject revive
        
        # Additional check: if red is present and dominant, reject revive
        if red_ratio >= 0.010 and red_ratio > white_ratio * 1.5:
            logger.debug(f"⚠️ Revive validation failed - red dominant (red: {red_ratio:.4f}, white: {white_ratio:.4f})")
            return result  # Reject revive
        
        # Validation 2: Green indicators support revive (optional but helpful)
        green_detected = self._detect_green_indicators(cropped_image)
        if green_detected:
            result['confidence'] += 0.2
            result['sources'].append("Green indicators")
            logger.debug(f"✅ Green indicators detected (ratio: {green_ratio:.4f})")
        
        # Validation 3: Require minimum confidence threshold
        min_confidence_required = 0.6  # Increased from 0.3
        if result['confidence'] < min_confidence_required:
            logger.debug(f"⚠️ Revive validation failed - insufficient confidence ({result['confidence']:.2f} < {min_confidence_required})")
            return result
        
        # Validation 4: Require multiple sources for lower confidence detections
        if len(result['sources']) < 2 and result['confidence'] < 0.8:
            logger.debug(f"⚠️ Revive validation failed - need multiple sources (sources: {len(result['sources'])}, confidence: {result['confidence']:.2f})")
            return result
        
        # All validations passed
        result['is_revive'] = True
        result['validation_passed'] = True
        
        return result
    
    def _is_revive_class(self, class_name):
        """Check if class name indicates revive status."""
        if not class_name:
            return False
        class_name_lower = class_name.lower()
        return (class_name_lower in ["revive", "revive_icon", "reviveicon"] or
                "revive" in class_name_lower or
                class_name_lower.startswith("revive") or
                class_name_lower.endswith("revive"))
    
    def _validate_revive_detections(self, revive_detections, context="", min_confidence=0.15):
        """
        Validate revive detections with enhanced adaptive confidence thresholds.
        Stricter thresholds to reduce false positives.
        
        Args:
            revive_detections: List of (class_name, confidence) tuples
            context: Context string for logging
            min_confidence: Minimum confidence threshold (default 0.15, stricter than before)
        
        Returns:
            bool: True if revive is validated, False otherwise
        """
        if not revive_detections:
            return False
        
        # Filter by minimum confidence first
        filtered_detections = [d for d in revive_detections if d[1] >= min_confidence]
        if not filtered_detections:
            return False
        
        best_revive = max(filtered_detections, key=lambda x: x[1])
        class_name, confidence = best_revive
        
        # Very high confidence - very reliable (increased threshold)
        if confidence >= 0.30:
            if len(filtered_detections) > 1:
                second_best = sorted(filtered_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.18:
                    logger.info(f"✅ revive detected{context} - Strong multiple confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            logger.info(f"✅ revive detected{context} - Very high confidence (Class: {class_name}, Confidence: {confidence:.3f})")
            return True
        
        # High confidence - reliable (increased threshold)
        if confidence >= 0.25:
            if len(filtered_detections) > 1:
                second_best = sorted(filtered_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.16:
                    logger.info(f"✅ revive detected{context} - Multiple high-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            logger.info(f"✅ revive detected{context} - High confidence (Class: {class_name}, Confidence: {confidence:.3f})")
            return True
        
        # Medium-high confidence - require multiple confirmations (stricter)
        if confidence >= 0.20:
            if len(filtered_detections) >= 2:
                second_best = sorted(filtered_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.15:
                    logger.info(f"✅ revive detected{context} - Multiple medium-high confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            # Single detection needs higher confidence
            if confidence >= 0.23:
                logger.info(f"✅ revive detected{context} - Single medium-high confidence (Class: {class_name}, Confidence: {confidence:.3f})")
                return True
        
        # Medium confidence - require strong multiple confirmations
        if confidence >= 0.15:
            if len(filtered_detections) >= 2:
                second_best = sorted(filtered_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.15:  # Both must be at least 0.15
                    logger.info(f"✅ revive detected{context} - Multiple medium-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            # Single detection needs higher confidence
            if confidence >= 0.20:
                logger.info(f"✅ revive detected{context} - Single medium-high confidence (Class: {class_name}, Confidence: {confidence:.3f})")
                return True
        
        # Below minimum confidence threshold - reject
        logger.debug(f"⚠️ Revive detection rejected{context} - confidence too low (best: {confidence:.3f}, required: {min_confidence})")
        return False
    
    def _process_yolo_revive_detection(self, image, resize_to=None, min_confidence=0.15):
        """
        Process YOLO revive detection on image with optional resize.
        
        Args:
            image: Input image (numpy array)
            resize_to: Optional tuple (width, height) to resize image
            min_confidence: Minimum confidence threshold for detections (default 0.15, stricter)
        
        Returns:
            bool: True if revive is detected and validated, False otherwise
        """
        if self.model is None or image is None or image.size == 0:
            return False
        
        height, width = image.shape[:2]
        if height < 30 or width < 30:
            return False
        
        if resize_to:
            processed_image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)
        elif height > 640 or width > 640:
            if width > height:
                new_width, new_height = 640, int(height * 640 / width)
            else:
                new_height, new_width = 640, int(width * 640 / height)
            processed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            processed_image = image
        
        try:
            # Use higher confidence threshold for initial filtering
            # Still scan with lower threshold but validate strictly
            results = self.model.predict(processed_image, conf=min_confidence * 0.6, verbose=False, device='cpu')
        except Exception as e:
            logger.debug(f"⚠️ YOLO predict error: {e}")
            return False
        
        if not results or len(results) == 0:
            return False
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return False
        
        names_map = getattr(self.model, 'names', None)
        revive_detections = []
        
        for box in result.boxes:
            try:
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = names_map.get(class_id, f"class_{class_id}") if names_map else f"class_{class_id}"
                
                # Collect all revive class detections (validation will filter by confidence)
                if self._is_revive_class(class_name):
                    revive_detections.append((class_name, confidence))
            except Exception as e:
                continue
        
        # Validate with stricter thresholds
        context = f" (min_conf: {min_confidence:.2f})"
        return self._validate_revive_detections(revive_detections, context=context, min_confidence=min_confidence)
    
    def _detect_revive_status(self, image):
        """revive detection using YOLOv11."""
        try:
            return self._process_yolo_revive_detection(image)
        except Exception as e:
            return False
    
    def _detect_revive_in_frame(self, frame):
        """Check entire frame for revive class from YOLO model."""
        try:
            return self._process_yolo_revive_detection(frame, resize_to=(1280, 720))
        except Exception as e:
            return False
    
    def _validate_status_consistency(self, status):
        """Final validation to ensure status consistency."""
        if status is None:
            return "gun knockout"
        
        status_lower = status.lower().strip()
        valid_statuses = ["revive", "gun knockout", "kill"]
        
        if status_lower not in valid_statuses:
            logger.warning(f"⚠️ Invalid status '{status}', defaulting to gun knockout")
            return "gun knockout"
        
        if status_lower == "revive":
            return "revive"
        
        if status_lower in ["gun knockout", "kill"]:
            return status_lower
        
        return "gun knockout"
    
    def _preprocess_text_region(self, image):
        """Preprocess image to enhance text visibility."""
        try:
            if image is None or image.size == 0:
                return image
            
            processed = image.copy()
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return processed
        except Exception as e:
            return image
    
    def _extract_region(self, image, x_start, x_end, y_start, y_end):
        """Extract a region from image with bounds checking."""
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            return None
        region = image[y_start:y_end, x_start:x_end]
        if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
            return None
        return region
    
    def _extract_victim_text_region(self, cropped_image):
        """Extract the precise victim name text region for color analysis."""
        try:
            height, width = cropped_image.shape[:2]
            return self._extract_region(cropped_image, int(width * 0.50), width, 
                                       int(height * 0.20), int(height * 0.80))
        except Exception as e:
            return None
    
    def _analyze_victim_color(self, cropped_image):
        """Analyzes victim name text color for gun knockout/kill status with enhanced accuracy."""
        try:
            if cropped_image is None or cropped_image.size == 0:
                return "UNKNOWN"
            
            height, width = cropped_image.shape[:2]
            if height < 10 or width < 10:
                return "UNKNOWN"
            
            text_region = self._extract_victim_text_region(cropped_image)
            if text_region is None:
                return "UNKNOWN"
            
            text_region = self._preprocess_text_region(text_region)
            
            total_pixels = text_region.shape[0] * text_region.shape[1]
            if total_pixels == 0:
                return "UNKNOWN"
            
            hsv_image = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)
            
            # Sharper red detection ranges - stricter for kill
            lower_red1_primary = np.array([0, 70, 80])
            upper_red1_primary = np.array([12, 255, 255])
            lower_red2_primary = np.array([168, 70, 80])
            upper_red2_primary = np.array([180, 255, 255])
            
            lower_red1_relaxed = np.array([0, 55, 65])
            upper_red1_relaxed = np.array([15, 255, 255])
            lower_red2_relaxed = np.array([165, 55, 65])
            upper_red2_relaxed = np.array([180, 255, 255])
            
            # Sharper white detection - stricter for gun knockout
            lower_white_strict = np.array([0, 0, 220])
            upper_white_strict = np.array([180, 12, 255])
            lower_white_relaxed = np.array([0, 0, 200])
            upper_white_relaxed = np.array([180, 18, 255])
            
            # Create masks
            mask_red1_primary = cv2.inRange(hsv_image, lower_red1_primary, upper_red1_primary)
            mask_red2_primary = cv2.inRange(hsv_image, lower_red2_primary, upper_red2_primary)
            mask_red_primary = cv2.bitwise_or(mask_red1_primary, mask_red2_primary)
            
            mask_red1_relaxed = cv2.inRange(hsv_image, lower_red1_relaxed, upper_red1_relaxed)
            mask_red2_relaxed = cv2.inRange(hsv_image, lower_red2_relaxed, upper_red2_relaxed)
            mask_red_relaxed = cv2.bitwise_or(mask_red1_relaxed, mask_red2_relaxed)
            
            mask_red_combined = cv2.bitwise_or(mask_red_primary, mask_red_relaxed)
            mask_white_strict = cv2.inRange(hsv_image, lower_white_strict, upper_white_strict)
            mask_white_relaxed = cv2.inRange(hsv_image, lower_white_relaxed, upper_white_relaxed)
            mask_white_combined = cv2.bitwise_or(mask_white_strict, mask_white_relaxed)
            
            # Exclude overlapping pixels
            mask_red_final = cv2.bitwise_and(mask_red_combined, cv2.bitwise_not(mask_white_combined))
            mask_white_final = cv2.bitwise_and(mask_white_combined, cv2.bitwise_not(mask_red_combined))
            
            red_pixels = cv2.countNonZero(mask_red_final)
            white_pixels = cv2.countNonZero(mask_white_final)
            white_strict_pixels = cv2.countNonZero(mask_white_strict)
            
            red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0
            white_strict_ratio = white_strict_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Optimized RGB sampling - faster with better coverage
            bgr_image = text_region.copy()
            h, w = bgr_image.shape[:2]
            sample_step = max(2, min(h // 15, w // 15))
            red_rgb_pixels = 0
            white_rgb_pixels = 0
            total_samples = 0
            
            for y in range(0, h, sample_step):
                for x in range(0, w, sample_step):
                    b, g, r = bgr_image[y, x]
                    brightness = (r + g + b) / 3
                    total_samples += 1
                    
                    # Stricter red detection
                    if r > g + 35 and r > b + 35 and r > 110 and brightness < 240:
                        red_rgb_pixels += 1
                    # Stricter white detection
                    elif brightness > 210 and abs(r - g) < 15 and abs(g - b) < 15 and abs(r - b) < 15:
                        white_rgb_pixels += 1
            
            red_rgb_ratio = red_rgb_pixels / total_samples if total_samples > 0 else 0.0
            white_rgb_ratio = white_rgb_pixels / total_samples if total_samples > 0 else 0.0
            
            logger.debug(f"🔍 HSV - Red: {red_ratio:.4f} ({red_pixels}px), White: {white_ratio:.4f} ({white_pixels}px), WhiteStrict: {white_strict_ratio:.4f}")
            logger.debug(f"🔍 RGB - Red: {red_rgb_ratio:.4f}, White: {white_rgb_ratio:.4f}")
            
            # Early exit: Clear gun knockout (strong white, no red)
            if white_strict_ratio >= 0.015 and red_ratio < 0.003:
                logger.info(f"✅ gun knockout - Strong white signal, no red (white: {white_strict_ratio:.4f}, red: {red_ratio:.4f})")
                return "gun knockout"
            
            if white_ratio >= 0.025 and red_ratio < 0.005:
                logger.info(f"✅ gun knockout - High white ratio, minimal red (white: {white_ratio:.4f}, red: {red_ratio:.4f})")
                return "gun knockout"
            
            # Early exit: Clear kill (strong red, no white)
            if red_ratio >= 0.020 and white_ratio < 0.005:
                logger.info(f"✅ kill - Strong red signal, no white (red: {red_ratio:.4f}, white: {white_ratio:.4f})")
                return "kill"
            
            # RGB boost for red detection
            rgb_red_boost = 0.0
            if red_rgb_ratio > 0.08:
                rgb_red_boost = 0.008
            elif red_rgb_ratio > 0.05:
                rgb_red_boost = 0.005
            elif red_rgb_ratio > 0.03:
                rgb_red_boost = 0.003
            
            red_ratio_adjusted = red_ratio + rgb_red_boost
            
            # Stricter thresholds for better accuracy
            red_threshold_primary = 0.018
            white_threshold_primary = 0.025
            dominance_ratio_primary = 3.0
            
            red_threshold_secondary = 0.012
            white_threshold_secondary = 0.018
            dominance_ratio_secondary = 2.5
            
            red_threshold_tertiary = 0.008
            white_threshold_tertiary = 0.012
            dominance_ratio_tertiary = 2.2
            
            # Primary decision layer - stricter
            if red_ratio_adjusted >= red_threshold_primary:
                if white_ratio < 0.008 or red_ratio_adjusted >= white_ratio * dominance_ratio_primary:
                    if red_ratio_adjusted > white_ratio + 0.010:
                        if red_rgb_ratio > 0.04 or red_ratio_adjusted > 0.020:
                            dominance = red_ratio_adjusted / white_ratio if white_ratio > 0 else float('inf')
                            logger.info(f"✅ kill - Primary red signal (ratio: {red_ratio_adjusted:.4f}, dominance: {dominance:.2f}x)")
                            return "kill"
            
            if white_strict_ratio >= white_threshold_primary:
                if red_ratio < 0.005 or white_strict_ratio >= red_ratio * dominance_ratio_primary:
                    if white_strict_ratio > red_ratio + 0.015:
                        dominance = white_strict_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ gun knockout - Primary white strict signal (ratio: {white_strict_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "gun knockout"
            
            if white_ratio >= white_threshold_primary:
                if red_ratio < 0.006 or white_ratio >= red_ratio * dominance_ratio_primary:
                    if white_ratio > red_ratio + 0.012:
                        dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ gun knockout - Primary white signal (ratio: {white_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "gun knockout"
            
            # Secondary decision layer
            if red_ratio_adjusted >= red_threshold_secondary:
                if white_ratio < 0.010 or red_ratio_adjusted >= white_ratio * dominance_ratio_secondary:
                    if red_ratio_adjusted > white_ratio * 2.0 and red_ratio_adjusted > white_ratio + 0.008:
                        if red_rgb_ratio > 0.05 or red_ratio_adjusted > 0.015:
                            dominance = red_ratio_adjusted / white_ratio if white_ratio > 0 else float('inf')
                            logger.info(f"✅ kill - Secondary red signal (ratio: {red_ratio_adjusted:.4f}, dominance: {dominance:.2f}x)")
                            return "kill"
            
            if white_strict_ratio >= white_threshold_secondary:
                if red_ratio < 0.008 or white_strict_ratio >= red_ratio * dominance_ratio_secondary:
                    if white_strict_ratio > red_ratio * 2.5 and white_strict_ratio > red_ratio + 0.010:
                        dominance = white_strict_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ gun knockout - Secondary white strict signal (ratio: {white_strict_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "gun knockout"
            
            if white_ratio >= white_threshold_secondary:
                if red_ratio < 0.010 or white_ratio >= red_ratio * dominance_ratio_secondary:
                    if white_ratio > red_ratio * 2.2 and white_ratio > red_ratio + 0.010:
                        dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ gun knockout - Secondary white signal (ratio: {white_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "gun knockout"
            
            # Tertiary decision layer - with RGB validation
            if red_ratio_adjusted >= red_threshold_tertiary:
                if white_ratio < 0.012 or red_ratio_adjusted >= white_ratio * dominance_ratio_tertiary:
                    if red_ratio_adjusted > white_ratio * 1.8 and red_ratio_adjusted > white_ratio + 0.006:
                        if red_rgb_ratio > 0.06 or (red_rgb_ratio > 0.04 and red_ratio_adjusted > 0.010):
                            dominance = red_ratio_adjusted / white_ratio if white_ratio > 0 else float('inf')
                            logger.info(f"✅ kill - Tertiary red signal (ratio: {red_ratio_adjusted:.4f}, dominance: {dominance:.2f}x)")
                            return "kill"
            
            if white_strict_ratio >= white_threshold_tertiary:
                if red_ratio < 0.010 or white_strict_ratio >= red_ratio * dominance_ratio_tertiary:
                    if white_strict_ratio > red_ratio * 2.0 and white_strict_ratio > red_ratio + 0.008:
                        if white_rgb_ratio > 0.05:
                            dominance = white_strict_ratio / red_ratio if red_ratio > 0 else float('inf')
                            logger.info(f"✅ gun knockout - Tertiary white strict signal (ratio: {white_strict_ratio:.4f}, dominance: {dominance:.2f}x)")
                            return "gun knockout"
            
            # Fallback with strict validation
            if white_strict_ratio >= 0.010 and red_ratio < 0.008:
                if white_rgb_ratio > 0.03:
                    logger.info(f"✅ gun knockout - Fallback white strict (white: {white_strict_ratio:.4f}, red: {red_ratio:.4f})")
                    return "gun knockout"
            
            if red_ratio_adjusted >= 0.010 and white_ratio < 0.010:
                if red_rgb_ratio > 0.05:
                    logger.info(f"✅ kill - Fallback red (red: {red_ratio_adjusted:.4f}, white: {white_ratio:.4f})")
                    return "kill"
            
            # Final decision with sharper thresholds to reduce ambiguous cases
            # Use tighter ratios and absolute differences for better discrimination
            white_dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
            red_dominance = red_ratio_adjusted / white_ratio if white_ratio > 0 else float('inf')
            absolute_diff = abs(white_ratio - red_ratio_adjusted)
            
            # Sharper decision: prefer whichever has both higher ratio AND absolute difference
            if white_ratio >= 0.008 and (white_dominance >= 1.3 or (white_ratio > red_ratio_adjusted + 0.003)):
                # White is clearly dominant
                logger.info(f"✅ gun knockout - Final decision (white dominant: {white_ratio:.4f} vs {red_ratio:.4f}, dominance: {white_dominance:.2f}x)")
                return "gun knockout"
            elif red_ratio_adjusted >= 0.008 and (red_dominance >= 1.5 or (red_ratio_adjusted > white_ratio + 0.004)):
                # Red is clearly dominant - require RGB validation for kill
                if red_rgb_ratio > 0.03:  # Lowered threshold slightly
                    logger.info(f"✅ kill - Final decision (red dominant: {red_ratio_adjusted:.4f} vs {white_ratio:.4f}, dominance: {red_dominance:.2f}x)")
                    return "kill"
                else:
                    # Red detected but RGB validation failed - likely gun knockout with slight red tint
                    logger.info(f"✅ gun knockout - Red detected but RGB validation failed (red: {red_ratio_adjusted:.4f}, white: {white_ratio:.4f})")
                    return "gun knockout"
            else:
                # Still ambiguous - use more sophisticated fallback
                # Check RGB ratios and absolute values
                if white_rgb_ratio > red_rgb_ratio * 1.2 and white_ratio >= 0.006:
                    logger.info(f"✅ gun knockout - RGB-based decision (white RGB: {white_rgb_ratio:.4f} vs red RGB: {red_rgb_ratio:.4f})")
                    return "gun knockout"
                elif red_rgb_ratio > white_rgb_ratio * 1.5 and red_ratio_adjusted >= 0.006:
                    logger.info(f"✅ kill - RGB-based decision (red RGB: {red_rgb_ratio:.4f} vs white RGB: {white_rgb_ratio:.4f})")
                    return "kill"
                else:
                    # Last resort: use whichever is higher with minimum threshold
                    if white_ratio >= 0.006 and white_ratio > red_ratio_adjusted:
                        logger.info(f"✅ gun knockout - Fallback decision (white: {white_ratio:.4f} vs red: {red_ratio:.4f})")
                        return "gun knockout"
                    elif red_ratio_adjusted >= 0.006 and red_ratio_adjusted > white_ratio:
                        logger.info(f"✅ kill - Fallback decision (red: {red_ratio_adjusted:.4f} vs white: {white_ratio:.4f})")
                        return "kill"
                    else:
                        # Truly ambiguous - log but default to gun knockout (more common)
                        logger.warning(f"⚠️ Ambiguous - defaulting to gun knockout (Red: {red_ratio:.4f}, White: {white_ratio:.4f}, RedRGB: {red_rgb_ratio:.4f}, WhiteRGB: {white_rgb_ratio:.4f})")
                        return "gun knockout"
            
        except Exception as e:
            logger.error(f"⚠️ Color analysis error: {e}")
            return "UNKNOWN"


def serve(model_path="best (1).pt", port=50051, max_workers=10):
    """
    Start the gRPC server. Server runs continuously until user stops it (Ctrl+C).
    Handles long processing times gracefully and continues running.
    
    Args:
        model_path: Path to YOLO model file
        port: Port number for the server
        max_workers: Maximum number of concurrent workers
    """
    # Create gRPC server with increased max message size for large images
    options = [
        ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    
    # Add servicer
    servicer = KillfeedDetectionServicer(model_path=model_path)
    killfeed_detection_pb2_grpc.add_KillfeedDetectionServiceServicer_to_server(servicer, server)
    
    # Listen on port (use 0.0.0.0 for IPv4 compatibility on Windows)
    listen_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(listen_addr)
    
    # Start server
    try:
        server.start()
        logger.info(f"🚀 gRPC server started on port {port}")
        logger.info(f"📡 Listening for requests at {listen_addr}")
        logger.info(f"⏳ Server will run continuously until stopped (Ctrl+C)")
        logger.info(f"💡 Processing may take time, but server will keep running...")
        # Flush output to ensure messages are visible
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Keep server running indefinitely until user stops it
    # Server handles timeouts gracefully and continues processing
    logger.info("🔄 Server loop started - will run until stopped (Ctrl+C)")
    logger.info("💡 Server is ready to accept connections. Run your client code now.")
    
    # Use a robust loop that keeps the server running FOREVER
    # wait_for_termination raises FutureTimeoutError on timeout (normal) and returns None when terminated
    consecutive_errors = 0
    max_consecutive_errors = 10000  # Very high limit - server should never stop on errors
    gc_counter = 0  # Counter for periodic garbage collection
    
    try:
        while True:
            try:
                # Periodic garbage collection every 100 iterations to prevent memory buildup
                gc_counter += 1
                if gc_counter >= 100:
                    gc.collect()
                    gc_counter = 0
                
                # Wait for termination with timeout
                # This raises a timeout exception when timeout expires (normal - server still running)
                # Returns None when server is actually terminated
                result = server.wait_for_termination(timeout=1.0)
                
                # If we get here without exception, server was terminated
                if result is None:
                    logger.info("🛑 Server was terminated")
                    break
                    
            except KeyboardInterrupt:
                # User pressed Ctrl+C - graceful shutdown
                logger.info("\n⏹️ User requested shutdown (Ctrl+C)")
                logger.info("🛑 Shutting down gRPC server gracefully...")
                try:
                    server.stop(0)
                    logger.info("✅ Server stopped")
                except Exception as e:
                    logger.error(f"⚠️ Error during shutdown: {e}")
                break
                
            except Exception as e:
                # Check if it's a timeout exception (normal case - server still running)
                error_name = type(e).__name__
                error_str = str(e).lower()
                if 'Timeout' in error_name or 'timeout' in error_str:
                    # Timeout is NORMAL - it means server is still running
                    consecutive_errors = 0  # Reset error counter on successful iteration
                    continue  # Continue the loop - server is fine
                
                # Not a timeout - this is an actual error
                # Unexpected error - log but continue running
                consecutive_errors += 1
                logger.warning(f"⚠️ Error in server loop (error #{consecutive_errors}, continuing): {e}")
                
                # Only log full traceback for first few errors to avoid spam
                if consecutive_errors <= 3:
                    import traceback
                    logger.debug(traceback.format_exc())
                
                # Never stop on errors - keep running indefinitely
                # Only log warning if errors are very high, but continue running
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(f"⚠️ High error count ({consecutive_errors}), but server will continue running")
                    logger.warning("💡 Server will keep running - errors are being handled gracefully")
                    # Reset counter to prevent spam, but keep running
                    consecutive_errors = max_consecutive_errors - 10  # Reset to allow more errors
                    # Continue running - never break
                
                # Continue running - server should not stop on errors
                time.sleep(0.5)  # Small delay before continuing to avoid tight error loop
                continue
                
    except KeyboardInterrupt:
        # Handle Ctrl+C at outer level too
        logger.info("\n⏹️ User requested shutdown (Ctrl+C)")
        logger.info("🛑 Shutting down gRPC server gracefully...")
        try:
            server.stop(0)
            logger.info("✅ Server stopped")
        except Exception as e:
            logger.error(f"⚠️ Error during shutdown: {e}")
    
    # Final cleanup
    logger.info("🔄 Server loop ended")


def start_server_in_background(model_path="best (1).pt", port=50051):
    """
    Start the gRPC server in a background process.
    
    Args:
        model_path: Path to YOLO model file
        port: Port number for the server
    
    Returns:
        subprocess.Popen: Process object for the server
    """
    try:
        # Start server as a subprocess
        process = subprocess.Popen(
            [sys.executable, __file__, '--serve', '--model', model_path, '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"🚀 Started gRPC server in background (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"❌ Failed to start server in background: {e}")
        return None


if __name__ == '__main__':
    import argparse
    
    # Check if gRPC code is generated
    if killfeed_detection_pb2 is None or killfeed_detection_pb2_grpc is None:
        print("❌ gRPC code not generated!", file=sys.stderr)
        print("Please run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. killfeed_detection.proto", file=sys.stderr)
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='gRPC Killfeed Detection Server')
    parser.add_argument('--serve', action='store_true', help='Start the gRPC server')
    parser.add_argument('--model', default='best (1).pt', help='Path to YOLO model file (best (1).pt)')
    parser.add_argument('--port', type=int, default=50051, help='Port number for the server')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of concurrent workers')
    
    args = parser.parse_args()
    
    if args.serve:
        try:
            serve(model_path=args.model, port=args.port, max_workers=args.max_workers)
        except Exception as e:
            print(f"❌ Fatal error in server: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        print("Use --serve flag to start the server")
        print("Example: python grpc_block.py --serve --model \"best (1).pt\" --port 50051")
