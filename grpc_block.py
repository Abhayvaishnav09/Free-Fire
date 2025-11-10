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
    
    def __init__(self, model_path="best.pt"):
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
        try:
            # Decode image from bytes
            frame_array = np.frombuffer(request.frame_image, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None or frame.size == 0:
                logger.warning("⚠️ Failed to decode frame image")
                return killfeed_detection_pb2.ProcessFrameResponse(
                    success=False,
                    error_message="Failed to decode frame image"
                )
            
            # Process frame for detections (this may take time, but server keeps running)
            logger.debug(f"🔄 Processing frame (size: {frame.shape})...")
            detections = self._process_frame(frame)
            
            # Build response
            response = killfeed_detection_pb2.ProcessFrameResponse(
                success=True,
                detections=[]
            )
            
            for detection in detections:
                try:
                    # Encode cropped image to bytes
                    _, cropped_encoded = cv2.imencode('.png', detection['cropped_image'])
                    cropped_bytes = cropped_encoded.tobytes()
                    
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
                except Exception as e:
                    logger.warning(f"⚠️ Error encoding detection result: {e}, skipping...")
                    continue
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Processed frame: {len(response.detections)} detections in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            # Log error but don't crash - server continues running
            logger.error(f"❌ Error processing frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
                - 'status': Status (KNOCKED/KILL/REVIVED)
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
                    
                    # Determine status (may take time for color analysis, but server continues)
                    status, detection_sources = self._determine_status(full_frame, cropped_image)
                    
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
    
    def _detect_killblocks(self, image):
        """Detect killblocks using YOLO model."""
        if self.model is None:
            return None
        
        try:
            processed_image = cv2.resize(image, (1280, 720))
            names_map = getattr(self.model, 'names', None)
            if names_map is None:
                return None
            
            try:
                # Use faster inference settings
                results = self.model.predict(
                    processed_image, 
                    conf=0.65, 
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
                    
                    if class_name.lower() == "killblock" and confidence >= 0.65:
                        bbox_resized = box.xyxy[0].cpu().numpy()
                        bbox = self._convert_bbox_coords(bbox_resized, orig_size, (1280, 720))
                        detections.append({'bbox': bbox, 'confidence': confidence})
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
    
    def _determine_status(self, full_frame, cropped_image):
        """
        Determine the status of a killfeed (KNOCKED/KILL/REVIVED).
        
        Returns:
            tuple: (status, detection_sources)
                - status: "KNOCKED", "KILL", or "REVIVED"
                - detection_sources: List of sources that detected REVIVED (empty for KNOCKED/KILL)
        """
        # Check for REVIVED first (highest priority)
        revive_in_crop = self._detect_revive_status(cropped_image)
        revive_in_frame = self._detect_revive_in_frame(full_frame)
        
        detection_sources = []
        revive_confidence_score = 0.0
        
        if revive_in_crop:
            detection_sources.append("YOLO killblock crop")
            revive_confidence_score += 0.4
        
        if revive_in_frame:
            detection_sources.append("YOLO full frame")
            revive_confidence_score += 0.3
        
        # REVIVED takes absolute priority
        if detection_sources and revive_confidence_score >= 0.3:
            status = "REVIVED"
            logger.info(f"✅ STATUS: REVIVED (detected via {' + '.join(detection_sources)}, confidence: {revive_confidence_score:.2f})")
            return status, detection_sources
        
        # If not REVIVED, analyze color for KNOCKED/KILL
        logger.debug("🔍 No REVIVED detected, checking color for KNOCKED/KILL...")
        status = self._analyze_victim_color(cropped_image)
        
        if status not in ["KNOCKED", "KILL", "UNKNOWN"]:
            logger.warning(f"⚠️ Invalid status '{status}', defaulting to KNOCKED")
            status = "KNOCKED"
        
        if status == "UNKNOWN":
            logger.warning(f"⚠️ Status detection returned UNKNOWN, defaulting to KNOCKED")
            status = "KNOCKED"
        
        status = self._validate_status_consistency(status)
        color_name = {'KNOCKED': 'WHITE', 'KILL': 'RED'}.get(status, 'UNKNOWN')
        logger.info(f"✅ STATUS: {status} (victim name text color: {color_name})")
        
        return status, []
    
    def _is_revive_class(self, class_name):
        """Check if class name indicates revive status."""
        if not class_name:
            return False
        class_name_lower = class_name.lower()
        return (class_name_lower in ["revive", "revived", "revive_icon", "reviveicon"] or
                "revive" in class_name_lower or
                class_name_lower.startswith("revive") or
                class_name_lower.endswith("revive"))
    
    def _validate_revive_detections(self, revive_detections, context=""):
        """Validate revive detections with enhanced adaptive confidence thresholds."""
        if not revive_detections:
            return False
        
        best_revive = max(revive_detections, key=lambda x: x[1])
        class_name, confidence = best_revive
        
        # High confidence - very reliable
        if confidence >= 0.25:
            if len(revive_detections) > 1:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.15:
                    logger.info(f"✅ REVIVED detected{context} - Strong multiple confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            logger.info(f"✅ REVIVED detected{context} - High confidence (Class: {class_name}, Confidence: {confidence:.3f})")
            return True
        
        # Medium-high confidence
        if confidence >= 0.20:
            if len(revive_detections) > 1:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.12:
                    logger.info(f"✅ REVIVED detected{context} - Multiple confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            logger.info(f"✅ REVIVED detected{context} - Medium-high confidence (Class: {class_name}, Confidence: {confidence:.3f})")
            return True
        
        # Medium confidence - require multiple confirmations
        if confidence >= 0.15:
            if len(revive_detections) > 1:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.10:
                    logger.info(f"✅ REVIVED detected{context} - Multiple medium-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            if confidence >= 0.18:
                logger.info(f"✅ REVIVED detected{context} - Single medium-high confidence (Class: {class_name}, Confidence: {confidence:.3f})")
                return True
        
        # Low confidence - require strong multiple confirmations
        if confidence >= 0.10:
            if len(revive_detections) >= 2:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.10:
                    logger.info(f"✅ REVIVED detected{context} - Multiple low-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            if confidence >= 0.13:
                logger.info(f"✅ REVIVED detected{context} - Single low-confidence (Class: {class_name}, Confidence: {confidence:.3f})")
                return True
        
        # Very low confidence - require multiple strong confirmations
        if confidence >= 0.08:
            if len(revive_detections) >= 3:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                third_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[2]
                if second_best[1] >= 0.08 and third_best[1] >= 0.08:
                    logger.info(f"✅ REVIVED detected{context} - Multiple very-low-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f}, Third: {third_best[1]:.3f})")
                    return True
            if len(revive_detections) >= 2:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.09:
                    logger.info(f"✅ REVIVED detected{context} - Strong pair confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
        
        return False
    
    def _process_yolo_revive_detection(self, image, resize_to=None):
        """Process YOLO revive detection on image with optional resize."""
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
            results = self.model.predict(processed_image, conf=0.08, verbose=False, device='cpu')
        except Exception as e:
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
                
                if self._is_revive_class(class_name) and confidence >= 0.08:
                    revive_detections.append((class_name, confidence))
            except Exception as e:
                continue
        
        return self._validate_revive_detections(revive_detections)
    
    def _detect_revive_status(self, image):
        """REVIVED detection using YOLOv11."""
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
            return "KNOCKED"
        
        status_upper = status.upper().strip()
        valid_statuses = ["REVIVED", "KNOCKED", "KILL"]
        
        if status_upper not in valid_statuses:
            logger.warning(f"⚠️ Invalid status '{status}', defaulting to KNOCKED")
            return "KNOCKED"
        
        if status_upper == "REVIVED":
            return "REVIVED"
        
        if status_upper in ["KNOCKED", "KILL"]:
            return status_upper
        
        return "KNOCKED"
    
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
        """Analyzes victim name text color for KNOCKED/KILL status with enhanced accuracy."""
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
            
            # Sharper red detection ranges - stricter for KILL
            lower_red1_primary = np.array([0, 70, 80])
            upper_red1_primary = np.array([12, 255, 255])
            lower_red2_primary = np.array([168, 70, 80])
            upper_red2_primary = np.array([180, 255, 255])
            
            lower_red1_relaxed = np.array([0, 55, 65])
            upper_red1_relaxed = np.array([15, 255, 255])
            lower_red2_relaxed = np.array([165, 55, 65])
            upper_red2_relaxed = np.array([180, 255, 255])
            
            # Sharper white detection - stricter for KNOCKED
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
            
            # Early exit: Clear KNOCKED (strong white, no red)
            if white_strict_ratio >= 0.015 and red_ratio < 0.003:
                logger.info(f"✅ KNOCKED - Strong white signal, no red (white: {white_strict_ratio:.4f}, red: {red_ratio:.4f})")
                return "KNOCKED"
            
            if white_ratio >= 0.025 and red_ratio < 0.005:
                logger.info(f"✅ KNOCKED - High white ratio, minimal red (white: {white_ratio:.4f}, red: {red_ratio:.4f})")
                return "KNOCKED"
            
            # Early exit: Clear KILL (strong red, no white)
            if red_ratio >= 0.020 and white_ratio < 0.005:
                logger.info(f"✅ KILL - Strong red signal, no white (red: {red_ratio:.4f}, white: {white_ratio:.4f})")
                return "KILL"
            
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
                            logger.info(f"✅ KILL - Primary red signal (ratio: {red_ratio_adjusted:.4f}, dominance: {dominance:.2f}x)")
                            return "KILL"
            
            if white_strict_ratio >= white_threshold_primary:
                if red_ratio < 0.005 or white_strict_ratio >= red_ratio * dominance_ratio_primary:
                    if white_strict_ratio > red_ratio + 0.015:
                        dominance = white_strict_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ KNOCKED - Primary white strict signal (ratio: {white_strict_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "KNOCKED"
            
            if white_ratio >= white_threshold_primary:
                if red_ratio < 0.006 or white_ratio >= red_ratio * dominance_ratio_primary:
                    if white_ratio > red_ratio + 0.012:
                        dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ KNOCKED - Primary white signal (ratio: {white_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "KNOCKED"
            
            # Secondary decision layer
            if red_ratio_adjusted >= red_threshold_secondary:
                if white_ratio < 0.010 or red_ratio_adjusted >= white_ratio * dominance_ratio_secondary:
                    if red_ratio_adjusted > white_ratio * 2.0 and red_ratio_adjusted > white_ratio + 0.008:
                        if red_rgb_ratio > 0.05 or red_ratio_adjusted > 0.015:
                            dominance = red_ratio_adjusted / white_ratio if white_ratio > 0 else float('inf')
                            logger.info(f"✅ KILL - Secondary red signal (ratio: {red_ratio_adjusted:.4f}, dominance: {dominance:.2f}x)")
                            return "KILL"
            
            if white_strict_ratio >= white_threshold_secondary:
                if red_ratio < 0.008 or white_strict_ratio >= red_ratio * dominance_ratio_secondary:
                    if white_strict_ratio > red_ratio * 2.5 and white_strict_ratio > red_ratio + 0.010:
                        dominance = white_strict_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ KNOCKED - Secondary white strict signal (ratio: {white_strict_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "KNOCKED"
            
            if white_ratio >= white_threshold_secondary:
                if red_ratio < 0.010 or white_ratio >= red_ratio * dominance_ratio_secondary:
                    if white_ratio > red_ratio * 2.2 and white_ratio > red_ratio + 0.010:
                        dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
                        logger.info(f"✅ KNOCKED - Secondary white signal (ratio: {white_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "KNOCKED"
            
            # Tertiary decision layer - with RGB validation
            if red_ratio_adjusted >= red_threshold_tertiary:
                if white_ratio < 0.012 or red_ratio_adjusted >= white_ratio * dominance_ratio_tertiary:
                    if red_ratio_adjusted > white_ratio * 1.8 and red_ratio_adjusted > white_ratio + 0.006:
                        if red_rgb_ratio > 0.06 or (red_rgb_ratio > 0.04 and red_ratio_adjusted > 0.010):
                            dominance = red_ratio_adjusted / white_ratio if white_ratio > 0 else float('inf')
                            logger.info(f"✅ KILL - Tertiary red signal (ratio: {red_ratio_adjusted:.4f}, dominance: {dominance:.2f}x)")
                            return "KILL"
            
            if white_strict_ratio >= white_threshold_tertiary:
                if red_ratio < 0.010 or white_strict_ratio >= red_ratio * dominance_ratio_tertiary:
                    if white_strict_ratio > red_ratio * 2.0 and white_strict_ratio > red_ratio + 0.008:
                        if white_rgb_ratio > 0.05:
                            dominance = white_strict_ratio / red_ratio if red_ratio > 0 else float('inf')
                            logger.info(f"✅ KNOCKED - Tertiary white strict signal (ratio: {white_strict_ratio:.4f}, dominance: {dominance:.2f}x)")
                            return "KNOCKED"
            
            # Fallback with strict validation
            if white_strict_ratio >= 0.010 and red_ratio < 0.008:
                if white_rgb_ratio > 0.03:
                    logger.info(f"✅ KNOCKED - Fallback white strict (white: {white_strict_ratio:.4f}, red: {red_ratio:.4f})")
                    return "KNOCKED"
            
            if red_ratio_adjusted >= 0.010 and white_ratio < 0.010:
                if red_rgb_ratio > 0.05:
                    logger.info(f"✅ KILL - Fallback red (red: {red_ratio_adjusted:.4f}, white: {white_ratio:.4f})")
                    return "KILL"
            
            # Final decision with bias toward KNOCKED (more common)
            if white_ratio > red_ratio * 1.5 and white_ratio >= 0.008:
                logger.info(f"✅ KNOCKED - Final decision (white dominant: {white_ratio:.4f} vs {red_ratio:.4f})")
                return "KNOCKED"
            elif red_ratio_adjusted > white_ratio * 1.8 and red_ratio_adjusted >= 0.008:
                if red_rgb_ratio > 0.04:
                    logger.info(f"✅ KILL - Final decision (red dominant: {red_ratio_adjusted:.4f} vs {white_ratio:.4f})")
                    return "KILL"
            else:
                logger.warning(f"⚠️ Ambiguous - defaulting to KNOCKED (Red: {red_ratio:.4f}, White: {white_ratio:.4f})")
                return "KNOCKED"
            
        except Exception as e:
            logger.error(f"⚠️ Color analysis error: {e}")
            return "UNKNOWN"


def serve(model_path="best.pt", port=50051, max_workers=10):
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
    
    # Listen on port
    listen_addr = f'[::]:{port}'
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
    
    try:
        # Keep server running indefinitely until user stops it
        # Server handles timeouts gracefully and continues processing
        while True:
            try:
                server.wait_for_termination(timeout=1.0)
                break  # Server was terminated
            except Exception as e:
                # Continue running even if there are errors
                # This ensures server keeps running through timeouts and errors
                continue
    except KeyboardInterrupt:
        logger.info("\n⏹️ User requested shutdown (Ctrl+C)")
        logger.info("🛑 Shutting down gRPC server gracefully...")
        server.stop(0)
        logger.info("✅ Server stopped")
    except Exception as e:
        logger.error(f"❌ Unexpected error in server: {e}")
        logger.info("🛑 Shutting down gRPC server...")
        server.stop(0)


def start_server_in_background(model_path="best.pt", port=50051):
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
    parser.add_argument('--model', default='best.pt', help='Path to YOLO model file')
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
        print("Example: python grpc_block.py --serve --model best.pt --port 50051")
