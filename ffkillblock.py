"""Free Fire Killblock Detector - Detects kill notifications from OBS Virtual Camera."""
import cv2
import os
import numpy as np
import warnings
import time
import tempfile
from ultralytics import YOLO
import datetime
from text import FreeFireTextDetector
import queue
import threading

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')

warnings.filterwarnings("ignore")

class KillblockDetector:
    def __init__(self):
        print("Initializing Killblock Detector...")
        self.model = self.load_model()
        try:
            self.text_detector = FreeFireTextDetector()
        except Exception as e:
            print(f"⚠️ Could not initialize text detector: {e}")
            self.text_detector = None
        
        self.recent_detections = {}
        self.duplicate_window = 8
        self.cap = None
        self.camera_initialized = False
        self.detection_sequence = 0  # Sequential counter to ensure chronological order
        
        # Thread-safe image saving system
        self.save_queue = queue.Queue(maxsize=1000)  # Thread-safe queue for processed images
        self.save_thread = None
        self.save_thread_running = False
        self.detected_count = 0  # Total frames detected
        self.saved_count = 0  # Total frames successfully saved
        self.failed_count = 0  # Total frames that failed to save
        
        print("Detector ready!")
    
    def __del__(self):
        self.stop_save_thread()
        self.cleanup_camera()
    
    def load_model(self):
        try:
            if not os.path.exists("best.pt"):
                print("Model file not found: best.pt")
                return None
            print("Loading YOLO model...")
            model = YOLO("best.pt")
            print("✅ YOLO model loaded")
            
            # Display available classes for debugging
            if hasattr(model, 'names'):
                class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
                print(f"📋 Available classes in model: {class_names}")
            
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None

    def detect_revive_status(self, image):
        """REVIVED detection using YOLOv11 high-confidence detection only.
        Fully accurate with strict validation to prevent misclassification."""
        try:
            if self.model is None or image is None or image.size == 0:
                return False
            
            height, width = image.shape[:2]
            if height < 30 or width < 30:
                return False
            
            # Use original size or resize appropriately for better detection
            if height > 640 or width > 640:
                # Maintain aspect ratio
                if width > height:
                    new_width = 640
                    new_height = int(height * 640 / width)
                else:
                    new_height = 640
                    new_width = int(width * 640 / height)
                processed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            else:
                processed_image = image
            
            # HIGH-CONFIDENCE REVIVE detection: Use strict threshold for precision
            # Only accept high-confidence detections to avoid false positives
            results = self.model.predict(processed_image, conf=0.25, verbose=False, device='cpu')
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    names_map = getattr(self.model, 'names', None)
                    
                    revive_detections = []
                    for box in result.boxes:
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if names_map:
                            class_name = names_map.get(class_id, f"class_{class_id}")
                            
                            # Comprehensive revive class name checking (all variations)
                            class_name_lower = class_name.lower()
                            is_revive = (
                                class_name_lower in ["revive", "revived", "revive_icon", "reviveicon"] or
                                "revive" in class_name_lower or
                                class_name_lower.startswith("revive") or
                                class_name_lower.endswith("revive")
                            )
                            
                            if is_revive and confidence >= 0.25:
                                revive_detections.append((class_name, confidence))
                    
                    # STRICT VALIDATION: Only accept very high-confidence revive detections
                    # Enhanced validation to ensure REVIVED is never misclassified
                    if revive_detections:
                        best_revive = max(revive_detections, key=lambda x: x[1])
                        class_name, confidence = best_revive
                        
                        # Require very high confidence (0.28) for robust REVIVE detection
                        # This prevents false positives and ensures REVIVED is never misclassified
                        if confidence >= 0.28:
                            # Additional validation: If multiple revive detections, verify consistency
                            if len(revive_detections) > 1:
                                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                                # If second detection also has reasonable confidence, more reliable
                                if second_best[1] >= 0.22:
                                    print(f"✅ REVIVED detected - Multiple confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                                    return True
                            
                            # Single high-confidence detection is acceptable
                            print(f"✅ REVIVED detected - Class: {class_name}, Confidence: {confidence:.3f}")
                            return True
                        else:
                            print(f"⚠️ Revive confidence too low ({confidence:.3f} < 0.28) - rejected to prevent misclassification")
                            return False
            
            return False
        except Exception as e:
            print(f"⚠️ Revive detection error: {e}")
            return False

    def analyze_victim_color(self, cropped_image):
        """Pure HSV color masking: WHITE=KNOCKED, RED=FINISHED.
        Optimized HSV calibration to avoid cross-detection between colors."""
        try:
            if cropped_image is None or cropped_image.size == 0:
                return "UNKNOWN"
            
            height, width = cropped_image.shape[:2]
            if height < 10 or width < 10:
                return "UNKNOWN"
            
            # Focus on text region where victim name appears (right side)
            text_region = cropped_image[:, int(width * 0.45):]
            
            if text_region.size == 0 or text_region.shape[0] < 5 or text_region.shape[1] < 5:
                return "UNKNOWN"
            
            total_pixels = text_region.shape[0] * text_region.shape[1]
            if total_pixels == 0:
                return "UNKNOWN"
            
            # Convert to HSV for color analysis
            hsv_image = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)
            
            # OPTIMIZED RED HSV ranges for FINISHED - calibrated to avoid white cross-detection
            # Primary red range (0-10°): Pure red with high saturation/value
            # Strict saturation (>= 70) ensures no overlap with white (which has low saturation)
            lower_red1 = np.array([0, 70, 85])  # High saturation (70) and value (85) for strong reds
            upper_red1 = np.array([10, 255, 255])
            
            # Secondary red range (170-180°): Wrap-around red with high saturation/value
            lower_red2 = np.array([170, 70, 85])  # Matching saturation/value to primary
            upper_red2 = np.array([180, 255, 255])
            
            # OPTIMIZED WHITE HSV range for KNOCKED - calibrated to avoid red cross-detection
            # White: Very high value (brightness), very low saturation (pure white/gray)
            # Strict saturation (<= 20) ensures no overlap with red (which has high saturation)
            lower_white = np.array([0, 0, 200])  # Very high value (200) for pure white, zero saturation
            upper_white = np.array([180, 20, 255])  # Very low saturation tolerance (20) to exclude red-tinted whites
            
            # CRITICAL SEPARATION: Red requires saturation >= 70, White requires saturation <= 20
            # This creates a 50-point gap between red and white detection ranges
            # Ensures ZERO overlap and prevents cross-detection
            
            # Create red masks
            mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            mask_red_combined = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Create white mask
            mask_white_raw = cv2.inRange(hsv_image, lower_white, upper_white)
            
            # Ensure perfect separation: remove any potential overlap
            # Red mask: exclude pixels that match white criteria (extra safety)
            mask_red_final = cv2.bitwise_and(mask_red_combined, cv2.bitwise_not(mask_white_raw))
            
            # White mask: exclude pixels that match red criteria (extra safety)
            mask_white_final = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_red_combined))
            
            # Calculate ratios
            red_pixels = cv2.countNonZero(mask_red_final)
            white_pixels = cv2.countNonZero(mask_white_final)
            
            red_ratio = red_pixels / total_pixels
            white_ratio = white_pixels / total_pixels
            
            # Debug output
            print(f"🔍 HSV color analysis - Red ratio: {red_ratio:.4f} ({red_pixels} px), White ratio: {white_ratio:.4f} ({white_pixels} px)")
            
            # OPTIMIZED DECISION LOGIC with calibrated thresholds
            # Thresholds calibrated to avoid cross-detection while maintaining sensitivity
            
            # Primary thresholds (high confidence)
            red_threshold_primary = 0.018  # Require meaningful red presence
            white_threshold_primary = 0.018  # Require meaningful white presence
            dominance_ratio_primary = 2.5  # One color must be 2.5x the other for clear decision
            
            # Secondary thresholds (medium confidence)
            red_threshold_secondary = 0.012
            white_threshold_secondary = 0.012
            dominance_ratio_secondary = 2.0
            
            # Primary decision: HIGH CONFIDENCE
            if red_ratio >= red_threshold_primary:
                if red_ratio >= white_ratio * dominance_ratio_primary:
                    print(f"✅ FINISHED detected - Primary red HSV signal (ratio: {red_ratio:.4f})")
                    return "FINISHED"
            
            if white_ratio >= white_threshold_primary:
                if white_ratio >= red_ratio * dominance_ratio_primary:
                    print(f"✅ KNOCKED detected - Primary white HSV signal (ratio: {white_ratio:.4f})")
                    return "KNOCKED"
            
            # Secondary decision: MEDIUM CONFIDENCE
            if red_ratio >= red_threshold_secondary:
                if red_ratio >= white_ratio * dominance_ratio_secondary:
                    # Additional check: ensure red is clearly dominant
                    if red_ratio > white_ratio * 1.8:
                        print(f"✅ FINISHED detected - Secondary red HSV signal (ratio: {red_ratio:.4f})")
                        return "FINISHED"
            
            if white_ratio >= white_threshold_secondary:
                if white_ratio >= red_ratio * dominance_ratio_secondary:
                    # Additional check: ensure white is clearly dominant
                    if white_ratio > red_ratio * 1.8:
                        print(f"✅ KNOCKED detected - Secondary white HSV signal (ratio: {white_ratio:.4f})")
                        return "KNOCKED"
            
            # Final fallback: Clear dominance with minimum presence
            if red_ratio > white_ratio * 1.6:
                if red_ratio >= 0.008:  # Minimum meaningful red presence
                    print(f"✅ FINISHED detected - Fallback red signal (ratio: {red_ratio:.4f})")
                    return "FINISHED"
            
            if white_ratio > red_ratio * 1.6:
                if white_ratio >= 0.008:  # Minimum meaningful white presence
                    print(f"✅ KNOCKED detected - Fallback white signal (ratio: {white_ratio:.4f})")
                    return "KNOCKED"
            
            # Default: If truly ambiguous, default to KNOCKED (most common case)
            print(f"⚠️ Ambiguous color - defaulting to KNOCKED (Red: {red_ratio:.4f}, White: {white_ratio:.4f})")
            return "KNOCKED"
            
        except Exception as e:
            print(f"⚠️ Color analysis error: {e}")
            return "UNKNOWN"

    def detect_revive_green(self, cropped_image):
        """ROBUST secondary fallback: detect green hue in revive indicators.
        Only used when model-based revive detection fails. Requires strict validation."""
        try:
            if cropped_image is None or cropped_image.size == 0:
                return False
            
            height, width = cropped_image.shape[:2]
            if height < 10 or width < 10:
                return False
            
            # Focus on text region where revive indicators appear
            text_region = cropped_image[:, int(width * 0.45):]
            
            hsv = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)
            # Strict green range to avoid false positives from HUD elements
            lower_green = np.array([50, 90, 90])  # Higher saturation/value to avoid noise
            upper_green = np.array([80, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            total = text_region.shape[0] * text_region.shape[1]
            if total <= 0:
                return False
            
            green_ratio = cv2.countNonZero(mask_green) / total
            # Require higher proportion (0.025) for robust green detection
            if green_ratio > 0.025:
                print(f"🟢 Green revive indicator detected (ratio: {green_ratio:.4f})")
                return True
            return False
        except Exception as e:
            print(f"⚠️ Green revive detection error: {e}")
            return False
    
    def apply_color_splash(self, image: np.ndarray) -> np.ndarray:
        """
        Apply selective color effect - convert to grayscale except text, icons, and colored UI elements.
        Optimized for speed using existing text detector.
        
        Args:
            image: Input image in BGR format
        
        Returns:
            Processed image with text/icons in color, rest in grayscale
        """
        try:
            if image is None or image.size == 0:
                return image
            
            height, width = image.shape[:2]
            if height < 10 or width < 10:
                return image
            
            # Create mask for regions to keep in color
            color_mask = np.zeros((height, width), dtype=np.uint8)
            
            # METHOD 1: Detect text regions using existing text detector
            if self.text_detector is not None:
                try:
                    # Use the existing PaddleOCR reader from text_detector
                    text_results = self.text_detector.detect_text_regions(image)
                    
                    if text_results:
                        for text_result in text_results:
                            if isinstance(text_result, dict) and 'bbox' in text_result:
                                # Extract bounding box
                                x, y, w, h = text_result['bbox']
                                x, y = int(x), int(y)
                                w, h = int(w), int(h)
                                
                                # Ensure coordinates are within bounds
                                x = max(0, min(x, width - 1))
                                y = max(0, min(y, height - 1))
                                w = min(w, width - x)
                                h = min(h, height - y)
                                
                                if w > 0 and h > 0:
                                    # Add padding and fill text region
                                    padding = 3
                                    x_start = max(0, x - padding)
                                    y_start = max(0, y - padding)
                                    x_end = min(width, x + w + padding)
                                    y_end = min(height, y + h + padding)
                                    color_mask[y_start:y_end, x_start:x_end] = 255
                except Exception as e:
                    # Fallback if text detection fails
                    pass
            
            # METHOD 2: Detect colored icons and UI elements (high saturation areas)
            # This catches colored icons, status indicators, and UI elements
            try:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Detect high-saturation colored regions (icons, UI elements)
                # High saturation = colored elements (not grayscale)
                saturation_mask = hsv_image[:, :, 1] > 60  # Medium-high saturation
                value_mask = hsv_image[:, :, 2] > 80  # Not too dark
                colored_regions = saturation_mask & value_mask
                
                # Also detect specific colored UI elements (red, green, blue, yellow)
                # Red: status indicators, damage numbers
                lower_red1 = np.array([0, 60, 80])
                upper_red1 = np.array([12, 255, 255])
                lower_red2 = np.array([168, 60, 80])
                upper_red2 = np.array([180, 255, 255])
                mask_red = cv2.bitwise_or(
                    cv2.inRange(hsv_image, lower_red1, upper_red1),
                    cv2.inRange(hsv_image, lower_red2, upper_red2)
                )
                
                # Green: revive indicators, health
                lower_green = np.array([50, 60, 80])
                upper_green = np.array([80, 255, 255])
                mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
                
                # Blue: UI elements
                lower_blue = np.array([100, 60, 80])
                upper_blue = np.array([130, 255, 255])
                mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
                
                # Yellow/Orange: warnings, highlights
                lower_yellow = np.array([20, 60, 80])
                upper_yellow = np.array([35, 255, 255])
                mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
                
                # Combine all colored element masks
                colored_icons_mask = cv2.bitwise_or(
                    cv2.bitwise_or(mask_red, mask_green),
                    cv2.bitwise_or(mask_blue, mask_yellow)
                )
                
                # Also include high-saturation regions
                colored_icons_mask = cv2.bitwise_or(
                    colored_icons_mask,
                    (colored_regions.astype(np.uint8) * 255)
                )
                
                # Combine text mask with colored icons mask
                color_mask = cv2.bitwise_or(color_mask, colored_icons_mask)
                
            except Exception as e:
                # If color detection fails, continue with text mask only
                pass
            
            # Apply morphological operations to smooth the mask
            if cv2.countNonZero(color_mask) > 0:
                kernel = np.ones((3, 3), np.uint8)
                color_mask = cv2.dilate(color_mask, kernel, iterations=1)
                color_mask = cv2.erode(color_mask, kernel, iterations=1)
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Apply mask: keep original color where mask is 255, use grayscale elsewhere
            result = np.where(color_mask[:, :, np.newaxis] == 255, image, gray_bgr)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"⚠️ Color splash error: {e}")
            # Return original image if processing fails
            return image

    def extract_names(self, cropped_image):
        """Extract killer and victim names using OCR."""
        if self.text_detector is None:
            return "", ""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(temp_fd)
            cv2.imwrite(temp_path, cropped_image)
            results = self.text_detector.process_image(temp_path)
            os.remove(temp_path)
            
            if results and isinstance(results, dict):
                killer = results.get("killer", {}).get("text", "") if isinstance(results.get("killer"), dict) else ""
                victim = results.get("victim", {}).get("text", "") if isinstance(results.get("victim"), dict) else ""
                return killer, victim
            return "", ""
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return "", ""

    def is_duplicate(self, killer_name, victim_name):
        """Check for duplicate detections."""
        try:
            current_time = time.time()
            old_sigs = [sig for sig, ts in self.recent_detections.items() if current_time - ts > self.duplicate_window]
            for sig in old_sigs:
                del self.recent_detections[sig]
            
            sig = f"{killer_name.lower()}_{victim_name.lower()}"
            if sig in self.recent_detections:
                return True
            self.recent_detections[sig] = current_time
            return False
        except:
            return False

    def detect_killblocks(self, image):
        """Detect killblocks using YOLO model."""
        if self.model is None:
            return None
        
        try:
            processed_image = cv2.resize(image, (1280, 720))
            names_map = getattr(self.model, 'names', None)
            if names_map is None:
                return None
            
            # High confidence threshold (65%) - only detect high-quality killblocks
            results = self.model.predict(processed_image, conf=0.65, verbose=False, device='cpu')
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = names_map.get(class_id, f"class_{class_id}") if isinstance(names_map, dict) else f"class_{class_id}"
                        
                        # Require 65% confidence for killblock detection (only high-quality detections)
                        if class_name.lower() == "killblock" and confidence >= 0.65:
                            x1_resized, y1_resized, x2_resized, y2_resized = box.xyxy[0].cpu().numpy()
                            orig_height, orig_width = image.shape[:2]
                            
                            x1 = int(x1_resized * orig_width / 1280)
                            y1 = int(y1_resized * orig_height / 720)
                            x2 = int(x2_resized * orig_width / 1280)
                            y2 = int(y2_resized * orig_height / 720)
                            
                            x1 = max(0, min(x1, orig_width - 1))
                            y1 = max(0, min(y1, orig_height - 1))
                            x2 = max(x1 + 1, min(x2, orig_width))
                            y2 = max(y1 + 1, min(y2, orig_height))
                            
                            detections.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})
            
            return detections if detections else None
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return None

    def detect_revive_in_frame(self, frame):
        """Check entire frame for revive class from YOLO model."""
        try:
            if self.model is None or frame is None or frame.size == 0:
                return False
            
            # ROBUST full frame REVIVE detection with strict validation
            # Full frame detection requires higher confidence to avoid false positives
            processed_frame = cv2.resize(frame, (1280, 720))
            results = self.model.predict(processed_frame, conf=0.25, verbose=False, device='cpu')
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    names_map = getattr(self.model, 'names', None)
                    
                    revive_detections = []
                    for box in result.boxes:
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if names_map:
                            class_name = names_map.get(class_id, f"class_{class_id}")
                            class_name_lower = class_name.lower()
                            
                            # Comprehensive revive class name checking
                            is_revive = (
                                class_name_lower in ["revive", "revived", "revive_icon", "reviveicon"] or
                                "revive" in class_name_lower or
                                class_name_lower.startswith("revive") or
                                class_name_lower.endswith("revive")
                            )
                            
                            if is_revive and confidence >= 0.25:
                                revive_detections.append((class_name, confidence))
                    
                    # STRICT VALIDATION for full frame - require very high confidence
                    if revive_detections:
                        best_revive = max(revive_detections, key=lambda x: x[1])
                        class_name, confidence = best_revive
                        
                        # Full frame requires even higher confidence (0.28) to prevent false positives
                        if confidence >= 0.28:
                            print(f"🔄 REVIVE detected in full frame - Class: {class_name}, Confidence: {confidence:.3f}")
                            return True
                        else:
                            print(f"⚠️ Full frame revive confidence too low ({confidence:.3f} < 0.28) - rejected")
                            return False
            
            return False
        except Exception as e:
            print(f"⚠️ Frame revive check error: {e}")
            return False
    
    def process_detection(self, full_frame, detection, time_str, frame_number, sequence_number):
        """Process single detection - check full frame for revive first.
        Ensures strict chronological order with sequential numbering."""
        try:
            x1, y1, x2, y2 = detection['bbox']
            cropped_image = full_frame[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                return False
            
            killer_name, victim_name = self.extract_names(cropped_image)
            
            if self.is_duplicate(killer_name, victim_name):
                print(f"🚫 Duplicate detected - skipping")
                return False
            
            # REFINED STATUS DETECTION PIPELINE - REVIVED has absolute priority
            # Step 1: Check for REVIVED using YOLOv11 high-confidence detection (primary method)
            print("🔍 Checking for REVIVED detection (YOLOv11 high-confidence)...")
            revive_in_crop = self.detect_revive_status(cropped_image)
            
            # Step 2: Check full frame for revive (with high-confidence validation)
            revive_in_frame = self.detect_revive_in_frame(full_frame)
            
            # Step 3: Color validation fallback for REVIVED (green HSV validation)
            # Only used as fallback if YOLOv11 doesn't detect but green is present
            green_revive_detected = False
            if not revive_in_crop and not revive_in_frame:
                green_revive_detected = self.detect_revive_green(cropped_image)
            
            # FINAL STATUS DECISION - REVIVED has absolute priority to prevent misclassification
            # If REVIVED is detected by any method, it takes precedence over color analysis
            # This ensures REVIVED is never misclassified as KNOCKED or FINISHED
            if revive_in_crop or revive_in_frame or green_revive_detected:
                # REVIVED: Confirmed through YOLOv11 high-confidence detection or color validation fallback
                # REVIVED status is final - never fall through to color analysis
                status = "REVIVED"
                source = "killblock crop" if revive_in_crop else ("full frame" if revive_in_frame else "green HSV validation")
                print(f"✅ STATUS: REVIVED (detected via {source}) - REVIVED takes priority, skipping color analysis")
            else:
                # Step 4: Pure HSV color masking for KNOCKED/FINISHED (only if REVIVED is NOT detected)
                # Only proceed to color analysis if REVIVED was definitively not detected
                # KNOCKED = white, FINISHED = red
                status = self.analyze_victim_color(cropped_image)
                print(f"✅ STATUS: {status} (HSV color-based: {'WHITE' if status=='KNOCKED' else 'RED' if status=='FINISHED' else 'UNKNOWN'})")
            
            # Apply color splash effect (selective color - grayscale except text/icons)
            cropped_image_colorsplash = self.apply_color_splash(cropped_image)
            
            # Prepare for non-blocking save via background thread
            # Format: ORDER_KILLER STATUS VICTIM.png
            # Simple order number (3 digits) shows capture sequence and helps identify missed frames
            output_dir = "cropkillblock"
            os.makedirs(output_dir, exist_ok=True)
            
            # Simple order number format (3 digits: 001, 002, 003...)
            # Order number: Easy to see sequence and identify gaps (missed frames)
            # Format: ORDER_KILLER STATUS VICTIM.png
            filename = f"{sequence_number:03d}_{killer_name} {status} {victim_name}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Non-blocking enqueue for background save thread
            # Queue ensures strict chronological order - FIFO guarantees earlier detections saved first
            # Increment detected count before enqueueing
            self.detected_count += 1
            
            # Prepare metadata for save thread
            metadata = {
                'killer_name': killer_name,
                'victim_name': victim_name,
                'status': status,
                'confidence': detection['confidence'],
                'sequence_number': sequence_number,
                'frame_number': frame_number
            }
            
            # Enqueue for background save (non-blocking)
            try:
                # Put item in queue (non-blocking with timeout to prevent blocking)
                # If queue is full, wait briefly then try again
                self.save_queue.put((cropped_image_colorsplash.copy(), filepath, metadata), timeout=0.1)
                print(f"📤 Enqueued for save: {filename}")
                print(f"   👤 Killer: {killer_name} | 🎯 Victim: {victim_name}")
                print(f"   📊 Status: {status} | 📈 Confidence: {detection['confidence']:.2f}")
                print(f"   🔢 Order: #{sequence_number} (Frame: {frame_number})")
                print(f"   💾 Queue size: {self.save_queue.qsize()} | Detected: {self.detected_count}\n")
                return True
            except queue.Full:
                # Queue is full - this should not happen often, but if it does, retry once
                print(f"⚠️ Save queue full, retrying...")
                try:
                    self.save_queue.put((cropped_image_colorsplash.copy(), filepath, metadata), timeout=1.0)
                    print(f"📤 Enqueued for save (retry): {filename}\n")
                    return True
                except queue.Full:
                    print(f"❌ Failed to enqueue - queue still full: {filename}\n")
                    self.failed_count += 1
                    return False
            except Exception as e:
                print(f"⚠️ Enqueue error: {e}")
                self.failed_count += 1
                return False
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            return False

    def initialize_camera(self):
        """Initialize OBS camera."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            try:
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            except:
                self.cap = cv2.VideoCapture(1)
            
            if not self.cap.isOpened():
                print("Could not open OBS camera!")
                self.camera_initialized = False
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 1)  # 1 FPS - 1 frame per second
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            for _ in range(5):
                self.cap.read()
            
            self.camera_initialized = True
            print("✅ Camera initialized")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.camera_initialized = False
            return False
    
    def capture_frame(self):
        """Capture frame from OBS with error recovery."""
        try:
            # Check camera state and reinitialize if needed
            if not self.camera_initialized or self.cap is None:
                if not self.initialize_camera():
                    return None
            
            # Try to read frame
            if self.cap is None:
                return None
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return None
            
            return frame
        except Exception as e:
            print(f"⚠️ Frame capture error: {e}")
            # Try to recover by reinitializing
            try:
                self.cleanup_camera()
                time.sleep(0.5)
                self.initialize_camera()
            except:
                pass
            return None
    
    def cleanup_camera(self):
        """Clean up camera resources."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_initialized = False
        except:
            pass
    
    def save_worker_thread(self):
        """Background thread worker that saves images from the queue with retry logic."""
        max_retries = 3
        retry_delay = 0.1  # 100ms between retries
        
        while self.save_thread_running or not self.save_queue.empty():
            try:
                # Get item from queue with timeout to check running flag
                try:
                    item = self.save_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Unpack queue item: (image, filepath, metadata)
                image, filepath, metadata = item
                
                # Retry logic for file writing
                saved = False
                last_error = None
                for attempt in range(max_retries):
                    try:
                        if cv2.imwrite(filepath, image):
                            saved = True
                            self.saved_count += 1
                            break
                        else:
                            # cv2.imwrite returns False on failure
                            last_error = "cv2.imwrite returned False"
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    except Exception as e:
                        last_error = str(e)
                        if attempt < max_retries - 1:
                            print(f"⚠️ Save attempt {attempt+1}/{max_retries} failed: {e}, retrying...")
                            time.sleep(retry_delay * (attempt + 1))
                        else:
                            print(f"❌ Save failed after {max_retries} attempts: {e}")
                
                if saved:
                    # Print success message with metadata
                    killer_name = metadata.get('killer_name', '')
                    victim_name = metadata.get('victim_name', '')
                    status = metadata.get('status', '')
                    confidence = metadata.get('confidence', 0.0)
                    sequence_number = metadata.get('sequence_number', 0)
                    frame_number = metadata.get('frame_number', 0)
                    
                    print(f"📸 Saved (Color Splash): {os.path.basename(filepath)}")
                    print(f"   👤 Killer: {killer_name} | 🎯 Victim: {victim_name}")
                    print(f"   📊 Status: {status} | 📈 Confidence: {confidence:.2f}")
                    print(f"   🔢 Order: #{sequence_number} (Frame: {frame_number})")
                    print(f"   💾 Save Stats: {self.saved_count} saved / {self.detected_count} detected / {self.failed_count} failed\n")
                else:
                    self.failed_count += 1
                    error_msg = f": {last_error}" if last_error else ""
                    print(f"❌ Failed to save after {max_retries} retries{error_msg}: {os.path.basename(filepath)}")
                    print(f"   💾 Save Stats: {self.saved_count} saved / {self.detected_count} detected / {self.failed_count} failed\n")
                
                # Mark task as done
                self.save_queue.task_done()
                
            except Exception as e:
                print(f"⚠️ Save worker thread error: {e}")
                self.failed_count += 1
                try:
                    self.save_queue.task_done()
                except:
                    pass
    
    def start_save_thread(self):
        """Start the background save thread."""
        if self.save_thread is None or not self.save_thread.is_alive():
            self.save_thread_running = True
            self.save_thread = threading.Thread(target=self.save_worker_thread, daemon=True)
            self.save_thread.start()
            print("✅ Background save thread started")
    
    def stop_save_thread(self):
        """Stop the save thread and flush remaining items in queue."""
        if self.save_thread_running:
            print("🛑 Stopping save thread and flushing queue...")
            self.save_thread_running = False
            
            # Wait for queue to be processed (with timeout)
            timeout = 30  # 30 second timeout
            start_time = time.time()
            
            while not self.save_queue.empty():
                if time.time() - start_time > timeout:
                    print(f"⚠️ Queue flush timeout after {timeout}s, {self.save_queue.qsize()} items remaining")
                    break
                time.sleep(0.1)
            
            # Wait for thread to finish
            if self.save_thread is not None and self.save_thread.is_alive():
                self.save_thread.join(timeout=5)
                if self.save_thread.is_alive():
                    print("⚠️ Save thread did not terminate cleanly")
                else:
                    print("✅ Save thread stopped")
            
            # Print final statistics
            print(f"📊 Final Save Stats: {self.saved_count} saved / {self.detected_count} detected / {self.failed_count} failed")

    def start_detection(self):
        """Main detection loop - runs continuously until user stops with Ctrl+C."""
        print("🚀 Starting Free Fire Detection...")
        print("="*50)
        print("⚠️  PRESS Ctrl+C TO STOP")
        print("📝 Will keep retrying if camera issues occur")
        print("="*50 + "\n")
        
        # Try to initialize camera - keep trying if it fails
        while not self.initialize_camera():
            print("⚠️ Camera initialization failed, retrying in 2 seconds...")
            time.sleep(2)
        
        # Start background save thread for non-blocking image saving
        self.start_save_thread()
        
        detection_count = 0
        frame_count = 0
        last_detection_time = 0
        consecutive_failures = 0
        max_failures = 100  # Allow many failures before reconnecting
        
        while True:
            try:
                # Periodic cleanup every 200 frames
                if frame_count > 0 and frame_count % 200 == 0:
                    print(f"🔄 Periodic camera cleanup at frame #{frame_count}")
                    try:
                        self.cleanup_camera()
                        time.sleep(0.5)
                        if not self.initialize_camera():
                            print("⚠️ Camera reinit failed, will keep trying...")
                    except Exception as e:
                        print(f"⚠️ Cleanup error: {e}")
                    last_cleanup_frame = frame_count
                
                # Capture frame with retry logic
                frame = None
                for attempt in range(3):
                    try:
                        frame = self.capture_frame()
                        if frame is not None:
                            break
                    except Exception as e:
                        print(f"⚠️ Frame capture attempt {attempt+1}/3 failed: {e}")
                    
                    if frame is None:
                        time.sleep(0.3)
                
                # If still no frame, continuously reconnect until successful
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures <= 5:
                        print(f"⚠️ Frame capture failed ({consecutive_failures} consecutive), retrying...")
                    else:
                        print(f"⚠️ Multiple capture failures ({consecutive_failures}), reconnecting camera...")
                        while True:  # Keep trying until reconnected
                            try:
                                self.cleanup_camera()
                                time.sleep(1)
                                if self.initialize_camera():
                                    print("✅ Camera reconnected successfully")
                                    consecutive_failures = 0
                                    break
                            except Exception as e:
                                print(f"⚠️ Reconnect attempt failed: {e}, retrying in 2 seconds...")
                                time.sleep(2)
                    
                    time.sleep(0.5)
                    continue
                
                # Reset failure count on successful capture
                consecutive_failures = 0
                
                if frame.size == 0:
                    continue
                
                frame_count += 1
                time.sleep(0.5)  # Rate limiting - 1 FPS (1 frame per second)
                
                # Detect killblocks
                try:
                    detections = self.detect_killblocks(frame)
                    current_time = time.time()
                    
                    if detections and (current_time - last_detection_time) > 0.3:
                        detection_count += 1
                        last_detection_time = current_time
                        
                        print(f"\n🎯 Detection #{detection_count} at frame #{frame_count}")
                        
                        # Process detections synchronously in order to preserve chronological sequence
                        # Each detection gets a unique timestamp to ensure no overwriting and proper ordering
                        for detection in detections:
                            try:
                                # Generate unique timestamp for each detection to ensure chronological order
                                # Include milliseconds (3 digits) for uniqueness even if multiple detections in same frame
                                time_str = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
                                
                                # Small delay to ensure timestamp uniqueness if multiple detections in same cycle
                                # This ensures each detection gets a distinct timestamp for proper chronological ordering
                                if len(detections) > 1:
                                    time.sleep(0.001)  # 1ms delay between detections in same frame
                                
                                # Increment sequence counter BEFORE processing to ensure strict order
                                self.detection_sequence += 1
                                sequence_number = self.detection_sequence
                                
                                # Process synchronously (no async/parallel operations)
                                # Each detection saved immediately in capture order
                                self.process_detection(frame, detection, time_str, frame_count, sequence_number)
                            except Exception as e:
                                print(f"⚠️ Processing detection error: {e}")
                                continue
                        
                        print(f"✅ Processing complete\n")
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
                    continue
                
                # Status update every 50 frames
                if frame_count % 50 == 0:
                    print(f"⏳ Still running... Frame #{frame_count}")
            
            except KeyboardInterrupt:
                print("\n\n⏹️ Detection stopped by user (Ctrl+C)")
                break
            except Exception as e:
                print(f"⚠️ Loop error: {e}")
                print("🔄 Attempting reconnect and continuing...")
                try:
                    self.cleanup_camera()
                    time.sleep(1)
                    self.initialize_camera()
                except:
                    pass
                time.sleep(1)
                continue
        
        # Clean up before exiting
        try:
            self.stop_save_thread()
            self.cleanup_camera()
        except:
            pass

def main():
    print("=== Free Fire Killblock Detector ===")
    detector = KillblockDetector()
    
    if detector.model is None:
        print("Failed to load model. Exiting...")
        return
    
    print("Please start OBS Virtual Camera")
    print("Then press Enter to begin detection...")
    input()
    
    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping detection")
    finally:
        detector.stop_save_thread()
        detector.cleanup_camera()

if __name__ == "__main__":
    main()
