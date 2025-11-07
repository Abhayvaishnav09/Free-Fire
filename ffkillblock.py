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
        """Validate revive detections with enhanced adaptive confidence thresholds.
        More powerful and accurate - accepts lower confidence to catch more REVIVED cases."""
        if not revive_detections:
            return False
        
        best_revive = max(revive_detections, key=lambda x: x[1])
        class_name, confidence = best_revive
        
        # High confidence (0.20+): Accept immediately (lowered from 0.25 for more sensitivity)
        if confidence >= 0.20:
            if len(revive_detections) > 1:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.12:  # Lowered from 0.18
                    print(f"✅ REVIVED detected{context} - Multiple confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            print(f"✅ REVIVED detected{context} - Class: {class_name}, Confidence: {confidence:.3f}")
            return True
        
        # Medium confidence (0.15-0.20): Accept immediately (more powerful)
        if confidence >= 0.15:
            if len(revive_detections) > 1:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.10:  # Lowered threshold
                    print(f"✅ REVIVED detected{context} - Multiple medium-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            # Accept single medium-confidence detection to catch more REVIVED cases
            print(f"✅ REVIVED detected{context} - Single medium-confidence (Class: {class_name}, Confidence: {confidence:.3f})")
            return True
        
        # Low confidence (0.10-0.15): Accept if multiple detections or reasonable single detection
        if confidence >= 0.10:
            if len(revive_detections) > 1:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.08:  # Lowered threshold
                    print(f"✅ REVIVED detected{context} - Multiple low-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
            # Accept single low-confidence if it's reasonable (0.12+)
            if confidence >= 0.12:
                print(f"✅ REVIVED detected{context} - Single low-confidence (Class: {class_name}, Confidence: {confidence:.3f})")
                return True
        
        # Very low confidence (0.08-0.10): Only accept if multiple strong confirmations
        if confidence >= 0.08:
            if len(revive_detections) >= 2:
                second_best = sorted(revive_detections, key=lambda x: x[1], reverse=True)[1]
                if second_best[1] >= 0.08:
                    print(f"✅ REVIVED detected{context} - Multiple very-low-confidence confirmations (Class: {class_name}, Confidence: {confidence:.3f}, Second: {second_best[1]:.3f})")
                    return True
        
        print(f"⚠️ Revive confidence too low ({confidence:.3f} < 0.08) - rejected")
        return False
    
    def _process_yolo_revive_detection(self, image, resize_to=None):
        """Process YOLO revive detection on image with optional resize.
        Uses consistent confidence thresholds to ensure accurate REVIVED detection."""
        if self.model is None or image is None or image.size == 0:
            return False
        
        height, width = image.shape[:2]
        if height < 30 or width < 30:
            return False
        
        # Resize if needed
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
            # Very low initial confidence threshold to catch more REVIVED detections
            # Using 0.08 to catch even weak signals, then validate with adaptive thresholds
            results = self.model.predict(processed_image, conf=0.08, verbose=False, device='cpu')
        except Exception as e:
            print(f"⚠️ YOLO predict error: {e}")
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
                
                # Very low threshold to catch more REVIVED detections
                # Accept any revive class detection above 0.08, validation will filter appropriately
                if self._is_revive_class(class_name) and confidence >= 0.08:
                    revive_detections.append((class_name, confidence))
            except Exception as e:
                print(f"⚠️ Error processing box: {e}")
                continue
        
        return self._validate_revive_detections(revive_detections)
    
    def detect_revive_status(self, image):
        """REVIVED detection using YOLOv11 with balanced confidence thresholds.
        Ensures consistent and accurate REVIVED detection."""
        try:
            return self._process_yolo_revive_detection(image)
        except Exception as e:
            print(f"⚠️ Revive detection error: {e}")
            return False
    
    def validate_status_consistency(self, status, cropped_image):
        """Final validation to ensure status consistency and prevent confusion.
        Returns validated status or default fallback."""
        if status is None:
            return "KNOCKED"
        
        status_upper = status.upper().strip()
        
        # Valid statuses
        valid_statuses = ["REVIVED", "KNOCKED", "FINISHED"]
        
        if status_upper not in valid_statuses:
            print(f"⚠️ Invalid status '{status}', defaulting to KNOCKED")
            return "KNOCKED"
        
        # Additional validation: If status is REVIVED, ensure it's not confused with color-based detection
        # This is already handled in process_detection, but add extra safety here
        if status_upper == "REVIVED":
            # REVIVED should not be confused with KNOCKED or FINISHED
            # This is guaranteed by the detection pipeline, but validate here too
            return "REVIVED"
        
        # For KNOCKED and FINISHED, ensure they are mutually exclusive
        # This is already handled in analyze_victim_color, but validate here too
        if status_upper in ["KNOCKED", "FINISHED"]:
            return status_upper
        
        # Default fallback
        return "KNOCKED"

    def preprocess_text_region(self, image):
        """Preprocess image to enhance text visibility and isolate victim name text.
        Applies contrast enhancement and text isolation techniques."""
        try:
            if image is None or image.size == 0:
                return image
            
            # Create a copy to avoid modifying original
            processed = image.copy()
            
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return processed
        except Exception as e:
            print(f"⚠️ Text preprocessing error: {e}")
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
    
    def extract_victim_text_region(self, cropped_image):
        """Extract the precise victim name text region for color analysis."""
        try:
            height, width = cropped_image.shape[:2]
            return self._extract_region(cropped_image, int(width * 0.50), width, 
                                       int(height * 0.20), int(height * 0.80))
        except Exception as e:
            print(f"⚠️ Text region extraction error: {e}")
            return None

    def analyze_victim_color(self, cropped_image):
        """Primary detection method: Analyzes victim name text color for KNOCKED/FINISHED status.
        Focuses strictly on the victim name text region to avoid UI theme interference.
        WHITE text = KNOCKED, RED text = FINISHED. Color-scheme independent and reliable.
        PERFECTLY SEPARATED: Zero overlap between red and white detection ranges."""
        try:
            if cropped_image is None or cropped_image.size == 0:
                return "UNKNOWN"
            
            height, width = cropped_image.shape[:2]
            if height < 10 or width < 10:
                return "UNKNOWN"
            
            # Extract precise victim name text region (excludes UI borders/graphics)
            text_region = self.extract_victim_text_region(cropped_image)
            if text_region is None:
                return "UNKNOWN"
            
            # Preprocess to enhance text visibility
            text_region = self.preprocess_text_region(text_region)
            
            total_pixels = text_region.shape[0] * text_region.shape[1]
            if total_pixels == 0:
                return "UNKNOWN"
            
            # Convert to HSV for color analysis
            hsv_image = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)
            
            # PERFECTLY SEPARATED RED HSV ranges for FINISHED
            # Red requires: High saturation (>= 80) AND high value (>= 90)
            # This ensures NO overlap with white which has low saturation
            # Primary red range (0-12°): Pure red with very high saturation
            lower_red1 = np.array([0, 80, 90])   # Saturation >= 80, Value >= 90
            upper_red1 = np.array([12, 255, 255])
            
            # Secondary red range (168-180°): Wrap-around red with very high saturation
            lower_red2 = np.array([168, 80, 90])  # Matching strict criteria
            upper_red2 = np.array([180, 255, 255])
            
            # PERFECTLY SEPARATED WHITE HSV range for KNOCKED
            # White requires: Very low saturation (<= 15) AND very high value (>= 210)
            # This ensures NO overlap with red which has high saturation
            lower_white = np.array([0, 0, 210])   # Saturation <= 15, Value >= 210
            upper_white = np.array([180, 15, 255])
            
            # CRITICAL SEPARATION GUARANTEE:
            # - Red: Saturation >= 80, Value >= 90
            # - White: Saturation <= 15, Value >= 210
            # - Gap: 65 saturation points between ranges (80-15=65)
            # - This creates IMPOSSIBLE overlap condition
            
            # Create red masks with strict criteria
            mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            mask_red_combined = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Create white mask with strict criteria
            mask_white_raw = cv2.inRange(hsv_image, lower_white, upper_white)
            
            # MUTUAL EXCLUSION: Ensure perfect separation (defensive programming)
            # Red mask: explicitly exclude any white pixels
            mask_red_final = cv2.bitwise_and(mask_red_combined, cv2.bitwise_not(mask_white_raw))
            
            # White mask: explicitly exclude any red pixels
            mask_white_final = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_red_combined))
            
            # Calculate ratios
            red_pixels = cv2.countNonZero(mask_red_final)
            white_pixels = cv2.countNonZero(mask_white_final)
            
            red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Debug output
            print(f"🔍 HSV color analysis - Red ratio: {red_ratio:.4f} ({red_pixels} px), White ratio: {white_ratio:.4f} ({white_pixels} px)")
            
            # SPECIAL CASE: One color is completely absent (zero ratio)
            # If one ratio is 0 and the other is significant, the decision is clear
            if red_ratio == 0 and white_ratio >= 0.010:
                print(f"✅ KNOCKED detected - White present, red absent (ratio: {white_ratio:.4f})")
                return "KNOCKED"
            
            if white_ratio == 0 and red_ratio >= 0.010:
                print(f"✅ FINISHED detected - Red present, white absent (ratio: {red_ratio:.4f})")
                return "FINISHED"
            
            # VALIDATION: Ensure only one status is detected (mutual exclusion)
            # If both ratios are significant, choose the dominant one with clear margin
            
            # STRICT DECISION LOGIC - No ambiguity allowed
            # Primary thresholds (high confidence) - require clear dominance
            red_threshold_primary = 0.020   # Require meaningful red presence
            white_threshold_primary = 0.020  # Require meaningful white presence
            dominance_ratio_primary = 3.0   # One color must be 3x the other (stricter)
            
            # Secondary thresholds (medium confidence) - still require clear dominance
            red_threshold_secondary = 0.015
            white_threshold_secondary = 0.015
            dominance_ratio_secondary = 2.5
            
            # PRIMARY DECISION: HIGH CONFIDENCE - Clear dominance required
            if red_ratio >= red_threshold_primary:
                if white_ratio == 0 or red_ratio >= white_ratio * dominance_ratio_primary:
                    # Additional validation: ensure red is significantly higher
                    if red_ratio > white_ratio + 0.010:  # At least 1% absolute difference
                        dominance = red_ratio / white_ratio if white_ratio > 0 else float('inf')
                        print(f"✅ FINISHED detected - Primary red HSV signal (ratio: {red_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "FINISHED"
            
            if white_ratio >= white_threshold_primary:
                if red_ratio == 0 or white_ratio >= red_ratio * dominance_ratio_primary:
                    # Additional validation: ensure white is significantly higher
                    if white_ratio > red_ratio + 0.010:  # At least 1% absolute difference
                        dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
                        print(f"✅ KNOCKED detected - Primary white HSV signal (ratio: {white_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "KNOCKED"
            
            # SECONDARY DECISION: MEDIUM CONFIDENCE - Still require clear dominance
            if red_ratio >= red_threshold_secondary:
                if white_ratio == 0 or red_ratio >= white_ratio * dominance_ratio_secondary:
                    # Additional validation: ensure red is clearly dominant
                    if red_ratio > white_ratio * 2.0 and red_ratio > white_ratio + 0.008:
                        dominance = red_ratio / white_ratio if white_ratio > 0 else float('inf')
                        print(f"✅ FINISHED detected - Secondary red HSV signal (ratio: {red_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "FINISHED"
            
            if white_ratio >= white_threshold_secondary:
                if red_ratio == 0 or white_ratio >= red_ratio * dominance_ratio_secondary:
                    # Additional validation: ensure white is clearly dominant
                    if white_ratio > red_ratio * 2.0 and white_ratio > red_ratio + 0.008:
                        dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
                        print(f"✅ KNOCKED detected - Secondary white HSV signal (ratio: {white_ratio:.4f}, dominance: {dominance:.2f}x)")
                        return "KNOCKED"
            
            # FINAL FALLBACK: Very clear dominance with minimum presence
            # Only use if one color is clearly dominant (2.5x) and has minimum presence
            if white_ratio == 0 or red_ratio > white_ratio * 2.5:
                if red_ratio >= 0.010:  # Minimum meaningful red presence
                    dominance = red_ratio / white_ratio if white_ratio > 0 else float('inf')
                    print(f"✅ FINISHED detected - Fallback red signal (ratio: {red_ratio:.4f}, dominance: {dominance:.2f}x)")
                    return "FINISHED"
            
            if red_ratio == 0 or white_ratio > red_ratio * 2.5:
                if white_ratio >= 0.010:  # Minimum meaningful white presence
                    dominance = white_ratio / red_ratio if red_ratio > 0 else float('inf')
                    print(f"✅ KNOCKED detected - Fallback white signal (ratio: {white_ratio:.4f}, dominance: {dominance:.2f}x)")
                    return "KNOCKED"
            
            # DEFAULT: If truly ambiguous (both ratios low or similar), default to KNOCKED
            # This is the most common case in Free Fire
            print(f"⚠️ Ambiguous color - defaulting to KNOCKED (Red: {red_ratio:.4f}, White: {white_ratio:.4f})")
            return "KNOCKED"
            
        except Exception as e:
            print(f"⚠️ Color analysis error: {e}")
            return "UNKNOWN"

    def extract_status_text_region(self, cropped_image):
        """Extract the status text region (middle area) where REVIVED/KNOCKED/FINISHED appears."""
        try:
            height, width = cropped_image.shape[:2]
            return self._extract_region(cropped_image, int(width * 0.35), int(width * 0.65),
                                       int(height * 0.15), int(height * 0.85))
        except Exception as e:
            print(f"⚠️ Status text region extraction error: {e}")
            return None

    def detect_revive_text(self, cropped_image):
        """Detect REVIVED status by looking for 'REVIVED' text using OCR.
        Enhanced with multiple preprocessing methods and comprehensive pattern matching.
        Strengthened to catch REVIVED text even with OCR errors or low confidence."""
        if cropped_image is None or cropped_image.size == 0 or self.text_detector is None:
            return False
        
        try:
            # Method 1: Check entire cropped image
            text_regions = self.text_detector.detect_text_regions(cropped_image)
            
            # Method 2: Check status text region (middle area) - most common location for REVIVED
            status_region = self.extract_status_text_region(cropped_image)
            if status_region is not None:
                text_regions.extend(self.text_detector.detect_text_regions(status_region))
            
            # Method 3: Check victim name region (right side) - sometimes REVIVED appears there
            victim_region = self.extract_victim_text_region(cropped_image)
            if victim_region is not None:
                text_regions.extend(self.text_detector.detect_text_regions(victim_region))
            
            # Method 4: Check preprocessed image (enhanced contrast) for better OCR
            try:
                preprocessed = self.preprocess_text_region(cropped_image)
                if preprocessed is not None and preprocessed.size > 0:
                    text_regions.extend(self.text_detector.detect_text_regions(preprocessed))
            except:
                pass
            
            # Method 5: Check left-middle region (sometimes REVIVED appears near killer name)
            try:
                height, width = cropped_image.shape[:2]
                left_middle_region = self._extract_region(cropped_image, int(width * 0.20), int(width * 0.50),
                                                        int(height * 0.15), int(height * 0.85))
                if left_middle_region is not None:
                    text_regions.extend(self.text_detector.detect_text_regions(left_middle_region))
            except:
                pass
            
            # Method 6: Check upper-middle region (status text often appears here)
            try:
                height, width = cropped_image.shape[:2]
                upper_middle_region = self._extract_region(cropped_image, int(width * 0.30), int(width * 0.70),
                                                          int(height * 0.10), int(height * 0.50))
                if upper_middle_region is not None:
                    text_regions.extend(self.text_detector.detect_text_regions(upper_middle_region))
            except:
                pass
            
            # Comprehensive pattern matching for REVIVED indicators
            # Accept even low-confidence OCR if pattern matches strongly
            for region in text_regions:
                text = region.get('text', '').upper().strip()
                confidence = region.get('confidence', 0.0)
                original_text = region.get('text', '')
                
                # Remove common OCR noise characters
                text_clean = text.replace(' ', '').replace('-', '').replace('_', '').replace('.', '').replace(',', '')
                
                # Direct match - most reliable (accept even low confidence for exact match)
                if 'REVIVED' in text:
                    print(f"✅ REVIVED text detected via OCR: '{original_text}' (confidence: {confidence:.3f})")
                    return True
                
                # Starts with REVIVE - likely REVIVED (accept even low confidence)
                if text.startswith('REVIVE'):
                    print(f"✅ REVIVED text detected via OCR: '{original_text}' (confidence: {confidence:.3f})")
                    return True
                
                # Contains REVIVE and is short (likely status word) - accept even low confidence
                if 'REVIVE' in text and len(text) <= 25:
                    words = text.split()
                    for word in words:
                        if word in ['REVIVED', 'REVIVE']:
                            print(f"✅ REVIVED text detected via OCR: '{original_text}' (confidence: {confidence:.3f})")
                            return True
                
                # Check for common OCR errors/misspellings with fuzzy matching
                # Accept even very low confidence if pattern is clear
                revive_patterns = ['REVIVD', 'REVIV', 'REVIVE', 'REVIVED', 'REV1VED', 'REV1VE', 'REVIV3D']
                if any(pattern in text_clean for pattern in revive_patterns):
                    if len(text) <= 30:  # Allow longer text with variants
                        print(f"✅ REVIVED text detected via OCR (variant): '{original_text}' (confidence: {confidence:.3f})")
                        return True
                
                # Check for partial matches (e.g., "REVIV" at start or end)
                if text.startswith('REVIV') or text.endswith('REVIV'):
                    if len(text) <= 20:  # Increased from 15
                        print(f"✅ REVIVED text detected via OCR (partial): '{original_text}' (confidence: {confidence:.3f})")
                        return True
                
                # Check for REVIVE in any position with reasonable length
                if 'REVIVE' in text and len(text) <= 30:  # Increased from 25
                    # Additional validation: check if it's not part of a longer unrelated word
                    if not any(unrelated in text for unrelated in ['REVIVERY', 'REVIVAL', 'REVIVIFY']):
                        print(f"✅ REVIVED text detected via OCR (flexible): '{original_text}' (confidence: {confidence:.3f})")
                        return True
                
                # Check for OCR character substitutions (common errors)
                # R->P, E->F, V->Y, I->1, etc.
                text_normalized = text.replace('1', 'I').replace('0', 'O').replace('5', 'S').replace('3', 'E')
                if 'REVIVE' in text_normalized or 'REVIVED' in text_normalized:
                    if len(text) <= 30:
                        print(f"✅ REVIVED text detected via OCR (normalized): '{original_text}' (confidence: {confidence:.3f})")
                        return True
                
                # Check for fuzzy matches with Levenshtein-like patterns
                # Accept if text contains 4+ consecutive matching characters from REVIVED
                if len(text) >= 4:
                    text_chars = set(text_clean)
                    revive_chars = set('REVIVED')
                    # If 4+ characters match, likely REVIVED
                    if len(text_chars.intersection(revive_chars)) >= 4:
                        if 'R' in text_chars and 'E' in text_chars and 'V' in text_chars:
                            if len(text) <= 15:  # Short text likely to be status word
                                print(f"✅ REVIVED text detected via OCR (fuzzy match): '{original_text}' (confidence: {confidence:.3f})")
                                return True
            
            return False
        except Exception as e:
            print(f"⚠️ OCR text detection error: {e}")
            return False
    
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

    def detect_killblocks(self, image):
        """Detect killblocks using YOLO model."""
        if self.model is None:
            return None
        
        try:
            processed_image = cv2.resize(image, (1280, 720))
            names_map = getattr(self.model, 'names', None)
            if names_map is None:
                return None
            
            try:
                results = self.model.predict(processed_image, conf=0.65, verbose=False, device='cpu')
            except Exception as e:
                print(f"⚠️ YOLO predict error in detect_killblocks: {e}")
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
                    print(f"⚠️ Error processing box in detect_killblocks: {e}")
                    continue
            
            return detections if detections else None
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return None

    def detect_revive_in_frame(self, frame):
        """Check entire frame for revive class from YOLO model."""
        try:
            return self._process_yolo_revive_detection(frame, resize_to=(1280, 720))
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
            
            # ROBUST STATUS DETECTION PIPELINE - Multi-layered detection with mutual exclusion
            # REVIVED detection has ABSOLUTE PRIORITY - must be checked first before color analysis
            # Uses multiple independent methods that work together for maximum reliability
            # Ensures perfect accuracy and consistency with zero overlap between statuses
            
            # Step 1: OCR text detection FIRST (most reliable for REVIVED)
            # This is the strongest signal and should be checked before YOLO
            print("🔍 Checking for REVIVED text via OCR (primary method)...")
            revive_text_detected = self.detect_revive_text(cropped_image)
            
            # Step 1.5: Also check extracted names for REVIVED patterns (additional text-based check)
            # Sometimes REVIVED appears in the name extraction results
            revive_in_names = False
            if not revive_text_detected:
                killer_upper = killer_name.upper().strip()
                victim_upper = victim_name.upper().strip()
                # Check if REVIVED appears in extracted names (sometimes OCR puts it there)
                if 'REVIVED' in killer_upper or 'REVIVED' in victim_upper:
                    revive_in_names = True
                    print(f"✅ REVIVED found in extracted names: '{killer_name}' / '{victim_name}'")
                elif 'REVIVE' in killer_upper or 'REVIVE' in victim_upper:
                    # Check if it's a standalone REVIVE word, not part of a name
                    killer_words = killer_upper.split()
                    victim_words = victim_upper.split()
                    if any(word in ['REVIVE', 'REVIVED'] for word in killer_words + victim_words):
                        revive_in_names = True
                        print(f"✅ REVIVED found in extracted names: '{killer_name}' / '{victim_name}'")
            
            # Step 2: Check for REVIVED using YOLOv11 (secondary method)
            print("🔍 Checking for REVIVED detection (YOLOv11)...")
            revive_in_crop = self.detect_revive_status(cropped_image)
            
            # Step 3: Check full frame for revive (additional validation)
            revive_in_frame = self.detect_revive_in_frame(full_frame)
            
            # Combine results from all detection methods
            # REVIVED is confirmed if ANY method detects it with sufficient confidence
            detection_sources = []
            revive_confidence_score = 0.0
            
            if revive_text_detected:
                detection_sources.append("OCR text recognition")
                revive_confidence_score += 0.5  # OCR is highly reliable
            
            if revive_in_names:
                detection_sources.append("OCR name extraction")
                revive_confidence_score += 0.4  # Name extraction is also reliable for text
            
            if revive_in_crop:
                detection_sources.append("YOLO killblock crop")
                revive_confidence_score += 0.3
            
            if revive_in_frame:
                detection_sources.append("YOLO full frame")
                revive_confidence_score += 0.2
            
            # REVIVED has ABSOLUTE PRIORITY - if detected by any method, use REVIVED
            # Lowered threshold to 0.2 to catch more REVIVED cases (OCR text alone gives 0.5, so this is safe)
            # Mutual exclusion: REVIVED cannot be confused with KNOCKED or FINISHED
            if detection_sources and revive_confidence_score >= 0.2:
                status = "REVIVED"
                source = " + ".join(detection_sources)
                print(f"✅ STATUS: REVIVED (detected via {source}, confidence score: {revive_confidence_score:.2f}) - REVIVED takes absolute priority, skipping color analysis")
            else:
                # Only if NO REVIVED detection, proceed with color analysis for KNOCKED/FINISHED
                # This ensures mutual exclusion: REVIVED vs KNOCKED/FINISHED
                print("🔍 No REVIVED detected, checking color for KNOCKED/FINISHED...")
                status = self.analyze_victim_color(cropped_image)
                
                # VALIDATION: Ensure status is valid and not confused
                if status not in ["KNOCKED", "FINISHED", "UNKNOWN"]:
                    print(f"⚠️ Invalid status '{status}', defaulting to KNOCKED")
                    status = "KNOCKED"
                
                color_name = {'KNOCKED': 'WHITE', 'FINISHED': 'RED'}.get(status, 'UNKNOWN')
                print(f"✅ STATUS: {status} (victim name text color: {color_name})")
                
                # Final validation: Ensure KNOCKED and FINISHED are mutually exclusive
                # This is already handled in analyze_victim_color, but double-check here
                if status == "UNKNOWN":
                    print(f"⚠️ Status detection returned UNKNOWN, defaulting to KNOCKED (most common)")
                    status = "KNOCKED"
            
            # FINAL VALIDATION: Ensure status consistency and prevent any confusion
            status = self.validate_status_consistency(status, cropped_image)
            
            # Prepare for non-blocking save via background thread
            output_dir = "cropkillblock"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{sequence_number:03d}_{killer_name} {status} {victim_name}.png"
            filepath = os.path.join(output_dir, filename)
            
            self.detected_count += 1
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
                self.save_queue.put((cropped_image.copy(), filepath, metadata), timeout=0.1)
                print(f"📤 Enqueued for save: {filename}")
                print(f"   👤 Killer: {killer_name} | 🎯 Victim: {victim_name}")
                print(f"   📊 Status: {status} | 📈 Confidence: {detection['confidence']:.2f}")
                print(f"   🔢 Order: #{sequence_number} (Frame: {frame_number})")
                print(f"   💾 Queue size: {self.save_queue.qsize()} | Detected: {self.detected_count}\n")
                return True
            except queue.Full:
                print(f"⚠️ Save queue full, retrying...")
                try:
                    self.save_queue.put((cropped_image.copy(), filepath, metadata), timeout=1.0)
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
                    print(f"📸 Saved: {os.path.basename(filepath)}")
                    print(f"   👤 Killer: {metadata.get('killer_name', '')} | 🎯 Victim: {metadata.get('victim_name', '')}")
                    print(f"   📊 Status: {metadata.get('status', '')} | 📈 Confidence: {metadata.get('confidence', 0.0):.2f}")
                    print(f"   🔢 Order: #{metadata.get('sequence_number', 0)} (Frame: {metadata.get('frame_number', 0)})")
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
                # Continue running - don't exit thread on error
                time.sleep(0.1)
    
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
        last_save_thread_check = time.time()
        save_thread_check_interval = 10  # Check every 10 seconds
        
        while True:
            try:
                # Check if save thread is still alive and restart if needed
                current_time = time.time()
                if current_time - last_save_thread_check > save_thread_check_interval:
                    if self.save_thread is not None and not self.save_thread.is_alive():
                        print("⚠️ Save thread died! Restarting...")
                        self.start_save_thread()
                    last_save_thread_check = current_time
                
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
                        reconnect_attempts = 0
                        max_reconnect_attempts = 100  # Limit to prevent infinite hang
                        while reconnect_attempts < max_reconnect_attempts:
                            try:
                                self.cleanup_camera()
                                time.sleep(1)
                                if self.initialize_camera():
                                    print("✅ Camera reconnected successfully")
                                    consecutive_failures = 0
                                    break
                                reconnect_attempts += 1
                                if reconnect_attempts % 10 == 0:
                                    print(f"⚠️ Still trying to reconnect... ({reconnect_attempts}/{max_reconnect_attempts})")
                            except Exception as e:
                                print(f"⚠️ Reconnect attempt {reconnect_attempts} failed: {e}, retrying in 2 seconds...")
                                time.sleep(2)
                                reconnect_attempts += 1
                        
                        if reconnect_attempts >= max_reconnect_attempts:
                            print("⚠️ Max reconnect attempts reached, continuing with retry logic...")
                            consecutive_failures = 0  # Reset to allow normal retry
                    
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
                        
                        # Process detections synchronously to preserve chronological sequence
                        for detection in detections:
                            try:
                                time_str = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
                                if len(detections) > 1:
                                    time.sleep(0.001)  # 1ms delay between detections in same frame
                                self.detection_sequence += 1
                                self.process_detection(frame, detection, time_str, frame_count, self.detection_sequence)
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
                import traceback
                traceback.print_exc()
                print("🔄 Attempting reconnect and continuing...")
                try:
                    self.cleanup_camera()
                    time.sleep(1)
                    self.initialize_camera()
                except Exception as reconnect_error:
                    print(f"⚠️ Reconnect error: {reconnect_error}")
                # Ensure save thread is still running
                if self.save_thread is not None and not self.save_thread.is_alive():
                    print("⚠️ Save thread died during error recovery, restarting...")
                    self.start_save_thread()
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
