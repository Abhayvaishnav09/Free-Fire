"""
Free Fire Kill Feed Text Detection and Cropping

This script detects killer and victim names from Free Fire kill feed images
and saves cropped images of each name.

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from collections import Counter

# Lazy import of paddleocr - will be imported only when FreeFireTextDetector is instantiated
# This prevents OSError during module import

class FreeFireTextDetector:
    """
    A class to detect killer and victim names from Free Fire kill feed images.
    """
    
    def __init__(self):
        """
        Initialize the Free Fire text detector with PaddleOCR.
        """
        # Lazy import - only import when actually needed
        try:
            import paddleocr
            import logging
            # Disable PaddleOCR debug output
            logging.getLogger('ppocr').setLevel(logging.WARNING)
            
            # PaddleOCR 2.10.x — tuned for small killfeed name crops
            self.paddleocr_reader = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                enable_mkldnn=False,
                det_db_box_thresh=0.4,
                drop_score=0.3,
            )
            self._use_predict_api = hasattr(self.paddleocr_reader, 'predict')
            version = getattr(paddleocr, '__version__', 'unknown')
            api = 'predict()' if self._use_predict_api else 'ocr()'
            print(f"PaddleOCR initialized successfully ({version}, {api})")
        except OSError as e:
            error_msg = str(e)
            if "shm.dll" in error_msg or "WinError 127" in error_msg:
                raise ImportError(
                    f"Failed to load PyTorch DLLs: {e}\n"
                    "💡 Solution: Install Visual C++ Redistributables from:\n"
                    "   https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                    "   Or reinstall PyTorch: pip install --force-reinstall torch"
                )
            else:
                raise ImportError(f"Failed to initialize PaddleOCR: {e}")
        except ImportError as e:
            raise ImportError(f"Failed to import paddleocr: {e}. Please install paddleocr: pip install paddleocr")
        except Exception as e:
            raise ImportError(f"Failed to initialize PaddleOCR: {e}")
        
        # Set up output directories for cropped images
        self.output_dir = "cropped_text"
        self.killer_dir = os.path.join(self.output_dir, "killer")
        self.victim_dir = os.path.join(self.output_dir, "victim")
        
        # Create directories if they don't exist
        os.makedirs(self.killer_dir, exist_ok=True)
        os.makedirs(self.victim_dir, exist_ok=True)
        
        # CSS color mapping for better color names
        self.css_colors = {
            'red': ['#FF0000', '#DC143C', '#B22222', '#8B0000', '#A52A2A', '#CD5C5C'],
            'white': ['#FFFFFF', '#F5F5F5', '#F0F0F0', '#E6E6E6', '#DCDCDC'],
            'black': ['#000000', '#1C1C1C', '#2F2F2F', '#404040', '#696969'],
            'gray': ['#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3', '#DCDCDC'],
            'blue': ['#0000FF', '#0000CD', '#191970', '#4169E1', '#6495ED'],
            'green': ['#008000', '#006400', '#228B22', '#32CD32', '#90EE90'],
            'yellow': ['#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF6347'],
            'orange': ['#FFA500', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500'],
            'purple': ['#800080', '#4B0082', '#8A2BE2', '#9370DB', '#DA70D6'],
            'pink': ['#FFC0CB', '#FF69B4', '#FF1493', '#DC143C', '#C71585']
        }
    
    def cleanup_old_images(self):
        """
        Clean up old cropped images, keeping only the latest ones.
        """
        return
    
    def rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """Convert RGB to hex color."""
        return f"#{r:02x}{g:02x}{b:02x}".upper()
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def get_dominant_color_kmeans(self, image: np.ndarray, k: int = 3) -> Tuple[int, int, int]:
        """
        Get dominant color using KMeans clustering.
        
        Args:
            image (np.ndarray): Input image
            k (int): Number of clusters
            
        Returns:
            Tuple[int, int, int]: RGB values of dominant color
        """
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        
        # Get the most common cluster
        labels = kmeans.labels_
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0]
        
        # Get the centroid of the dominant cluster
        dominant_color = kmeans.cluster_centers_[dominant_label]
        
        return tuple(map(int, dominant_color))
    
    def get_dominant_color_median(self, image: np.ndarray) -> Tuple[int, int, int]:
        """
        Get dominant color using median (more robust than mean).
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[int, int, int]: RGB values of dominant color
        """
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        
        # Get median color
        median_color = np.median(data, axis=0)
        
        return tuple(map(int, median_color))
    
    def find_nearest_css_color(self, rgb: Tuple[int, int, int]) -> str:
        """
        Find the nearest CSS color name for given RGB values.
        Returns White, Red (orange to pink range), or Unknown.
        
        Args:
            rgb (Tuple[int, int, int]): RGB values
            
        Returns:
            str: "white", "red", or "unknown"
        """
        r, g, b = rgb
        brightness = (r + g + b) / 3
        
        # Check for white (high brightness)
        if brightness > 180:
            return "white"
        
        # Check for red range (more strict constraints)
        # Red range: R should be significantly higher than G and B
        if r > g and r > b:
            # Calculate how "red-like" the color is
            red_ratio = r / max(g, b) if max(g, b) > 0 else r
            
            # Much stricter red detection
            if red_ratio > 2.0 and r > 120 and r > g + 40 and r > b + 40:
                return "red"
            elif r > 150 and g < 80 and b < 80 and r > g + 60 and r > b + 60:
                return "red"
        # Check for pink range (more strict)
        if r > 150 and b > 120 and r > g + 30 and b > g + 20 and r > 180:
            return "red"
        
        # Check for orange range (more strict)
        if r > 150 and g > 80 and b < 60 and r > g + 40 and g > b + 30:
            return "red"
        return "unknown"
    
    def detect_color_method_a(self, image: np.ndarray) -> Dict:
        """
        Method A: OCR-based color detection with text pixel isolation.
        
        Args:
            image (np.ndarray): Cropped text image
            
        Returns:
            Dict: Color information with hex and name
        """
        # Convert to RGB if image is in BGR format
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Create a mask to isolate text pixels
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Use adaptive thresholding to separate text from background
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Invert binary image so text pixels are white
        binary = cv2.bitwise_not(binary)
        
        # Find contours to get text regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Create mask from largest contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Get text pixels using mask
            text_pixels = rgb_image[mask > 0]
            
            if len(text_pixels) > 0:
                # Use KMeans to find dominant color
                dominant_rgb = self.get_dominant_color_kmeans(text_pixels, k=3)
            else:
                # Fallback to median
                dominant_rgb = self.get_dominant_color_median(rgb_image)
        else:
            # Fallback to median of entire region
            dominant_rgb = self.get_dominant_color_median(rgb_image)
        
        # Convert to hex and find nearest CSS color
        hex_color = self.rgb_to_hex(*dominant_rgb)
        color_name = self.find_nearest_css_color(dominant_rgb)
        
        return {
            "rgb": dominant_rgb,
            "hex": hex_color,
            "name": color_name
        }
    
    def detect_color_method_b(self, image: np.ndarray) -> Dict:
        """
        Method B: HSV color segmentation for red-like hues.
        
        Args:
            image (np.ndarray): Cropped text image
            
        Returns:
            Dict: Color information with hex and name
        """
        # Convert to RGB if image is in BGR format
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Convert to HSV for better color segmentation
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Define range for red colors in HSV
        # Red has two ranges in HSV due to the circular nature of hue
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red colors
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # If we found red pixels, use them
        if np.any(red_mask):
            red_pixels = rgb_image[red_mask > 0]
            if len(red_pixels) > 0:
                dominant_rgb = self.get_dominant_color_kmeans(red_pixels, k=2)
            else:
                dominant_rgb = self.get_dominant_color_median(rgb_image)
        else:
            # Fallback to median of entire region
            dominant_rgb = self.get_dominant_color_median(rgb_image)
        
        # Convert to hex and find nearest CSS color
        hex_color = self.rgb_to_hex(*dominant_rgb)
        color_name = self.find_nearest_css_color(dominant_rgb)
        
        return {
            "rgb": dominant_rgb,
            "hex": hex_color,
            "name": color_name
        }
    
    def detect_text_color(self, cropped_image_path: str) -> Dict:
        """
        Detect the color of text in a cropped image using both methods.
        
        Args:
            cropped_image_path (str): Path to the cropped text image
            
        Returns:
            Dict: Color detection results
        """
        # Check if path is valid
        if not cropped_image_path or not os.path.exists(cropped_image_path):
            return {"name": "unknown", "rgb": (0, 0, 0), "hex": "#000000"}
        
        # Load the cropped image
        image = cv2.imread(cropped_image_path)
        if image is None or image.size == 0:
            return {"name": "unknown", "rgb": (0, 0, 0), "hex": "#000000"}
        
        
        # Try Method A (OCR-based with text pixel isolation)
        try:
            result_a = self.detect_color_method_a(image)
            color_a = result_a["name"]
            rgb_a = result_a["rgb"]
        except Exception as e:
            color_a = "gray"
            rgb_a = (128, 128, 128)
        
        # Try Method B (HSV color segmentation)
        try:
            result_b = self.detect_color_method_b(image)
            color_b = result_b["name"]
            rgb_b = result_b["rgb"]
        except Exception as e:
            color_b = "gray"
            rgb_b = (128, 128, 128)
        
        # Choose the best result based on confidence
        # Priority: Red > White > Unknown
        if color_a == "red" and color_b != "red":
            final_result = result_a
        elif color_b == "red" and color_a != "red":
            final_result = result_b
        elif color_a == "white" and color_b != "white":
            final_result = result_a
        elif color_b == "white" and color_a != "white":
            final_result = result_b
        elif color_a == "unknown" and color_b != "unknown":
            final_result = result_b
        elif color_b == "unknown" and color_a != "unknown":
            final_result = result_a
        else:
            # Both methods agree
            final_result = result_a
        
        # Keep the result as is (white, red, or unknown)
        
        return final_result
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve text detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to BGR for PaddleOCR
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and validate detected text."""
        if not text:
            return ""
        
        # Remove common OCR errors
        text = text.replace('O', '0')  # Replace O with 0 in numbers
        text = text.replace('l', '1')  # Replace l with 1 in numbers
        text = text.replace('I', '1')  # Replace I with 1 in numbers
        
        # Remove special characters except dots and spaces
        cleaned = ''.join(c for c in text if c.isalnum() or c in ' ._-')
        
        # Remove extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _poly_to_bbox(self, poly):
        """Convert polygon points to (x, y, w, h)."""
        if hasattr(poly, 'tolist'):
            poly = poly.tolist()
        if not isinstance(poly, list) or len(poly) == 0:
            return None
        if isinstance(poly[0], (list, tuple)) and len(poly[0]) >= 2:
            x_coords = [float(point[0]) for point in poly]
            y_coords = [float(point[1]) for point in poly]
            x, y = min(x_coords), min(y_coords)
            return int(x), int(y), int(max(x_coords) - x), int(max(y_coords) - y)
        return None
    
    def _append_text_region(self, formatted_results, text, conf, poly):
        """Append one OCR hit if it passes size/confidence filters."""
        if conf <= 0.005 or not text:
            return
        if not isinstance(text, str):
            text = str(text.item()) if hasattr(text, 'item') else str(text)
        text = text.strip()
        if not text:
            return
        cleaned_text = self._clean_text(text)
        if not cleaned_text or len(cleaned_text) <= 1:
            return
        bbox = self._poly_to_bbox(poly)
        if bbox is None:
            return
        x, y, w, h = bbox
        if w > 10 and h > 8:
            formatted_results.append({
                'text': cleaned_text,
                'bbox': (x, y, w, h),
                'confidence': float(conf),
            })
    
    def _parse_predict_api_result(self, predict_result):
        """Parse PaddleOCR 3.x predict() output."""
        formatted_results = []
        if not isinstance(predict_result, list) or len(predict_result) == 0:
            return formatted_results
        result_dict = predict_result[0]
        if not isinstance(result_dict, dict):
            return formatted_results
        rec_texts = result_dict.get('rec_texts', [])
        rec_scores = result_dict.get('rec_scores', [])
        rec_polys = result_dict.get('rec_polys', [])
        for i, text in enumerate(rec_texts):
            conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            poly = rec_polys[i] if i < len(rec_polys) else None
            try:
                self._append_text_region(formatted_results, text, conf, poly)
            except Exception:
                continue
        return formatted_results
    
    def _parse_ocr_api_result(self, ocr_result):
        """Parse PaddleOCR 2.x ocr() output."""
        formatted_results = []
        if not ocr_result or not isinstance(ocr_result, list):
            return formatted_results
        lines = ocr_result[0] if ocr_result and isinstance(ocr_result[0], list) else ocr_result
        if not lines:
            return formatted_results
        for line in lines:
            if line is None or len(line) < 2:
                continue
            try:
                bbox, text_info = line
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, conf = text_info[0], text_info[1]
                else:
                    continue
                self._append_text_region(formatted_results, text, conf, bbox)
            except Exception:
                continue
        return formatted_results
    
    def _run_paddle_ocr(self, image):
        """Run PaddleOCR using predict() (3.x) or ocr() (2.x) and return normalized regions."""
        if self._use_predict_api:
            raw = self.paddleocr_reader.predict(image)
            return self._parse_predict_api_result(raw)
        raw = self.paddleocr_reader.ocr(image, cls=True)
        return self._parse_ocr_api_result(raw)
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in the image using PaddleOCR.
        
        Args:
            image (np.ndarray): Input image as numpy array
            
        Returns:
            List[Dict]: List of text detection results
        """
        formatted_results = []
        
        # Validate image before processing
        if image is None or not isinstance(image, np.ndarray):
            return []
        
        # Ensure image is in correct format
        if len(image.shape) != 3 or image.shape[2] != 3:
            return []
        
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            return []
        
        # Ensure uint8 type
        if image.dtype != np.uint8:
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Run OCR (compatible with PaddleOCR 2.x ocr() and 3.x predict())
        try:
            formatted_results = self._run_paddle_ocr(image)
            if formatted_results:
                return formatted_results
        except Exception:
            pass
        
        # Retry on preprocessed image
        if not formatted_results:
            try:
                processed_image = self.preprocess_image(image)
                formatted_results = self._run_paddle_ocr(processed_image)
            except Exception as e:
                print(f"[WARN] OCR error on processed image: {e}")
        
        return formatted_results
    
    def extract_text_left_to_right(self, text_regions: List[Dict]) -> List[Dict]:
        """
        Extract text regions sorted strictly left-to-right by x-coordinate.
        This ensures killer (leftmost) and victim (rightmost) are in correct order.
        
        Args:
            text_regions (List[Dict]): List of detected text regions
            
        Returns:
            List[Dict]: Text regions sorted by x-coordinate (left to right)
        """
        if not text_regions:
            return []
        
        # Sort by leftmost x-coordinate (bbox[0])
        sorted_regions = sorted(text_regions, key=lambda r: r['bbox'][0])
        return sorted_regions
    
    def _best_text_in_half(self, half_image: np.ndarray, side: str) -> Tuple[str, float]:
        """OCR one killfeed half (killer left / victim right)."""
        if half_image is None or half_image.size == 0:
            return "", 0.0
        regions = self.detect_text_regions(half_image)
        if not regions:
            return "", 0.0
        filtered = [r for r in regions if r.get("confidence", 0) > 0.22]
        if not filtered:
            filtered = regions
        filtered.sort(key=lambda r: r["bbox"][0])
        if side == "killer":
            pick = filtered[0]
        else:
            pick = filtered[-1]
        text = (pick.get("text") or "").strip()
        return text, float(pick.get("confidence", 0.0))

    def extract_killfeed_split(self, image: np.ndarray) -> Dict:
        """
        Fast split OCR: killer from left ~46%, victim from right ~46%.
        Better than full-width sequence when icon sits between names.
        """
        if image is None or image.size == 0:
            return {"killer": "", "victim": "", "status": None, "confidence": 0.0}

        work = image
        h, w = work.shape[:2]
        if h < 32 or w < 120:
            work = cv2.resize(work, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            h, w = work.shape[:2]

        left_w = max(12, int(w * 0.50))
        right_start = max(left_w + 2, w - int(w * 0.50))
        left = work[:, :left_w]
        right = work[:, right_start:]

        killer, killer_conf = self._best_text_in_half(left, "killer")
        victim, victim_conf = self._best_text_in_half(right, "victim")

        if killer and victim and killer.upper() != victim.upper():
            return {
                "killer": killer,
                "victim": victim,
                "status": None,
                "confidence": (killer_conf + victim_conf) / 2.0,
            }

        return self.extract_killfeed_sequence(work)

    def extract_killfeed_sequence(self, image: np.ndarray) -> Dict:
        """
        Extract killfeed text in correct sequence: killer → status → victim.
        Reads text from left to right to maintain correct order and validates positions.
        
        Args:
            image (np.ndarray): Cropped killfeed image
            
        Returns:
            Dict: {
                'killer': str,  # Leftmost text (killer name)
                'victim': str,  # Rightmost text (victim name)
                'status': None,  # Status determined by color analysis (not here)
                'confidence': float  # Average confidence
            }
        """
        if image is None or image.size == 0:
            return {'killer': '', 'victim': '', 'status': None, 'confidence': 0.0}
        
        # Detect all text regions
        text_regions = self.detect_text_regions(image)
        
        if not text_regions or len(text_regions) < 2:
            # If less than 2 regions, try to extract what we can
            if len(text_regions) == 1:
                # Single text region - assume it's victim (right side is more common)
                region = text_regions[0]
                return {
                    'killer': '',  # Unknown killer
                    'victim': region['text'].strip(),
                    'status': None,
                    'confidence': region['confidence']
                }
            return {'killer': '', 'victim': '', 'status': None, 'confidence': 0.0}
        
        # Sort text regions strictly left-to-right by x-coordinate
        sorted_regions = self.extract_text_left_to_right(text_regions)
        
        # Filter out very low confidence regions
        filtered_regions = [r for r in sorted_regions if r['confidence'] > 0.3]
        
        if len(filtered_regions) < 2:
            # If filtering removed too many, use original sorted list
            filtered_regions = sorted_regions[:2]
        
        # STRICT LEFT-TO-RIGHT: First (leftmost) = killer, Second (rightmost) = victim
        # NO SWAPPING ALLOWED - trust the X-coordinate sorting
        killer_region = filtered_regions[0]  # First = leftmost = killer
        victim_region = filtered_regions[-1]  # Last = rightmost = victim
        
        # If we have more than 2 regions, still use first and last (strict left-to-right)
        if len(filtered_regions) > 2:
            # Use first (leftmost) as killer - NEVER SWAP
            killer_region = filtered_regions[0]
            # Use last (rightmost) as victim - NEVER SWAP
            victim_region = filtered_regions[-1]
        
        # Position validation: Verify left-to-right order (for logging only, NO SWAPPING)
        killer_x_left = killer_region['bbox'][0]  # Left edge of killer
        victim_x_left = victim_region['bbox'][0]  # Left edge of victim
        
        # Log if order seems incorrect (but NEVER swap)
        if killer_x_left > victim_x_left:
            print(f"⚠️ OCR WARNING: Killer left edge ({killer_x_left}) > Victim left edge ({victim_x_left})")
            print(f"   Killer: '{killer_region['text']}' | Victim: '{victim_region['text']}'")
            print(f"   ⚠️  ORDER MAINTAINED AS-IS (no swap) - trusting X-coordinate sort")
        
        # Additional validation: Log position info (for debugging, NO SWAPPING)
        image_width = image.shape[1]
        midpoint = image_width / 2
        
        killer_x_center = killer_region['bbox'][0] + killer_region['bbox'][2] / 2
        victim_x_center = victim_region['bbox'][0] + victim_region['bbox'][2] / 2
        
        killer_is_left_half = killer_x_center < midpoint
        victim_is_right_half = victim_x_center > midpoint
        
        if not killer_is_left_half or not victim_is_right_half:
            print(f"⚠️ OCR: Position validation - Killer: '{killer_region['text']}' (left: {killer_is_left_half}), Victim: '{victim_region['text']}' (right: {victim_is_right_half})")
            print(f"   ⚠️  ORDER MAINTAINED AS-IS (no swap) - trusting strict left-to-right X-coordinate sort")
        
        # Calculate average confidence
        avg_confidence = (killer_region['confidence'] + victim_region['confidence']) / 2.0
        
        return {
            'killer': killer_region['text'].strip(),
            'victim': victim_region['text'].strip(),
            'status': None,  # Status determined by color analysis in ffkillblock.py
            'confidence': avg_confidence
        }
    
    def classify_text_positions(self, text_regions: List[Dict], image_width: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify text regions as killer (left side) or victim (right side).
        
        Args:
            text_regions (List[Dict]): List of detected text regions
            image_width (int): Width of the image
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (killer_texts, victim_texts)
        """
        # Use 40% of image width as the dividing line (killer names are usually on the left 40%)
        divider_x = int(image_width * 0.4)
        killer_texts = []
        victim_texts = []
        
        
        for region in text_regions:
            x, y, w, h = region['bbox']
            center_x = x + w // 2
            
            # Classify as killer (left side) or victim (right side)
            if center_x < divider_x:
                killer_texts.append(region)
            else:
                victim_texts.append(region)
        
        return killer_texts, victim_texts
    
    def find_largest_text(self, text_list: List[Dict]) -> Dict:
        """
        Find the largest text from a list based on area.
        
        Args:
            text_list (List[Dict]): List of text regions
            
        Returns:
            Dict: Largest text region or None
        """
        if not text_list:
            return None
        
        # Sort by area (width * height) in descending order
        text_list.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        return text_list[0]
    
    def crop_and_save_text(self, image: np.ndarray, text_region: Dict, text_type: str, index: int = 0) -> str:
        """
        Crop text region from image and save it with precise text-only cropping.
        
        Args:
            image (np.ndarray): Original image
            text_region (Dict): Text region information
            text_type (str): 'killer' or 'victim'
            index (int): Index for multiple texts of same type
            
        Returns:
            str: Path to saved cropped image
        """
        x, y, w, h = text_region['bbox']
        text = text_region['text']
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        # Extract the text region first
        text_area = image[y:y+h, x:x+w]
        
        # Convert to grayscale for better text detection
        gray = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours to get exact text boundaries
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the bounding rectangle of all text contours
            all_points = np.concatenate(contours)
            x_min, y_min, w_contour, h_contour = cv2.boundingRect(all_points)
            
            # No padding - crop to exact text boundaries
            x_start = x_min
            y_start = y_min
            x_end = x_min + w_contour
            y_end = y_min + h_contour
            
            # Crop to exact text boundaries
            cropped_text = text_area[y_start:y_end, x_start:x_end]
        else:
            # Fallback: use original coordinates with no padding
            cropped_text = text_area
        
        # Ensure we have a valid cropped image
        if cropped_text.size == 0:
            return ""
        
        # Generate filename with timestamp and text content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        safe_text = "".join(c for c in text if c.isalnum() or c in '._-')[:20]  # Limit length
        filename = f"{index:03d}_{safe_text}_{timestamp}.png"
        
        # Determine output directory based on text type
        if text_type == "killer":
            output_dir = self.killer_dir
        elif text_type == "victim":
            output_dir = self.victim_dir
        else:
            output_dir = self.output_dir
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the cropped image
        filepath = os.path.join(output_dir, filename)
        success = cv2.imwrite(filepath, cropped_text)
        
        if success:
            return filepath
        else:
            return ""
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a Free Fire kill feed image to detect and crop killer and victim names.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Dict: Results containing killer and victim information
        """
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect text regions
        text_regions = self.detect_text_regions(image)
        
        if not text_regions:
            return {"message": "No text detected in the image"}
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Classify text as killer or victim
        killer_texts, victim_texts = self.classify_text_positions(text_regions, width)
        
        # Find the largest text for each type
        killer = self.find_largest_text(killer_texts)
        victim = self.find_largest_text(victim_texts)
        
        results = {
            "killer": None,
            "victim": None,
            "killer_cropped_path": None,
            "victim_cropped_path": None
        }
        
        # Process killer text (no color detection)
        if killer:
            killer_path = self.crop_and_save_text(image, killer, "killer")
            results["killer"] = {
                "text": killer['text'],
                "position": killer['bbox'],
                "confidence": killer['confidence'],
                "color": {"name": "not_detected", "rgb": (0, 0, 0), "hex": "#000000"}
            }
            results["killer_cropped_path"] = killer_path
        else:
            print("No killer text detected")
        
        # Process victim text
        if victim:
            victim_path = self.crop_and_save_text(image, victim, "victim")
            victim_color = self.detect_text_color(victim_path) if victim_path else {"name": "unknown", "rgb": (0,0,0), "hex": "#000000"}
            results["victim"] = {
                "text": victim['text'],
                "position": victim['bbox'],
                "confidence": victim['confidence'],
                "color": victim_color
            }
            results["victim_cropped_path"] = victim_path
        else:
            print("No victim text detected")
        
        return results

def main():
    """
    Main function to run the Free Fire text detection.
    """
    print("=== Free Fire Kill Feed Text Detection ===")
    print("This program detects killer and victim names from Free Fire images.")
    print("It will also save cropped images of each detected name.\n")
    
    try:
        # Initialize detector
        detector = FreeFireTextDetector()
        
        # Get image path from user
        image_path = input("Enter the path to your Free Fire image: ").strip()
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
        
        # Process the image
        results = detector.process_image(image_path)
        
        # Clean up old images
        detector.cleanup_old_images()
        
        # Display results
        print("\n=== Detection Results ===")
        
        if results.get("message"):
            print(results["message"])
        else:
            # Show killer information
            if results["killer"]:
                killer_color = results['killer']['color']
                print(f"1. Killer: '{results['killer']['text']}'")
                print(f"   Position: {results['killer']['position']}")
                print(f"   Confidence: {results['killer']['confidence']:.2f}")
                print(f"   Color: {killer_color['name'].title()} (RGB: {killer_color['rgb']}, Hex: {killer_color['hex']})")
                print(f"   Cropped image: {results['killer_cropped_path']}")
            else:
                print("1. Killer: Not detected")
            
            # Show victim information
            if results["victim"]:
                victim_color = results['victim']['color']
                print(f"2. Victim: '{results['victim']['text']}'")
                print(f"   Position: {results['victim']['position']}")
                print(f"   Confidence: {results['victim']['confidence']:.2f}")
                print(f"   Color: {victim_color['name'].title()} (RGB: {victim_color['rgb']}, Hex: {victim_color['hex']})")
                print(f"   Cropped image: {results['victim_cropped_path']}")
            else:
                print("2. Victim: Not detected")
        
        # Clean summary
        print("\n=== Summary ===")
        if results.get("killer"):
            killer_text = results['killer']['text']
        else:
            killer_text = "Not detected"
            
        if results.get("victim"):
            victim_text = results['victim']['text']
            victim_color = results['victim']['color']['name'].lower()
            status = "kill" if victim_color == "red" else "gun knockout"
            print(f"{killer_text} - {status} - {victim_text}")
        else:
            print(f"{killer_text} - No victim detected")
        
        print(f"\n=== Output Directories ===")
        print(f"Killer images saved in: {detector.killer_dir}")
        print(f"Victim images saved in: {detector.victim_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

