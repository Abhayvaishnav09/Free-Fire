"""
PaddleOCR Image Text Recognition
Accepts an uploaded image, runs PaddleOCR, and returns the recognized text.
"""

import os
import sys
import io
from pathlib import Path
from datetime import datetime
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2
try:
    from tkinter import filedialog, Tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class ImageOCR:
    """Class to handle image OCR using PaddleOCR"""
    
    def __init__(self, use_textline_orientation=True, lang='en'):
        """
        Initialize PaddleOCR
        
        Args:
            use_textline_orientation: Whether to use textline orientation detection
            lang: Language code (default: 'en' for English)
        """
        print("🔄 Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_textline_orientation=use_textline_orientation, lang=lang)
        print("✅ PaddleOCR initialized successfully")
    
    def detect_weapon_position(self, image, gray_image):
        """
        Detect weapon/icon position in the middle of the image.
        Weapons are usually non-text regions (icons/symbols) in the center.
        
        Args:
            image: Original BGR image
            gray_image: Grayscale image
            
        Returns:
            int: X coordinate of weapon center, or None if not found
        """
        height, width = gray_image.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # Define search region around center (middle 40% of image width)
        search_left = int(width * 0.3)
        search_right = int(width * 0.7)
        search_top = int(height * 0.2)
        search_bottom = int(height * 0.8)
        
        # Extract center region
        center_region = gray_image[search_top:search_bottom, search_left:search_right]
        
        if center_region.size == 0:
            return center_x  # Fallback to image center
        
        # Apply threshold for contour detection (convert grayscale to binary for contour finding)
        _, thresh_region = cv2.threshold(center_region, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in center region (weapons are usually distinct shapes)
        contours, _ = cv2.findContours(thresh_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour in center (likely the weapon)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box of largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate center of weapon in original image coordinates
            weapon_center_x = search_left + x + w // 2
            
            # Validate: weapon should be roughly in center (30-70% of width)
            if search_left <= weapon_center_x <= search_right:
                print(f"🔫 Detected weapon at X position: {weapon_center_x} (image width: {width})")
                return weapon_center_x
        
        # Fallback: use image center
        print(f"🔫 Using image center as weapon position: {center_x}")
        return center_x
    
    def _parse_ocr_result(self, result, text_type):
        """
        Parse OCR result and return list of text information.
        
        Args:
            result: PaddleOCR result
            text_type: "killer" or "victim"
            
        Returns:
            List of text info dictionaries
        """
        text_lines = []
        
        if isinstance(result, list) and len(result) > 0:
            result_dict = result[0]
            if isinstance(result_dict, dict):
                rec_texts = result_dict.get('rec_texts', [])
                rec_scores = result_dict.get('rec_scores', [])
                rec_polys = result_dict.get('rec_polys', [])
                
                for i, text in enumerate(rec_texts):
                    if i < len(rec_scores) and i < len(rec_polys):
                        confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                        coordinates = rec_polys[i] if i < len(rec_polys) else None
                        
                        # Convert text to string if needed
                        if not isinstance(text, str):
                            if hasattr(text, 'item'):
                                text = str(text.item())
                            else:
                                text = str(text)
                        
                        if text and len(text.strip()) > 0:
                            # Convert polygon to list if it's a numpy array
                            if coordinates is not None and hasattr(coordinates, 'tolist'):
                                coordinates = coordinates.tolist()
                            
                            text_info = {
                                'text': text.strip(),
                                'confidence': confidence,
                                'coordinates': coordinates,
                                'type': text_type
                            }
                            
                            text_lines.append(text_info)
        
        return text_lines
    
    def expand_bounding_box(self, coordinates, image_width, image_height, padding_ratio=0.15):
        """
        Expand bounding box to capture full text including left and right parts.
        
        Args:
            coordinates: List of polygon points [(x1,y1), (x2,y2), ...]
            image_width: Width of the image
            image_height: Height of the image
            padding_ratio: Ratio of padding to add (default 15%)
            
        Returns:
            tuple: Expanded bounding box (x, y, w, h)
        """
        if not coordinates or len(coordinates) == 0:
            return None
        
        # Get all x and y coordinates
        x_coords = [point[0] for point in coordinates]
        y_coords = [point[1] for point in coordinates]
        
        # Calculate original bounding box
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Calculate width and height
        w = x_max - x_min
        h = y_max - y_min
        
        # Add padding (15% of width/height, with minimum values)
        padding_x = max(int(w * padding_ratio), 10)  # At least 10 pixels
        padding_y = max(int(h * padding_ratio), 5)   # At least 5 pixels
        
        # Expand bounding box
        x_expanded = max(0, x_min - padding_x)
        y_expanded = max(0, y_min - padding_y)
        w_expanded = min(image_width - x_expanded, x_max - x_min + 2 * padding_x)
        h_expanded = min(image_height - y_expanded, y_max - y_min + 2 * padding_y)
        
        return (int(x_expanded), int(y_expanded), int(w_expanded), int(h_expanded))
    
    def process_image(self, image_data):
        """
        Process an image from server and extract text using PaddleOCR
        
        Args:
            image_data: Image data from server (numpy array or bytes)
            
        Returns:
            dict: Dictionary containing:
                - text: Combined recognized text
                - lines: List of detected text lines with coordinates
                - killer: Killer name
                - victim: Victim name
                - raw_result: Raw PaddleOCR result
        """
        # Convert image_data to numpy array if it's bytes
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, np.ndarray):
            image = image_data
        else:
            raise ValueError("image_data must be bytes or numpy array")
        
        if image is None:
            raise ValueError("Could not decode image from server data")
        
        # Convert to pure grayscale (no thresholding - smooth grayscale)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = gray_image.shape[:2]
        center_x = width // 2
        
        # Split image into left (killer) and right (victim) halves
        left_half = gray_image[:, :center_x]  # Left part = killer
        right_half = gray_image[:, center_x:]  # Right part = victim
        
        # Convert to 3-channel BGR for PaddleOCR
        left_bgr = cv2.cvtColor(left_half, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right_half, cv2.COLOR_GRAY2BGR)
        
        # Run OCR on left half (killer)
        killer_result = self.ocr.predict(left_bgr)
        killer_texts = self._parse_ocr_result(killer_result, "killer")
        
        # Run OCR on right half (victim)
        victim_result = self.ocr.predict(right_bgr)
        victim_texts = self._parse_ocr_result(victim_result, "victim")
        
        # Combine all text
        all_texts = killer_texts + victim_texts
        combined_text = [t['text'] for t in all_texts]
        recognized_text = '\n'.join(combined_text)
        
        # Get killer and victim text
        killer_text = ' '.join([t['text'] for t in killer_texts]) if killer_texts else ""
        victim_text = ' '.join([t['text'] for t in victim_texts]) if victim_texts else ""
        
        return {
            'text': recognized_text,
            'lines': all_texts,
            'killer': killer_text,
            'victim': victim_text,
            'killer_texts': killer_texts,
            'victim_texts': victim_texts,
            'center_x': center_x,
            'raw_result': {'killer': killer_result, 'victim': victim_result}
        }
    


def process_server_image(image_data, use_textline_orientation=True, lang='en'):
    """
    Process an image from server and extract killer/victim names
    
    Args:
        image_data: Image data from server (bytes or numpy array)
        use_textline_orientation: Whether to use textline orientation detection
        lang: Language code
        
    Returns:
        dict: Dictionary containing:
            - killer: Killer name
            - victim: Victim name
            - text: Combined recognized text
            - lines: List of detected text lines
    """
    ocr_engine = ImageOCR(use_textline_orientation=use_textline_orientation, lang=lang)
    result = ocr_engine.process_image(image_data)
    return result



