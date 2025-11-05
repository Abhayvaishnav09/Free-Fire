"""
Free Fire Killblock Detector
Crops top-right area and checks for killblock presence.

Author: AI Assistant
Date: 2024
"""

import cv2
import os
import numpy as np
from text import FreeFireTextDetector

class KillblockDetector:
    """
    Simple class to detect killblock in cropped area.
    """
    
    def __init__(self):
        """Initialize the killblock detector."""
        print("=== Free Fire Killblock Detector ===")
        print("Crops top-right area and checks for killblock.\n")
        
        # Initialize text detector
        self.text_detector = FreeFireTextDetector()
    
    def crop_killblock_area(self, image_path: str):
        """
        Crop the killblock area from top-right corner.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            str: Path to cropped killblock image
        """
        print(f"\n=== Cropping Killblock Area: {os.path.basename(image_path)} ===")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image")
            return None
        
        height, width = image.shape[:2]
        print(f"Original image: {width}x{height} pixels")
        
        # Fixed cropping parameters - always crop the same area
        if width > 1920:  # High resolution
            crop_x = int(width * 0.75)   # Start from 75% of width
            crop_y = int(height * 0.05)  # Start from 5% from top
            crop_width = int(width * 0.24)   # 24% of width
            crop_height = int(height * 0.15) # 15% of height
        elif width > 1280:  # Medium resolution
            crop_x = int(width * 0.72)  # Start from 72% of width
            crop_y = int(height * 0.05)  # Start from 5% from top
            crop_width = int(width * 0.27)  # 27% of width
            crop_height = int(height * 0.18) # 18% of height
        else:  # Lower resolution
            crop_x = int(width * 0.68)   # Start from 68% of width
            crop_y = int(height * 0.05)  # Start from 5% from top
            crop_width = int(width * 0.31)   # 31% of width
            crop_height = int(height * 0.15) # 15% of height
        
        print(f"Cropping area: x={crop_x}, y={crop_y}, w={crop_width}, h={crop_height}")
        
        # Ensure crop coordinates are within image bounds
        crop_x = max(0, min(crop_x, width - 1))
        crop_y = max(0, min(crop_y, height - 1))
        crop_width = min(crop_width, width - crop_x)
        crop_height = min(crop_height, height - crop_y)
        
        # Crop the killblock area
        killblock_crop = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        
        if killblock_crop.size == 0:
            print("Error: Cropped area is empty")
            return None
        
        # Save cropped image
        crop_filename = f"killblock_crop_{os.path.basename(image_path)}"
        crop_path = os.path.join(os.getcwd(), crop_filename)
        cv2.imwrite(crop_path, killblock_crop)
        
        print(f"✓ Killblock area cropped and saved: {crop_filename}")
        print(f"  Cropped size: {killblock_crop.shape[1]}x{killblock_crop.shape[0]} pixels")
        
        return crop_path

    def check_killblock_in_crop(self, crop_path: str):
        """
        Check if killblock is present in the cropped area.
        
        Args:
            crop_path (str): Path to cropped killblock image
            
        Returns:
            dict: Detection results
        """
        print(f"\n=== Checking for Killblock in Cropped Area ===")
        
        if not os.path.exists(crop_path):
            print(f"Error: Cropped image not found: {crop_path}")
            return None
        
        try:
            # Use the text detector on the cropped image
            text_results = self.text_detector.process_image(crop_path)
            
            # Check if both killer and victim are detected
            killer_detected = text_results.get("killer") is not None
            victim_detected = text_results.get("victim") is not None
            
            killblock_detected = killer_detected and victim_detected
            
            print(f"Killer detected: {killer_detected}")
            print(f"Victim detected: {victim_detected}")
            print(f"Killblock detected: {killblock_detected}")
            
            if killblock_detected:
                print(f"  Killer: {text_results['killer']['text']}")
                print(f"  Victim: {text_results['victim']['text']}")
                
                # Determine victim status
                victim_color = text_results['victim']['color']['name'].lower()
                if victim_color == "red":
                    victim_status = "finished"
                elif victim_color == "white":
                    victim_status = "knocked"
                else:
                    victim_status = "unknown"
                
                print(f"  Status: {victim_status.upper()}")
            
            return {
                "killblock_detected": killblock_detected,
                "killer": text_results.get("killer"),
                "victim": text_results.get("victim"),
                "victim_status": victim_status if killblock_detected else "not_detected"
            }
            
        except Exception as e:
            print(f"Error checking killblock in cropped area: {e}")
            return {"killblock_detected": False, "error": str(e)}

    def process_image(self, image_path: str):
        """
        Complete process: crop area and check for killblock.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            dict: Complete results
        """
        print(f"\n=== Processing Image: {os.path.basename(image_path)} ===")
        
        # Step 1: Always crop the same area
        crop_path = self.crop_killblock_area(image_path)
        if crop_path is None:
            return {"error": "Failed to crop killblock area"}
        
        # Step 2: Check for killblock in cropped area
        results = self.check_killblock_in_crop(crop_path)
        if results is None:
            return {"error": "Failed to check killblock in cropped area"}
        
        # Step 3: Display results
        print("\n" + "="*60)
        print("           KILLBLOCK DETECTION RESULTS")
        print("="*60)
        
        if results["killblock_detected"]:
            print("🔴 KILLBLOCK: DETECTED")
            print(f"   Killer: {results['killer']['text']}")
            print(f"   Victim: {results['victim']['text']}")
            print(f"   Status: {results['victim_status'].upper()}")
        else:
            print("⚪ KILLBLOCK: NOT DETECTED")
            print("   No killblock found in the cropped area")
        
        print("="*60)
        print(f"\n📁 Cropped image saved as: {os.path.basename(crop_path)}")
        print("   Check this image to see what was analyzed!")
        
        return results

def main():
    """
    Main function to run the killblock detector.
    """
    print("=== Free Fire Killblock Detector ===")
    print("Crops top-right area and checks for killblock.\n")
    
    # Initialize detector
    detector = KillblockDetector()
    
    # Get image path
    image_path = input("Enter path to your Free Fire image: ").strip()
    
    if not image_path:
        print("No image path provided. Exiting...")
        return
    
    # Process the image
    results = detector.process_image(image_path)
    
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
    else:
        print(f"\n✅ Processing completed successfully!")
    
    print("\n=== Program Complete ===")

if __name__ == "__main__":
    main()