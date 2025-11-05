"""
Live Killblock Detection from OBS
Uses ffkillblock.py to detect killblocks from live OBS video capture
Detects every 0.6 seconds and saves cropped results to cropkillblock/ folder
"""

import cv2
import torch
import os
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from ffkillblock import SimpleKillblockDetector

def capture_from_obs():
    """Capture frame from OBS Virtual Camera."""
    try:
        cap = cv2.VideoCapture(1)  # OBS Virtual Camera
        
        if not cap.isOpened():
            print("❌ Could not open OBS Virtual Camera!")
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Could not capture frame from OBS")
            return None
        
        return frame
        
    except Exception as e:
        print(f"❌ Error capturing from OBS: {e}")
        return None

def live_killblock_detection():
    """
    Live killblock detection from OBS Virtual Camera.
    Detects killblocks every 0.6 seconds and saves cropped results.
    """
    print("=== Live Killblock Detection from OBS ===")
    print("Detecting killblocks every 0.6 seconds...")
    print("Press Ctrl+C to stop\n")
    
    # Initialize killblock detector
    try:
        detector = SimpleKillblockDetector()
        if detector.model is None:
            print("❌ Failed to load YOLO model. Exiting...")
            return
        print("✅ Killblock detector initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return
    
    # Create output directory
    output_dir = r"C:\Users\PRISHA\OneDrive\Desktop\Free Fire\cropkillblock"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    detection_count = 0
    total_killblocks = 0
    
    try:
        while True:
            # Capture frame from OBS
            frame = capture_from_obs()
            if frame is None:
                print("⏳ Waiting for OBS camera...")
                time.sleep(1)
                continue
            
            # Process frame for killblock detection
            print(f"\n🔍 Detection attempt #{detection_count + 1} - {datetime.now().strftime('%H:%M:%S')}")
            results = detector.process_frame(frame)
            
            if results:
                total_killblocks += 1
                print(f"✅ Killblock detected! Total detected: {total_killblocks}")
            else:
                print("❌ No killblock detected")
            
            detection_count += 1
            
            # Wait 0.6 seconds before next detection
            time.sleep(0.6)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Detection stopped by user")
        print(f"📊 Total detection attempts: {detection_count}")
        print(f"📊 Total killblocks detected: {total_killblocks}")
        print(f"📁 Results saved in: {output_dir}")
    except Exception as e:
        print(f"❌ Error in detection loop: {e}")

def detect_and_crop(image_path):
    """Detect and crop objects from a single image"""
    print(f"=== Processing Image: {os.path.basename(image_path)} ===")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Initialize detector
    detector = SimpleKillblockDetector()
    if detector.model is None:
        print("❌ Failed to load YOLO model")
        return False
    
    # Test killblock detection
    print("\n🎯 Testing killblock detection...")
    killblock_detections = detector.test_killblock_detection(image)
    
    if killblock_detections:
        print(f"✅ Killblocks found: {len(killblock_detections)}")
        
        # Test revive detection
        print("\n🔍 Testing revive detection...")
        revive_detected = detector.detect_revive_ff_style(image)
        print(f"   Revive detected: {revive_detected}")
        
        # Process each killblock
        for i, detection in enumerate(killblock_detections):
            print(f"\n--- Processing Killblock {i+1} ---")
            x1, y1, x2, y2 = detection['bbox']
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size > 0:
                # Test victim status detection
                print("🎨 Testing victim status detection...")
                status = detector.detect_victim_status(cropped_image)
                print(f"   Victim status: {status}")
                
                # Test full processing
                print("📸 Testing full processing...")
                time_str = datetime.now().strftime("%H%M%S")
                success = detector.process_killblock(
                    cropped_image, 
                    detection['confidence'], 
                    revive_detected, 
                    time_str
                )
                
                if success:
                    print("✅ Processing completed successfully")
                else:
                    print("❌ Processing failed")
            else:
                print("❌ Invalid cropped image")
    else:
        print("❌ No killblocks detected")
    
    return True

def main():
    """Main function to get user input and run detection"""
    print("=== YOLOv11 Object Detection and Cropping ===")
    print("This tool detects objects using YOLOv11 and crops them")
    print()
    
    # Get image path from user
    image_path = input("Enter the path to your image file: ").strip()
    
    # Remove quotes if user added them
    image_path = image_path.strip('"\'')
    
    # Run detection and cropping
    detect_and_crop(image_path)

if __name__ == "__main__":
    main()
