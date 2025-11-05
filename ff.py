"""
Simple YOLO Object Detection and Cropping Test Script
Uses best_2.pt model to predict classes and store detected object cropped images
"""

import cv2
import torch
import os
import sys
import numpy as np
from datetime import datetime
from ultralytics import YOLO

def load_model():
    """Load the YOLO model from best_2.pt"""
    try:
        model_path = "best_2.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        print(f"📁 Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
        
        print("✅ YOLO model loaded successfully")
        if hasattr(model, 'names'):
            print(f"📋 Available classes: {list(model.names.values())}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for YOLO inference"""
    try:
        # Resize image to standard size for better detection
        target_width, target_height = 1280, 720
        image_resized = cv2.resize(image, (target_width, target_height))
        return image_resized
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def detect_objects(model, image):
    """Detect objects in image using YOLO model"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return []
        
        print(f"🔍 Processed image size: {processed_image.shape[1]}x{processed_image.shape[0]}")
        
        # Run YOLO prediction with very low confidence threshold for testing
        results = model(processed_image, conf=0.01)
        
        detections = []
        original_height, original_width = image.shape[:2]
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates from resized image
                    x1_resized, y1_resized, x2_resized, y2_resized = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name from model
                    class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"
                    
                    # Scale coordinates back to original image size
                    scale_x = original_width / 1280
                    scale_y = original_height / 720
                    
                    x1_orig = int(x1_resized * scale_x)
                    y1_orig = int(y1_resized * scale_y)
                    x2_orig = int(x2_resized * scale_x)
                    y2_orig = int(y2_resized * scale_y)
                    
                    # Ensure coordinates are within original image bounds
                    x1_orig = max(0, min(x1_orig, original_width - 1))
                    y1_orig = max(0, min(y1_orig, original_height - 1))
                    x2_orig = max(x1_orig + 1, min(x2_orig, original_width))
                    y2_orig = max(y1_orig + 1, min(y2_orig, original_height))
                    
                    detections.append({
                        'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'detection_id': i + 1
                    })
        
        return detections
        
    except Exception as e:
        print(f"❌ Error in object detection: {e}")
        return []

def crop_and_save_objects(image, detections, output_dir="detected_objects"):
    """Crop detected objects and save them"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        timestamp = datetime.now().strftime("%H%M%S")
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            detection_id = detection['detection_id']
            
            # Crop the detected object
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.size > 0:
                # Generate filename
                filename = f"detection_{detection_id}_{class_name}_{confidence:.2f}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save cropped image
                success = cv2.imwrite(filepath, cropped_image)
                
                if success:
                    saved_files.append(filename)
                    print(f"📸 Saved: {filename}")
                    print(f"   Class: {class_name}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Bounding box: [{x1}, {y1}, {x2}, {y2}]")
                    print()
                else:
                    print(f"❌ Failed to save: {filename}")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ Error cropping and saving objects: {e}")
        return []

def test_with_sample_image():
    """Test detection with a sample image from detect_block folder"""
    print("=== Testing with Sample Image ===")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Look for sample images in detect_block folder
    sample_dir = "detect_block"
    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory not found: {sample_dir}")
        return
    
    # Find all PNG images
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.png')]
    if not sample_files:
        print(f"❌ No PNG images found in {sample_dir}")
        return
    
    print(f"📁 Found {len(sample_files)} images to test")
    
    total_detections = 0
    successful_tests = 0
    
    # Test each image
    for i, sample_file in enumerate(sample_files):
        print(f"\n--- Testing image {i+1}/{len(sample_files)}: {sample_file} ---")
        
        image_path = os.path.join(sample_dir, sample_file)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            continue
        
        print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Detect objects
        print("🔍 Running object detection...")
        detections = detect_objects(model, image)
        
        if not detections:
            print("❌ No objects detected in this image")
            continue
        
        print(f"✅ Found {len(detections)} objects")
        total_detections += len(detections)
        successful_tests += 1
        
        # Crop and save objects
        print("📸 Cropping and saving detected objects...")
        saved_files = crop_and_save_objects(image, detections, f"detected_objects_{i+1}")
        
        if saved_files:
            print(f"✅ Successfully saved {len(saved_files)} cropped objects")
        else:
            print("❌ No objects were saved")
    
    print(f"\n📊 Summary:")
    print(f"   Images tested: {len(sample_files)}")
    print(f"   Successful detections: {successful_tests}")
    print(f"   Total objects detected: {total_detections}")

def test_with_main_screenshot():
    """Test detection with the main screenshot"""
    print("=== Testing with Main Screenshot ===")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Test with main screenshot
    screenshot_path = "Screenshot 2025-09-11 133327.png"
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return
    
    print(f"📷 Testing with screenshot: {screenshot_path}")
    
    # Load image
    image = cv2.imread(screenshot_path)
    if image is None:
        print(f"❌ Could not load image: {screenshot_path}")
        return
    
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Detect objects
    print("\n🔍 Running object detection...")
    detections = detect_objects(model, image)
    
    if not detections:
        print("❌ No objects detected")
        return
    
    print(f"✅ Found {len(detections)} objects")
    
    # Crop and save objects
    print("\n📸 Cropping and saving detected objects...")
    saved_files = crop_and_save_objects(image, detections, "detected_objects_main")
    
    if saved_files:
        print(f"\n✅ Successfully saved {len(saved_files)} cropped objects")
        print(f"📁 Saved in: detected_objects_main/")
    else:
        print("❌ No objects were saved")

def detect_from_image_path(image_path):
    """Detect objects from a specific image path"""
    print(f"=== Detecting Objects in: {image_path} ===")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📷 Processing image: {os.path.basename(image_path)}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Detect objects
    print("\n🔍 Running object detection...")
    detections = detect_objects(model, image)
    
    if not detections:
        print("❌ No objects detected")
        return
    
    print(f"✅ Found {len(detections)} objects")
    
    # Crop and save objects
    print("\n📸 Cropping and saving detected objects...")
    saved_files = crop_and_save_objects(image, detections, "detected_objects")
    
    if saved_files:
        print(f"\n✅ Successfully saved {len(saved_files)} cropped objects")
        print(f"📁 Saved in: detected_objects/")
    else:
        print("❌ No objects were saved")

def main():
    """Main function with CLI support"""
    print("=== YOLO Object Detection and Cropping Test ===")
    print("Using best_2.pt model to detect and crop objects\n")
    
    # Check if image path provided as command line argument
    if len(sys.argv) > 1:
        # Image path provided as command line argument
        image_path = sys.argv[1]
        print(f"Using image from command line: {image_path}")
        detect_from_image_path(image_path)
    else:
        # Get image path from user input
        image_path = input("Enter the path to your image file: ").strip()
        
        # Remove quotes if user added them
        image_path = image_path.strip('"\'')
        
        if image_path:
            detect_from_image_path(image_path)
        else:
            print("❌ No image path provided")
            print("\nUsage:")
            print("  python ff.py <image_path>")
            print("  or")
            print("  python ff.py")
            print("  (then enter image path when prompted)")

if __name__ == "__main__":
    main()
