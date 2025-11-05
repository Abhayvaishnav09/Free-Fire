import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
from typing import List, Optional
import os

try:
    import paddleocr
    import logging
    # Disable PaddleOCR debug output
    logging.getLogger('ppocr').setLevel(logging.WARNING)
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


def detect_text_regions(image: np.ndarray) -> List[tuple]:
    """
    Detect all text regions in the image using PaddleOCR.
    
    Args:
        image: Input image in BGR format
    
    Returns:
        List of bounding boxes (x, y, w, h) for text regions
    """
    if not PADDLEOCR_AVAILABLE:
        st.error("PaddleOCR is not installed. Please install it: pip install paddleocr")
        return []
    
    try:
        # Initialize PaddleOCR reader
        ocr_reader = paddleocr.PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            use_gpu=False
        )
        
        # Run OCR on the image
        results = ocr_reader.ocr(image, cls=True)
        
        text_boxes = []
        if results and results[0]:
            for line in results[0]:
                if line is None or len(line) < 2:
                    continue
                
                try:
                    bbox, (text, conf) = line
                    # Only include text with reasonable confidence
                    if conf > 0.1:
                        # Extract bounding box coordinates
                        x_coords = [int(point[0]) for point in bbox]
                        y_coords = [int(point[1]) for point in bbox]
                        x = min(x_coords)
                        y = min(y_coords)
                        w = max(x_coords) - x
                        h = max(y_coords) - y
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, min(x, image.shape[1] - 1))
                        y = max(0, min(y, image.shape[0] - 1))
                        w = min(w, image.shape[1] - x)
                        h = min(h, image.shape[0] - y)
                        
                        if w > 0 and h > 0:
                            text_boxes.append((x, y, w, h))
                except Exception as e:
                    continue
        
        return text_boxes
    except Exception as e:
        st.warning(f"Text detection error: {str(e)}")
        return []


def create_text_mask(image: np.ndarray, text_boxes: List[tuple]) -> np.ndarray:
    """
    Create a mask for all text regions in the image.
    
    Args:
        image: Input image in BGR format
        text_boxes: List of bounding boxes (x, y, w, h) for text regions
    
    Returns:
        Binary mask where 255 represents text regions
    """
    # Create empty mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Fill text regions in the mask
    for x, y, w, h in text_boxes:
        # Add some padding to ensure entire text is captured
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        mask[y_start:y_end, x_start:x_end] = 255
    
    # Apply morphological operations to smooth the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask


def apply_text_color_splash(image: np.ndarray) -> np.ndarray:
    """
    Apply text color splash effect - convert image to grayscale except text regions.
    
    Args:
        image: Input image in BGR format
    
    Returns:
        Processed image with text preserved in color, rest in grayscale
    """
    # Detect text regions
    text_boxes = detect_text_regions(image)
    
    if not text_boxes:
        # If no text detected, convert entire image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Create mask for text regions
    text_mask = create_text_mask(image, text_boxes)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Apply mask: keep original color where text is, use grayscale elsewhere
    result = np.where(text_mask[:, :, np.newaxis] == 255, image, gray_bgr)
    
    return result.astype(np.uint8)


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Load image from Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Image as numpy array (BGR format) or None if error
    """
    try:
        # Read image bytes
        image_bytes = uploaded_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def convert_image_for_display(image: np.ndarray) -> Image.Image:
    """
    Convert BGR image to RGB for Streamlit display.
    
    Args:
        image: Image in BGR format (OpenCV)
    
    Returns:
        PIL Image in RGB format
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    return Image.fromarray(rgb_image)


def get_image_download_buffer(image: np.ndarray, format: str = 'PNG') -> io.BytesIO:
    """
    Convert image to bytes buffer for download.
    
    Args:
        image: Image in BGR format
        format: Image format ('PNG', 'JPEG', etc.)
    
    Returns:
        BytesIO buffer
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Save to buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    return buffer


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Process a single image with text color splash effect.
    
    Args:
        image: Input image as numpy array (BGR format)
    
    Returns:
        Processed image
    """
    return apply_text_color_splash(image)


def process_batch_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Process multiple images with text color splash effect.
    
    Args:
        images: List of input images (BGR format)
    
    Returns:
        List of processed images
    """
    processed = []
    for img in images:
        processed.append(process_image(img))
    return processed


def main():
    """
    Main Streamlit application for text color splash effect.
    """
    st.set_page_config(
        page_title="Text Color Splash",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Text Color Splash Effect")
    st.markdown("**Upload an image to automatically convert it to grayscale while preserving text in color.**")
    
    if not PADDLEOCR_AVAILABLE:
        st.error("⚠️ PaddleOCR is not installed. Please install it using: `pip install paddleocr`")
        st.stop()
    
    # Processing mode selection
    processing_mode = st.radio(
        "Processing Mode",
        ["Single Image", "Batch Processing"],
        horizontal=True,
        help="Process one image or multiple images at once"
    )
    
    # Main content area
    if processing_mode == "Single Image":
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            help="Upload a single image to process"
        )
        
        if uploaded_file is not None:
            # Load image
            image = load_image_from_upload(uploaded_file)
            
            if image is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(convert_image_for_display(image))
                    st.caption(f"Dimensions: {image.shape[1]} x {image.shape[0]} pixels")
                
                with col2:
                    st.subheader("✨ Processed Image")
                    
                    # Process image
                    with st.spinner("Detecting text and processing image..."):
                        processed = process_image(image)
                    
                    st.image(convert_image_for_display(processed))
                    
                    # Download button
                    download_buffer = get_image_download_buffer(processed, format='PNG')
                    st.download_button(
                        label="⬇️ Download Processed Image",
                        data=download_buffer,
                        file_name="text_colorsplash_result.png",
                        mime="image/png"
                    )
    
    else:  # Batch Processing
        st.subheader("📤 Upload Multiple Images")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            accept_multiple_files=True,
            help="Upload multiple images to process"
        )
        
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} image(s) uploaded")
            
            # Load all images
            images = []
            image_names = []
            for uploaded_file in uploaded_files:
                img = load_image_from_upload(uploaded_file)
                if img is not None:
                    images.append(img)
                    image_names.append(uploaded_file.name)
            
            if images:
                # Process all images
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed_images = []
                for i, img in enumerate(images):
                    status_text.text(f"Processing image {i+1} of {len(images)}...")
                    processed_images.append(process_image(img))
                    progress_bar.progress((i + 1) / len(images))
                
                status_text.text("✅ Processing complete!")
                progress_bar.empty()
                
                st.success(f"✅ Processed {len(processed_images)} image(s) successfully!")
                
                # Display results in a grid
                num_cols = 2
                for i in range(0, len(processed_images), num_cols):
                    cols = st.columns(num_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(processed_images):
                            with col:
                                st.image(
                                    convert_image_for_display(processed_images[idx]),
                                    caption=image_names[idx]
                                )
                                
                                # Download button for each image
                                download_buffer = get_image_download_buffer(
                                    processed_images[idx],
                                    format='PNG'
                                )
                                st.download_button(
                                    label=f"⬇️ Download {image_names[idx]}",
                                    data=download_buffer,
                                    file_name=f"text_colorsplash_{image_names[idx]}",
                                    mime="image/png",
                                    key=f"download_{idx}"
                                )
                
                # Batch download option (create ZIP)
                if st.button("📦 Download All as ZIP"):
                    import zipfile
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zip_file:
                            for idx, processed_img in enumerate(processed_images):
                                img_buffer = get_image_download_buffer(processed_img, format='PNG')
                                zip_file.writestr(
                                    f"text_colorsplash_{image_names[idx]}",
                                    img_buffer.read()
                                )
                        
                        # Read ZIP file
                        with open(tmp_zip.name, 'rb') as f:
                            zip_data = f.read()
                        
                        st.download_button(
                            label="⬇️ Download ZIP File",
                            data=zip_data,
                            file_name="text_colorsplash_batch.zip",
                            mime="application/zip",
                            key="batch_download"
                        )
                        
                        # Cleanup
                        os.unlink(tmp_zip.name)
    
    # Footer with instructions
    st.markdown("---")


if __name__ == "__main__":
    main()
