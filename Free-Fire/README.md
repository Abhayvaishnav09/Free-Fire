# Free Fire Killblock Detector

A Python-based detection system that automatically detects kill notifications from Free Fire gameplay using OBS Virtual Camera. The system uses YOLOv11 for object detection and PaddleOCR for text extraction to identify kill blocks, extract player names, and determine kill status (KNOCKED, FINISHED, or REVIVED).

## Features

- 🎯 **Automatic Killblock Detection**: Real-time detection of kill notifications using YOLOv11
- 📝 **Text Extraction**: OCR-based extraction of killer and victim player names
- 🎨 **Status Detection**: 
  - **KNOCKED**: White text (player knocked down)
  - **FINISHED**: Red text (player eliminated)
  - **REVIVED**: Green indicators (player revived)
- 🖼️ **Color Splash Effect**: Selective color processing - keeps text/icons in color, rest in grayscale
- 📸 **Auto-Save**: Automatically saves detected killblocks with formatted filenames
- 🔄 **Duplicate Prevention**: Smart duplicate detection to avoid processing same kill multiple times
- 📹 **OBS Integration**: Works seamlessly with OBS Virtual Camera

## Requirements

- Python 3.8+
- OBS Studio with Virtual Camera enabled
- YOLO model file (`best.pt`)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd "Free Fire"
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place YOLO model**:
   - Download or obtain `best.pt` (YOLO model file)
   - Place it in the project root directory

4. **Install PaddleOCR** (if not included in requirements):
   ```bash
   pip install paddlepaddle paddleocr
   ```

## Usage

1. **Start OBS Virtual Camera**:
   - Open OBS Studio
   - Configure your scene
   - Start Virtual Camera (Tools → Start Virtual Camera)

2. **Run the detector**:
   ```bash
   python ffkillblock.py
   ```

3. **Wait for prompt**:
   - The script will prompt you to start OBS Virtual Camera
   - Press Enter to begin detection

4. **Detection**:
   - The script continuously monitors the camera feed
   - Detected killblocks are automatically saved to `cropkillblock/` folder
   - Press `Ctrl+C` to stop detection

## Output Format

Detected killblocks are saved with the following filename format:
```
{ORDER_NUMBER}_{KILLER_NAME} {STATUS} {VICTIM_NAME}.png
```

Example:
```
001_GENS.AYUSH KNOCKED GENS.ETAZ015.png
002_TT.FLASH11 FINISHED TG.MAF1A7.png
003_K98.B0TG0D REVIVED K98.S0UL.png
```

## Project Structure

```
Free Fire/
├── ffkillblock.py          # Main detection script
├── text.py                 # Text detection module (PaddleOCR)
├── best.pt                 # YOLO model file (not included in repo)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── cropkillblock/         # Output directory for detected images
├── ff.py                  # Alternative/test script
└── colorsplash.py         # Color splash effect utilities
```

## Configuration

### Camera Settings
- Default: Camera index 1 (OBS Virtual Camera)
- Resolution: 1920x1080
- FPS: 1 frame per second (configurable in code)

### Detection Thresholds
- **Killblock Detection**: 65% confidence
- **Revive Detection**: 28% confidence (high precision)
- **Color Analysis**: HSV-based thresholds for KNOCKED/FINISHED

### Duplicate Window
- Default: 8 seconds (prevents duplicate detections)

## Status Detection Logic

1. **REVIVED Detection** (Highest Priority):
   - Primary: YOLOv11 high-confidence detection (≥0.28)
   - Fallback: Green HSV color validation
   - Takes absolute priority over color analysis

2. **KNOCKED/FINISHED Detection**:
   - **KNOCKED**: White HSV color detection (high brightness, low saturation)
   - **FINISHED**: Red HSV color detection (high saturation)
   - Only used if REVIVED is not detected

## Troubleshooting

### Camera Not Detected
- Ensure OBS Virtual Camera is running
- Check camera index in code (default: 1)
- Try different camera indices (0, 1, 2, etc.)

### Model Not Found
- Ensure `best.pt` is in the project root
- Check file permissions

### Low Detection Accuracy
- Adjust confidence thresholds in code
- Ensure good lighting/visibility in OBS scene
- Check camera resolution settings

### OCR Errors
- Ensure PaddleOCR is properly installed
- Check if text is clearly visible in kill notifications
- Model may need retraining for better accuracy

## Performance

- **Detection Rate**: ~1 FPS (configurable)
- **Processing**: CPU-based (YOLO on CPU, PaddleOCR on CPU)
- **Memory**: Moderate (depends on model size)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Notes

- The YOLO model (`best.pt`) is not included in the repository due to size
- Generated images in `cropkillblock/` are excluded from git
- Model file must be obtained separately or trained custom
