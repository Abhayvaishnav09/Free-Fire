# Free Fire Killblock Detection - Architecture Overview

## 🏗️ System Architecture

The system uses a **client-server architecture** with gRPC communication:

```
┌─────────────────────┐         gRPC          ┌─────────────────────┐
│                     │  ──────────────────>  │                     │
│  ffkillblock.py     │   Send Frame          │  grpc_block.py      │
│  (CLIENT)           │                       │  (SERVER)            │
│                     │  <──────────────────  │                     │
│  - Camera Capture   │   Return Results      │  - YOLO Detection   │
│  - OCR (Names)      │                       │  - Status Analysis  │
│  - API Push         │                       │  - Color Analysis   │
│  - File Saving      │                       │  - Image Cropping    │
└─────────────────────┘                       └─────────────────────┘
```

## 📁 File Roles

### 1. **grpc_block.py** = **SERVER** ✅ (ESSENTIAL)
   - **Role**: gRPC server that processes frames
   - **What it does**:
     - Loads YOLO model for killfeed detection
     - Receives full frames from client
     - Detects killblocks using YOLO
     - Crops killfeed regions
     - Determines status (KNOCKED/KILL/REVIVED) using color analysis
     - Returns processed results back to client
   - **Runs**: Continuously, listening on port 50051
   - **Can run**: 
     - Automatically (started by ffkillblock.py)
     - Manually: `python grpc_block.py --serve`

### 2. **ffkillblock.py** = **CLIENT** ✅ (ESSENTIAL)
   - **Role**: Main application that orchestrates everything
   - **What it does**:
     - Captures frames from OBS Virtual Camera
     - Sends frames to gRPC server for processing
     - Receives detection results from server
     - Extracts killer/victim names using OCR
     - Handles duplicate detection
     - Saves images locally
     - Pushes data to TMS API
   - **Runs**: Main entry point - `python ffkillblock.py`

## 🔄 How They Work Together

1. **You run**: `python ffkillblock.py`
2. **ffkillblock.py** checks if server is running on port 50051
3. **If not running**: ffkillblock.py automatically starts grpc_block.py as a background process
4. **ffkillblock.py** connects to the server
5. **For each frame**:
   - ffkillblock.py captures frame from camera
   - Sends frame to grpc_block.py via gRPC
   - grpc_block.py processes frame and returns results
   - ffkillblock.py handles OCR, saving, and API push

## ✅ Is grpc_block.py Useful?

**YES! It's ESSENTIAL!** 

- **grpc_block.py** is the **SERVER** that does all the heavy processing:
  - YOLO model loading and inference
  - Killfeed detection
  - Status determination (color analysis)
  - Image cropping

- **Without it**: The system won't work! ffkillblock.py needs the server to process frames.

## 🚀 Usage

### Option 1: Automatic (Recommended)
```bash
python ffkillblock.py
```
- Server starts automatically if not running
- Everything handled for you

### Option 2: Manual Server Start
```bash
# Terminal 1: Start server
python grpc_block.py --serve --model best.pt --port 50051

# Terminal 2: Start client
python ffkillblock.py
```

## 📊 Summary

| File | Type | Purpose | Required? |
|------|------|---------|-----------|
| **grpc_block.py** | **SERVER** | Process frames, detect killfeeds, determine status | ✅ **YES** |
| **ffkillblock.py** | **CLIENT** | Capture frames, OCR, save files, push to API | ✅ **YES** |
| killfeed_detection.proto | Protocol | Defines gRPC service interface | ✅ **YES** |
| killfeed_detection_pb2*.py | Generated | gRPC Python code (auto-generated) | ✅ **YES** |

**Both files are essential and work together!**

