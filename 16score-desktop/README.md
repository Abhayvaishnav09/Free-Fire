# 16Score AI Desktop Application

A modern Python PyQt5 desktop application for real-time kill feed capture and esports score management with AI-powered analysis.

## 🚀 Features

- **Secure Login System**: Email-based authentication with token management
- **Modern Responsive UI**: Clean, adaptive interface with professional styling
- **Multi-Organization Support**: Select from different organizations with role-based access
- **Real-time Kill Feed Capture**: AI-powered detection using gRPC + YOLO
- **OCR Integration**: Automatic player name extraction from kill feed
- **SIFT Weapon Detection**: Template matching for weapon identification
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Executable Build**: PyInstaller support for standalone distribution

## 📁 Project Structure

```
16score-desktop/
├── main.py                     # Application entry point
├── config.json                 # Centralized configuration file
├── requirements.txt            # Python dependencies
├── 16ScoreAI.spec             # PyInstaller build specification
├── 16ScoreAI_Installer.iss    # Inno Setup installer script
│
├── src/
│   └── score_ai/              # Main application package
│       ├── __init__.py
│       │
│       ├── core/              # Core business logic
│       │   ├── config.py          # Static configuration constants
│       │   ├── config_manager.py  # Dynamic JSON config loader
│       │   └── api_service.py     # REST API client services
│       │
│       ├── detection/         # Kill feed detection modules
│       │   ├── killblocks.py          # Main detection logic
│       │   ├── killfeed_detections.py # Detection data management
│       │   ├── killfeed_service.proto # gRPC service definition
│       │   ├── killfeed_pb2.py        # Generated protobuf classes
│       │   ├── killfeed_pb2_grpc.py   # Generated gRPC stubs
│       │   └── single-template/       # Weapon template images
│       │
│       ├── ui/                # User interface
│       │   ├── main_window.py     # Main application window
│       │   ├── components/        # Reusable UI widgets
│       │   │   └── header_widget.py
│       │   └── pages/             # Application screens
│       │       ├── login_page.py
│       │       ├── home_page.py
│       │       ├── organization_page.py
│       │       ├── tournament_page.py
│       │       ├── match_page.py
│       │       ├── streaming_source_page.py
│       │       ├── camera_setup_pyqt.py
│       │       └── instruction_page.py
│       │
│       ├── utils/             # Utility modules
│       │   ├── helpers.py         # Common helper functions
│       │   └── responsive_utils.py # Responsive design utilities
│       │
│       └── resources/         # Static assets
│           └── images/
│               ├── TMS_logo.jpg
│               ├── background_img.png
│               └── kill_feed.svg
│
├── game_logs/                 # Runtime detection logs
│   ├── debug/
│   ├── match_positions/
│   ├── result.txt
│   └── ocr_detections.log
│
└── hooks/                     # PyInstaller hooks
    ├── hook-paddleocr.py
    ├── hook-paddlex.py
    ├── hook-score_ai.py
    └── hook-torch.py
```

## 🛠️ Requirements

- **Python**: 3.8 or higher
- **PyQt5**: GUI framework
- **OpenCV**: Image processing
- **gRPC**: Communication with detection server
- **rapidfuzz**: Player name matching

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 16score-desktop
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Application

```bash
python main.py
```

### Application Flow

1. **Login** → Enter credentials to authenticate
2. **Organization Selection** → Choose your organization/tenant
3. **Tournament Selection** → Pick active tournament
4. **Match Selection** → Select match to score
5. **Camera Setup** → Configure video capture source
6. **Live Detection** → Real-time kill feed capture and scoring

### Building Executable

```bash
pyinstaller 16ScoreAI.spec
```

The executable will be created in the `dist/` directory.

## 🔧 Configuration

### Configuration Files

1. **`config.json`** - Main configuration (use `config.example.json` as template)
2. **`.env`** - Environment variables for secrets (use `ENV_TEMPLATE.txt` as template)

### Setting Up Configuration

```bash
# 1. Copy example config
cp config.example.json config.json

# 2. Copy environment template
cp ENV_TEMPLATE.txt .env

# 3. Edit both files with your actual values
```

### Example `config.json`:

```json
{
  "api": {
    "backend_url": "https://webapi.16score.com",
    "timeout": 30
  },
  "grpc": {
    "killfeed_server": "127.0.0.1:50050",
    "use_tls": true
  },
  "security": {
    "enforce_https": true,
    "use_secure_storage": true
  }
}
```

### Environment Variables

Sensitive values can be set via environment variables (overrides config.json):

| Variable | Description |
|----------|-------------|
| `SCORE_API_URL` | Backend API URL |
| `SCORE_GRPC_SERVER` | gRPC server address |
| `SCORE_GRPC_API_KEY` | gRPC authentication key |
| `SCORE_GRPC_USE_TLS` | Enable TLS for gRPC (true/false) |
| `SCORE_ENFORCE_HTTPS` | Warn on HTTP URLs (true/false) |

### Key Configuration Options

| Setting | Description |
|---------|-------------|
| `api.backend_url` | Main backend server URL |
| `grpc.killfeed_server` | gRPC detection server address |
| `detection.frame_capture_interval` | Frame capture rate (seconds) |
| `detection.confidence_threshold` | Minimum detection confidence |
| `camera.default_index` | Default camera device index |

## 🏗️ Architecture

### Core Components

- **ConfigManager**: Singleton for centralized JSON configuration access
- **APIService**: REST client with auth token management and error handling
- **KillblockDetector**: Real-time kill feed detection with gRPC
- **ResponsiveUtils**: DPI-aware responsive UI scaling

### Detection Pipeline

```
Camera/Stream → Frame Capture → gRPC Server → YOLO Detection
                                    ↓
                              OCR Extraction
                                    ↓
                         SIFT Weapon Matching
                                    ↓
                          API Killfeed Post
```

## 🔒 Security Features

- **Environment Variable Support**: Secrets loaded from `.env` file (never committed)
- **Secure Token Storage**: Uses system keyring for persistent token storage
- **Input Validation**: Email and password validation before API calls
- **HTTPS Enforcement**: Warnings when using insecure HTTP connections
- **TLS Support**: Optional TLS encryption for gRPC connections
- **Bearer Token Auth**: Secure API authentication
- **Masked Password Input**: Password fields use secure echo mode
- **Gitignore Protection**: Sensitive files excluded from version control

### Security Best Practices

1. **Never commit** `config.json` or `.env` with real credentials
2. **Use HTTPS** for all production API URLs
3. **Enable TLS** for gRPC in production (`grpc.use_tls: true`)
4. **Use environment variables** for CI/CD deployments
5. **Rotate API keys** regularly

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Run `pip install -r requirements.txt` |
| gRPC connection failed | Check `grpc.killfeed_server` in config.json |
| Camera not detected | Verify camera index in config |
| PyInstaller build fails | Ensure all hooks are present |

### Debug Mode

Enable detailed logging in `config.json`:
```json
{
  "logging": {
    "level": "DEBUG"
  },
  "ocr": {
    "log_detections": true
  }
}
```

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

**Note**: This application requires a backend API server and gRPC detection server for full functionality.
