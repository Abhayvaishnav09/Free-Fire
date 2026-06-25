# 16Score AI Suite - Installer Guide

This guide explains how to build and distribute the 16Score AI Suite application using the provided installer system.

## Prerequisites

### Required Software
1. **Python 3.8 or later**
   - Download from: https://python.org
   - Make sure Python is added to PATH during installation

2. **Inno Setup 6** (for creating installers)
   - Download from: https://jrsoftware.org/isdl.php
   - Install with default settings

### Required Files
Make sure these files are present in your project directory:
- `main.py` - Main application entry point
- `16ScoreAI.spec` - PyInstaller specification file
- `requirements.txt` - Python dependencies
- `16score_icon.ico` - Application icon
- `16Score_logo.png` - Application logo
- `16ScoreAI_Installer.iss` - Inno Setup installer script

## Building the Application

### Option 1: Using the Batch File (Recommended)
```bash
# Simple build with all features
build.bat

# Build with specific options
build.bat --clean --version 1.1.0
```

### Option 2: Using Python Script Directly
```bash
# Basic build
python build_exe.py

# Build with options
python build_exe.py --clean --skip-deps --version 1.1.0
```

### Option 3: Using PowerShell Script
```powershell
# Build installer only (requires existing PyInstaller build)
.\Build-Installer.ps1

# Build with specific version
.\Build-Installer.ps1 -Version 1.1.0 -Clean
```

## Build Options

### Python Build Script Options
- `--skip-deps` - Skip dependency installation
- `--skip-installer` - Skip installer creation
- `--clean` - Clean build artifacts before building
- `--version <version>` - Set version number (default: 1.0.0)

### PowerShell Script Options
- `-InnoSetupPath <path>` - Path to ISCC.exe
- `-Version <version>` - Version number
- `-Clean` - Clean build directory
- `-Help` - Show help message

## Build Process

The build process consists of several steps:

1. **Dependency Check** - Verifies all required files are present
2. **Clean Build** - Removes previous build artifacts (if requested)
3. **Dependency Installation** - Installs/updates Python packages
4. **Executable Build** - Creates standalone executable using PyInstaller
5. **Installer Creation** - Creates Windows installer using Inno Setup

## Output Files

### Executable
- **Location**: `dist/16ScoreAI.exe`
- **Size**: ~2-3 GB (includes all dependencies)
- **Type**: Standalone executable (no installation required)

### Installer
- **Location**: `installer_output/16ScoreAI_Installer_v<version>.exe`
- **Size**: ~2-3 GB (includes executable and installer overhead)
- **Type**: Windows installer with automatic dependency management

## Installer Features

The generated installer includes:

### System Requirements Check
- Windows 10 or later (64-bit)
- 4GB RAM minimum (8GB recommended)
- DirectX 11 compatible graphics card
- 2GB free disk space

### Installation Options
- **Full Installation** - Complete application with all features
- **Minimal Installation** - Basic application only
- **Desktop Shortcut** - Optional desktop icon
- **Start Menu Integration** - Automatic start menu entry
- **File Associations** - Associate .16score files (optional)

### Post-Installation
- Automatic game_logs directory creation
- Registry entries for uninstallation
- Optional application launch after installation
- README file access

## Distribution

### For End Users
1. **Installer** (Recommended)
   - Provides professional installation experience
   - Handles system requirements
   - Easy uninstallation
   - File associations

2. **Standalone Executable**
   - No installation required
   - Portable (can be run from USB drive)
   - Larger file size
   - Manual dependency management

### File Sizes
- **Executable**: ~2-3 GB
- **Installer**: ~2-3 GB
- **Compressed**: ~1-2 GB (using 7-Zip or similar)

## Troubleshooting

### Common Issues

#### PyInstaller Build Fails
```bash
# Clean and rebuild
python build_exe.py --clean

# Check for missing dependencies
python -m pip install -r requirements.txt
```

#### Inno Setup Not Found
```bash
# Install Inno Setup 6
# Download from: https://jrsoftware.org/isdl.php

# Or specify custom path
.\Build-Installer.ps1 -InnoSetupPath "C:\Custom\Path\ISCC.exe"
```

#### Missing Dependencies
```bash
# Reinstall all dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

#### Large File Size
- The application includes AI models and libraries
- File size is normal for AI applications
- Consider using compression for distribution

### Build Logs
- PyInstaller logs: Check console output during build
- Inno Setup logs: Check `installer_output/` directory
- Application logs: Check `game_logs/` directory after installation

## Customization

### Modifying the Installer
Edit `16ScoreAI_Installer.iss` to customize:
- Application information
- Installation options
- File associations
- System requirements
- Visual appearance

### Version Management
```bash
# Update version in all files
python build_exe.py --version 1.1.0
```

### Adding Files to Installer
Add files to the `[Files]` section in `16ScoreAI_Installer.iss`:
```ini
Source: "path/to/file"; DestDir: "{app}"; Flags: ignoreversion
```

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/ItsBluexkye/16score-desktop/issues
- **Documentation**: Check README.md in the project root
- **Build Scripts**: Review build_exe.py and Build-Installer.ps1

## License

This installer system is part of the 16Score AI Suite project.
See LICENSE.txt for licensing information. 