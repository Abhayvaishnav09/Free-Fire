# 16Score AI Suite - Professional Installer

This guide explains how to create a professional installer for the 16Score AI Suite using Inno Setup Compiler.

## 🎯 Features

### **Beautiful Splash Screen Loader**
- Custom gradient background
- Progress bar with real-time updates
- Loading status messages
- Professional branding

### **Professional Installation Experience**
- Modern wizard interface
- System requirements checking
- Desktop shortcut (enabled by default)
- Administrator mode (enabled by default)
- All dependencies included

### **Complete Application Package**
- Standalone executable
- All AI models and libraries
- Configuration files
- Documentation

## 🛠️ Prerequisites

### Required Software
1. **Inno Setup 6**
   - Download from: https://jrsoftware.org/isdl.php
   - Install with default settings

2. **Python 3.8+ with PyInstaller**
   - For building the executable

### Required Files
Make sure these files are present:
- `main.py` - Main application with splash screen
- `16ScoreAI.spec` - PyInstaller specification
- `16ScoreAI_Installer.iss` - Inno Setup script
- `16score_icon.ico` - Application icon
- `16Score_logo.png` - Application logo

## 🚀 Building the Professional Installer

### Step 1: Build the Executable
```bash
pyinstaller 16ScoreAI.spec --clean
```

### Step 2: Create the Installer
```bash
build_with_inno.bat
```

Or manually:
```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" 16ScoreAI_Installer.iss
```

## 📦 Installer Features

### **Splash Screen Loader**
The application now includes a beautiful splash screen that displays:
- **Gradient Background** - Professional dark blue-gray gradient
- **Progress Bar** - Real-time loading progress
- **Status Messages** - Step-by-step loading information
- **Branding** - 16Score AI Suite branding and version

### **Loading Steps**
1. Initializing AI models...
2. Loading computer vision libraries...
3. Preparing OCR engine...
4. Setting up camera detection...
5. Configuring neural networks...
6. Loading weapon recognition models...
7. Initializing tournament system...
8. Setting up real-time processing...
9. Preparing user interface...
10. Starting 16Score AI Suite...

### **Installation Features**
- **Desktop Shortcut** - Created automatically
- **Administrator Mode** - Runs with admin privileges by default
- **Start Menu Integration** - Professional start menu entry
- **File Associations** - Optional .16score file associations
- **System Requirements Check** - Validates system before installation

### **Professional UI**
- Modern wizard interface
- System requirements validation
- Installation type selection
- Progress tracking
- Professional branding

## 📁 Output Files

### Generated Files
- **Executable**: `dist/16ScoreAI.exe` (~2-3 GB)
- **Installer**: `installer_output/16ScoreAI_Installer_v1.0.0.exe` (~2-3 GB)

### File Structure
```
installer_output/
├── 16ScoreAI_Installer_v1.0.0.exe
└── (other installer files)

dist/
├── 16ScoreAI.exe
└── (executable files)
```

## 🎨 Splash Screen Design

### **Visual Elements**
- **Background**: Dark blue-gray gradient
- **Title**: "16Score AI Suite" (24pt, bold)
- **Subtitle**: "AI-Powered Esports Analysis" (12pt)
- **Version**: "Version 1.0.0" (9pt)
- **Progress Bar**: Blue gradient with rounded corners
- **Status Text**: White text with real-time updates

### **Color Scheme**
- **Primary**: #4A90E2 (Blue)
- **Secondary**: #357ABD (Dark Blue)
- **Background**: #191923 to #414151 (Gradient)
- **Text**: #FFFFFF (White)
- **Subtitle**: #B0B0B0 (Light Gray)

## 🔧 Customization

### **Modifying the Splash Screen**
Edit `main.py` to customize:
- Colors and gradients
- Loading messages
- Progress timing
- Visual elements

### **Modifying the Installer**
Edit `16ScoreAI_Installer.iss` to customize:
- Application information
- Installation options
- File associations
- System requirements

### **Version Management**
Update version numbers in:
- `16ScoreAI_Installer.iss` (AppVersion)
- `main.py` (splash screen version)
- `INSTALL_INFO.txt` and `POST_INSTALL_INFO.txt`

## 📋 Installation Process

### **For End Users**
1. **Download** the installer
2. **Run** as administrator
3. **Follow** the installation wizard
4. **Launch** from desktop shortcut
5. **Enjoy** the beautiful splash screen!

### **Installation Steps**
1. **Welcome** - Introduction and system check
2. **Requirements** - System validation
3. **Installation Type** - Full or minimal
4. **Directory** - Installation location
5. **Installation** - File copying and setup
6. **Finish** - Launch application

## 🎯 Professional Features

### **Administrator Mode**
- Runs as administrator by default
- Registry entry for persistent admin mode
- Optimal performance for AI operations

### **Desktop Integration**
- Desktop shortcut (enabled by default)
- Start Menu integration
- File associations (optional)
- Professional uninstallation

### **System Integration**
- Windows Programs and Features
- Proper registry entries
- Clean uninstallation
- User data management

## 🚀 Distribution

### **Professional Package**
- Single installer file
- All dependencies included
- No additional downloads required
- Professional branding

### **File Sizes**
- **Executable**: ~2-3 GB (includes all AI models)
- **Installer**: ~2-3 GB (includes executable)
- **Compressed**: ~1-2 GB (using 7-Zip)

### **Distribution Methods**
1. **Direct Download** - Host installer on website
2. **USB Distribution** - Portable installation
3. **Network Deployment** - Enterprise deployment
4. **Cloud Storage** - Google Drive, Dropbox, etc.

## 🔍 Troubleshooting

### **Common Issues**

#### Installer Won't Build
```bash
# Check Inno Setup installation
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" --version

# Check PyInstaller build
pyinstaller 16ScoreAI.spec --clean
```

#### Splash Screen Issues
- Ensure PyQt5 is properly installed
- Check for missing dependencies
- Verify main.py imports

#### Executable Issues
- Run as administrator
- Check Windows Defender settings
- Verify all dependencies included

### **Build Logs**
- **PyInstaller**: Check console output
- **Inno Setup**: Check `installer_output/` directory
- **Application**: Check `game_logs/` directory

## 📞 Support

For issues and questions:
- **GitHub Issues**: https://github.com/ItsBluexkye/16score-desktop/issues
- **Documentation**: Check README files
- **Build Scripts**: Review build scripts for errors

## 📄 License

This installer system is part of the 16Score AI Suite project.
See LICENSE.txt for licensing information.

---

**🎉 Your professional installer is ready for distribution!** 