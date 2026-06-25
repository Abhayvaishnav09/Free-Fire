# 16Score AI - Centralized Configuration Guide

## Overview
The 16Score AI application now uses a centralized configuration system through `config.json`, ensuring that all URLs, settings, and parameters can be easily modified in one place.

## Configuration Files

### Primary Configuration: `config.json`
This is the main configuration file that controls all application behavior:

```json
{
  "api": {
    "backend_url": "https://webapi.16score.com/",
    "yolo_api_url": "http://192.168.29.224:8000/api/v1/yolo/predict", 
    "yolo_api_key": "test-api-key-123",
    "timeout": 30
  },
  "endpoints": {
    "match_info": "LeagueMatchData/LeagueMatch/info",
    "team_players": "LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players",
    "killfeed": "LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed",
    "match_states": "LeagueMatch/getMatchStates/details", 
    "start_match": "LeagueMatch/startMatchById",
    "user_login": "User/Login",
    "user_from_token": "User/GetUserFromToken",
    "user_tenants": "Tenant/CurrentUserTenants",
    "get_leagues": "League/GetAllLeagues",
    "today_matches": "LeagueMatch/getTodayLeagueMatch",
    "league_groups": "LeagueGroup/getAllLeagueGroup",
    "matches_by_league": "LeagueMatch/getMatchesByLeagueId"
  },
  "detection": {
    "frame_capture_interval": 0.12,
    "confidence_threshold": 0.20,
    "sift_min_matches": 10,
    "sift_match_ratio": 0.75,
    "sift_features": 400,
    "max_queue_size": 100,
    "process_alternate_frames": false,
    "frame_queue_drop_oldest_when_full": true,
    "num_grpc_worker_threads": 4,
    "killfeed_api_pool_workers": 5
  },
  "ocr": {
    "use_angle_cls": true,
    "lang": "en", 
    "use_gpu": false,
    "det_db_box_thresh": 0.4,
    "drop_score": 0.3,
    "show_log": false
  },
  "camera": {
    "default_width": 1920,
    "default_height": 1080,
    "default_index": 1
  },
  "processing": {
    "image_quality": 85,
    "image_resize_factor": 0.75,
    "duplicate_check_count": 5,
    "similarity_threshold_high": 0.90,
    "similarity_threshold_low": 0.50
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
  }
}
```

## How Configuration Works

### Configuration Manager (`src/score_ai/core/config_manager.py`)
The `ConfigManager` class provides a singleton interface to access configuration:

```python
from score_ai.core.config_manager import config

# Get simple values
backend_url = config.get('api.backend_url')
timeout = config.get('api.timeout', 30)  # with default

# Get full API URLs
login_url = config.get_api_url('user_login')
# Returns: "https://webapi.16score.com/User/Login"

# Get configuration sections
camera_config = config.get_camera_config()
detection_config = config.get_detection_config()
```

### Legacy Config Wrapper (`src/score_ai/core/config.py`)
For backward compatibility, the old `Config` class has been updated to use the config manager:

```python
from score_ai.core.config import Config

# These now dynamically read from config.json
backend_url = Config.get_backend_url()
api_endpoint = Config.get_api_endpoint('User/Login')
```

## Updated Components

### 1. API Service (`src/score_ai/core/api_service.py`)
All hardcoded endpoints now use configuration:

```python
# Before: hardcoded URLs
return self.post("/User/Login", data=data)

# After: config-based URLs  
endpoint = config.get_api_url('user_login')
return self.post(endpoint, data=data)
```

### 2. Camera Setup (`src/score_ai/ui/pages/camera_setup_pyqt.py`)
- Start match API call uses config
- Camera resolutions use config defaults
- All settings dynamically read from config

### 3. Detection System (`src/score_ai/detection/killblocks.py`)
- Frame capture intervals from config
- Confidence thresholds from config
- All detection parameters configurable

## Making Configuration Changes

### Change Backend URL
Edit `config.json`:
```json
{
  "api": {
    "backend_url": "https://your-new-api.example.com/"
  }
}
```

### Change YOLO API
Edit `config.json`:
```json
{
  "api": {
    "yolo_api_url": "http://your-yolo-server:8000/api/v1/yolo/predict",
    "yolo_api_key": "your-api-key"
  }
}
```

### Adjust Detection Settings
Edit `config.json`:
```json
{
  "detection": {
    "frame_capture_interval": 0.3,  // Faster capture
    "confidence_threshold": 0.15    // Lower threshold
  }
}
```

### Add New Endpoints
Edit `config.json`:
```json
{
  "endpoints": {
    "new_endpoint": "api/v2/new-feature"
  }
}
```

Then use in code:
```python
endpoint = config.get_api_url('new_endpoint')
```

## Configuration Precedence

1. **Primary**: `config.json` in project root
2. **Fallback**: Built-in defaults in `config_manager.py`
3. **Legacy**: Environment variables (limited support)

## Benefits

✅ **Single Source of Truth**: All configuration in one file  
✅ **Environment Flexibility**: Easy to switch between dev/staging/prod  
✅ **No Hardcoded Values**: All URLs and settings configurable  
✅ **Dynamic Updates**: Changes take effect without code modification  
✅ **Backward Compatibility**: Existing code continues to work  
✅ **Type Safety**: Configuration validation and defaults  

## Testing Configuration Changes

1. **Backup**: Always backup `config.json` before changes
2. **Validate**: Ensure JSON syntax is correct
3. **Test**: Verify application starts without errors
4. **Monitor**: Check logs for configuration loading messages

## Troubleshooting

### Configuration Not Loading
- Check `config.json` syntax with JSON validator
- Verify file exists in project root
- Check console for "Configuration loaded from:" message

### API Calls Failing
- Verify `backend_url` is correct and accessible
- Check endpoint names match exactly
- Ensure API key is valid for external services

### Detection Issues
- Adjust `confidence_threshold` for better/worse detection
- Modify `frame_capture_interval` for faster/slower processing
- Check camera settings match your hardware

## Future Enhancements

- Hot reloading of configuration without restart
- Environment-specific config files
- Configuration validation schemas
- Web-based configuration interface 