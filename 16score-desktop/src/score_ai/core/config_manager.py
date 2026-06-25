"""
Configuration Manager for 16Score AI Application

Manages application configuration with support for:
- JSON configuration files
- Environment variable overrides for sensitive values
- Secure defaults
"""

import json
import os
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to load dotenv for environment variable support
try:
    from dotenv import load_dotenv
    # Load .env file if it exists
    if os.path.exists(".env"):
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
except ImportError:
    pass  # python-dotenv not installed, use system env vars only


class ConfigManager:
    """
    Configuration manager for loading and accessing JSON configuration.
    
    Supports environment variable overrides for sensitive values:
    - SCORE_API_URL: Overrides api.backend_url
    - SCORE_GRPC_SERVER: Overrides grpc.killfeed_server
    - SCORE_GRPC_API_KEY: Overrides grpc.api_key
    - SCORE_GRPC_USE_TLS: Overrides grpc.use_tls
    """
    
    _instance = None
    _config = None
    
    # Environment variable mappings (env_var -> config_path)
    ENV_OVERRIDES = {
        "SCORE_API_URL": "api.backend_url",
        "SCORE_API_TIMEOUT": "api.timeout",
        "SCORE_GRPC_SERVER": "grpc.killfeed_server",
        "SCORE_GRPC_API_KEY": "grpc.api_key",
        "SCORE_GRPC_USE_TLS": "grpc.use_tls",
        "SCORE_OCR_API_KEY": "ocr.api_key",
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
            self._apply_env_overrides()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            # Try to find config.json in the current directory first
            config_paths = [
                "config.json",  # Current directory
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.json"),  # Project root
                os.path.join(os.path.dirname(__file__), "..", "..", "config.json"),  # Alternative path
            ]
            
            config_loaded = False
            for config_path in config_paths:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self._config = json.load(f)
                    logger.info(f"Configuration loaded from: {config_path}")
                    config_loaded = True
                    break
            
            if not config_loaded:
                # Create default config if none exists
                self._create_default_config()
                logger.info("Default configuration created")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_config()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides for sensitive config values"""
        for env_var, config_path in self.ENV_OVERRIDES.items():
            env_value = os.environ.get(env_var)
            if env_value:
                self._set_nested(config_path, env_value)
                logger.debug(f"Config override from env: {config_path}")
    
    def _set_nested(self, key_path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation"""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Handle type conversion for known types
        final_key = keys[-1]
        if final_key in ('timeout', 'default_index', 'default_width', 'default_height'):
            value = int(value)
        elif final_key in ('use_tls', 'log_detections', 'enforce_https'):
            value = value.lower() == 'true'
        
        config[final_key] = value
    
    def _create_default_config(self):
        """Create default configuration"""
        self._config = {
            "api": {
                "backend_url": "https://webapi.16score.com",
                "timeout": 30
            },
            "grpc": {
                "killfeed_server": "127.0.0.1:50050",
                "api_key": "",
                "use_tls": False,
                "tls_cert_path": ""
            },
            "security": {
                "enforce_https": True,
                "use_secure_storage": True
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
                "enable_stage_detection": False,
                "max_queue_size": 100,
                "process_alternate_frames": False,
                "frame_queue_drop_oldest_when_full": True,
                "num_grpc_worker_threads": 4,
                "killfeed_api_pool_workers": 5,
                "grpc_input_width": 1280,
                "grpc_input_height": 720,
                "grpc_jpeg_quality": 90,
                "grpc_use_compression": False,
                "log_grpc_ocr_table": False,
                "killfeed_roi_enabled": False,
                "killfeed_roi": [0.62, 0.0, 1.0, 0.42],
                "killfeed_roi_min_size_px": 32
            },
            "ocr": {
                "use_angle_cls": False,
                "lang": "en",
                "log_detections": False
            },
            "camera": {
                "default_width": 1920,
                "default_height": 1080,
                "default_index": 1
            },
            "processing": {
                "image_quality": 92,
                "image_resize_factor": 0.75,
                "duplicate_check_count": 5,
                "similarity_threshold_high": 0.90,
                "similarity_threshold_low": 0.50
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            },
            "api_weapons": [
                "molotov-kill",
                "molotov-knockout",
                "carblast-knockout",
                "carblast-kill",
                "car-kill",
                "car-knockout",
                "car-blast-kill",
                "car-blast-knockout",
                "carblow-kill",
                "carblow-knockout",
                "grenade-knockout",
                "grenade-kill",
                "playzone-kill",
                "playzone-knockout",
                "self-knockout",
                "self-kill",
                "kill",
                "knife-kill",
                "pan-kill",
                "pan-knockout",
                "knife-knockout",
                "knife-head-kill",
                "knife-head-knockout"
            ]
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation (e.g., 'api.backend_url')
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                value = value[key]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_url(self, endpoint: str) -> str:
        """
        Get full API URL for an endpoint
        
        Args:
            endpoint: Endpoint name from config
            
        Returns:
            Full API URL
        """
        backend_url = self.get('api.backend_url', '').rstrip('/')
        endpoint_path = self.get(f'endpoints.{endpoint}', '').lstrip('/')
        return f"{backend_url}/{endpoint_path}"
    
    def get_grpc_config(self) -> Dict[str, Any]:
        """Get gRPC killfeed detection service configuration"""
        return {
            'server': self.get('grpc.killfeed_server', '127.0.0.1:50050'),
            'api_key': self.get('grpc.api_key', ''),
            'timeout': self.get('api.timeout', 30),
            'use_tls': self.get('grpc.use_tls', False),
            'tls_cert_path': self.get('grpc.tls_cert_path', '')
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'enforce_https': self.get('security.enforce_https', True),
            'use_secure_storage': self.get('security.use_secure_storage', True)
        }
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR configuration"""
        return {
           
            'lang': self.get('ocr.lang'),
            'log_detections': self.get('ocr.log_detections', False)
        }
    
    def _get_user_setting(self, key: str, default: Any = None) -> Any:
        """Get user preference from .esports_ai_config.json (app-only settings)"""
        config_path = os.path.join(os.path.expanduser("~"), ".esports_ai_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get(key, default)
            except Exception:
                pass
        return default

    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration (game from app settings, not config.json)"""
        return {
            'frame_capture_interval': self.get('detection.frame_capture_interval'),
            'confidence_threshold': self.get('detection.confidence_threshold'),
            'sift_min_matches': self.get('detection.sift_min_matches'),
            'sift_match_ratio': self.get('detection.sift_match_ratio'),
            'sift_features': self.get('detection.sift_features'),
            'game': self._get_user_setting('game', 'freefire')  # BGMI or FreeFire - set in Settings
        }
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return {
            'default_width': self.get('camera.default_width'),
            'default_height': self.get('camera.default_height'),
            'default_index': self.get('camera.default_index')
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return {
            'image_quality': self.get('processing.image_quality'),
            'image_resize_factor': self.get('processing.image_resize_factor'),
            'duplicate_check_count': self.get('processing.duplicate_check_count'),
            'similarity_threshold_high': self.get('processing.similarity_threshold_high'),
            'similarity_threshold_low': self.get('processing.similarity_threshold_low')
        }
    
    def get_api_weapons(self) -> list:
        """Get list of API weapons"""
        return self.get('api_weapons', [])
    
    def reload_config(self):
        """Reload configuration from file"""
        self._config = None
        self._load_config()
    
    def save_config(self, config_path: str = "config.json") -> bool:
        """Save current configuration to file"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False


# Global instance
config = ConfigManager() 