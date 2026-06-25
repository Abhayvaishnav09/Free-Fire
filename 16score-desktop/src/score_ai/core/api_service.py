"""
API Service for 16Score AI Application

This module provides a centralized service for all API operations including
authentication, error handling, and request management.
"""

import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from score_ai.core.config import Config
from .config_manager import config

logger = logging.getLogger(__name__)


def _check_url_security(url: str) -> None:
    """Log warning if URL is not using HTTPS"""
    if url and not url.startswith("https://"):
        enforce_https = config.get('security.enforce_https', True)
        if enforce_https:
            logger.warning(
                f"SECURITY WARNING: URL is using HTTP instead of HTTPS: {url}. "
                "Set security.enforce_https=false in config to disable this warning."
            )

@dataclass
class APIResponse:
    """Standardized API response wrapper"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    raw_response: Optional[requests.Response] = None

class APIService:
    """Centralized API service for all HTTP operations"""
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url or config.get('api.backend_url', 'http://192.168.1.11:5006')
        self.timeout = timeout
        self.session = requests.Session()
        self._auth_token: Optional[str] = None
        
        # Check URL security on initialization
        _check_url_security(self.base_url)
        
    def set_auth_token(self, token: str) -> None:
        """Set authentication token for subsequent requests"""
        self._auth_token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        logger.info("Authentication token set")
    
    def clear_auth_token(self) -> None:
        """Clear authentication token"""
        self._auth_token = None
        if "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]
        logger.info("Authentication token cleared")
    
    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make HTTP request with standardized error handling"""
        
        # Check if endpoint is already a full URL
        if endpoint.startswith(('http://', 'https://')):
            url = endpoint
        else:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers(headers)
        try:
            logger.info(f"Making {method.upper()} request to {endpoint}")
            
            response = self.session.request(
                method=method.upper(),
                url=url,
                json=data,
                params=params,
                headers=request_headers,
                timeout=self.timeout
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            # Handle successful responses
            if 200 <= response.status_code < 300:
                try:
                    response_data = response.json() if response.content else {}
                    return APIResponse(
                        success=True,
                        data=response_data,
                        status_code=response.status_code,
                        raw_response=response
                    )
                except ValueError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    return APIResponse(
                        success=True,
                        data={"message": "Request successful but no JSON response"},
                        status_code=response.status_code,
                        raw_response=response
                    )
            
            # Handle error responses
            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', f"HTTP {response.status_code}")
                except ValueError:
                    error_message = f"HTTP {response.status_code}: {response.text}"
                
                logger.error(f"API error: {error_message}")
                return APIResponse(
                    success=False,
                    error_message=error_message,
                    status_code=response.status_code,
                    raw_response=response
                )
                
        except Timeout:
            error_msg = f"Request timeout for {endpoint}"
            logger.error(error_msg)
            return APIResponse(success=False, error_message=error_msg)
            
        except ConnectionError:
            error_msg = f"Connection error for {endpoint}. Please check your internet connection."
            logger.error(error_msg)
            return APIResponse(success=False, error_message=error_msg)
            
        except RequestException as e:
            error_msg = f"Request failed for {endpoint}: {str(e)}"
            logger.error(error_msg)
            return APIResponse(success=False, error_message=error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error for {endpoint}: {str(e)}"
            logger.error(error_msg)
            return APIResponse(success=False, error_message=error_msg)
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make GET request"""
        return self._make_request("GET", endpoint, params=params, headers=headers)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make POST request"""
        return self._make_request("POST", endpoint, data=data, headers=headers)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make PUT request"""
        return self._make_request("PUT", endpoint, data=data, headers=headers)
    
    def delete(self, endpoint: str, headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make DELETE request"""
        return self._make_request("DELETE", endpoint, headers=headers)
    
    def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make PATCH request"""
        return self._make_request("PATCH", endpoint, data=data, headers=headers)

    def get_tournaments(self, page_number=1, page_size=6, filter_by='status', filter_value=1):
        """Fetches the list of tournaments (leagues) from the API with pagination."""
        endpoint = config.get('endpoints.get_leagues', 'League/GetAllLeagues')
        params = {
            'PageNumber': page_number,
            'PageSize': page_size,
            'filterBy': filter_by,
            'filterValue': filter_value
        }
        return self._make_request("GET", endpoint, params=params)

    def get_today_league_matches(self, league_id=None):
        """Fetch today's matches for a specific league."""
        base_endpoint = config.get('endpoints.today_matches', 'LeagueMatch/getTodayLeagueMatch')
        if league_id:
            endpoint = f"{base_endpoint}?leagueId={league_id}"
        else:
            endpoint = base_endpoint
        return self._make_request("GET", endpoint)

    def get_all_league_groups(self):
        """Fetch all league groups."""
        endpoint = config.get('endpoints.league_groups', 'LeagueGroup/getAllLeagueGroup')
        return self._make_request("GET", endpoint)

    def get_matches_by_league_id(self, league_id):
        """Fetch all matches for a given league/tournament ID."""
        endpoint = config.get('endpoints.matches_by_league', 'LeagueMatch/getMatchesByLeagueId')
        params = {'leagueId': league_id}
        return self._make_request("GET", endpoint, params=params)

# User-specific API methods
class UserAPIService(APIService):
    """User-specific API operations"""
    
    def login(self, email: str, password: str) -> APIResponse:
        """Authenticate user"""
        data = {"email": email, "password": password}
        endpoint = config.get('endpoints.user_login', 'User/Login')
        return self.post(endpoint, data=data)
    
    def get_user_from_token(self) -> APIResponse:
        """Get user information from current token"""
        endpoint = config.get('endpoints.user_from_token', 'User/GetUserFromToken')
        return self._make_request("GET", endpoint)
    
    def get_user_tenants(self) -> APIResponse:
        """Fetches the tenants for the current authenticated user."""
        endpoint = config.get('endpoints.user_tenants', 'Tenant/CurrentUserTenants')
        return self._make_request("GET", endpoint)
    
    def logout(self) -> APIResponse:
        """Logout user"""
        response = self.post("/User/Logout")
        if response.success:
            self.clear_auth_token()
        return response
    
    def get_user_profile(self, user_id: str) -> APIResponse:
        """Get user profile by ID"""
        return self.get(f"/User/{user_id}")
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> APIResponse:
        """Update user profile"""
        return self.put(f"/User/{user_id}", data=profile_data)

# Organization-specific API methods
class OrganizationAPIService(APIService):
    """Organization-specific API operations"""
    
    def get_organizations(self) -> APIResponse:
        """Get list of organizations"""
        return self.get("/Organizations")
    
    def get_organization(self, org_id: str) -> APIResponse:
        """Get organization details"""
        return self.get(f"/Organizations/{org_id}")
    
    def create_organization(self, org_data: Dict[str, Any]) -> APIResponse:
        """Create new organization"""
        return self.post("/Organizations", data=org_data)
    
    def update_organization(self, org_id: str, org_data: Dict[str, Any]) -> APIResponse:
        """Update organization"""
        return self.put(f"/Organizations/{org_id}", data=org_data)
    
    def delete_organization(self, org_id: str) -> APIResponse:
        """Delete organization"""
        return self.delete(f"/Organizations/{org_id}")

# Score-specific API methods
class ScoreAPIService(APIService):
    """Score and kill feed specific API operations"""
    
    def capture_kill_feed(self, feed_data: Dict[str, Any]) -> APIResponse:
        """Capture kill feed data"""
        return self.post("/Score/CaptureKillFeed", data=feed_data)
    
    def get_scores(self, match_id: Optional[str] = None) -> APIResponse:
        """Get scores for a match"""
        params = {"match_id": match_id} if match_id else None
        return self.get("/Score/GetScores", params=params)
    
    def calculate_score(self, score_data: Dict[str, Any]) -> APIResponse:
        """Calculate score from data"""
        return self.post("/Score/Calculate", data=score_data)
    
    def get_match_history(self, user_id: Optional[str] = None) -> APIResponse:
        """Get match history"""
        params = {"user_id": user_id} if user_id else None
        return self.get("/Score/MatchHistory", params=params)

# Global API service instance
api_service = APIService()
user_api = UserAPIService()
org_api = OrganizationAPIService()
score_api = ScoreAPIService()
