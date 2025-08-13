"""
Helper utilities for Locust performance tests.

This module provides utilities for user generation, authentication helpers,
and common testing patterns.
"""

import json
import random
import string
import uuid
from typing import Dict, Any, Optional


class AuthenticationHelper:
    """Helper class for managing authentication in performance tests."""
    
    def __init__(self, client):
        self.client = client
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.auth_headers: Optional[Dict[str, str]] = None
    
    def generate_unique_credentials(self) -> Dict[str, str]:
        """Generate unique user credentials for registration."""
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return {
            "username": f"testuser_{random_suffix}",
            "email": f"test_{random_suffix}@example.com",
            "password": f"SecurePass123!_{random_suffix[:4]}",
            "first_name": "Test",
            "last_name": "User"
        }
    
    def register_and_login(self, max_retries: int = 3) -> bool:
        """
        Register a new user and login in one operation.
        
        Args:
            max_retries: Maximum number of registration attempts
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            credentials = self.generate_unique_credentials()
            
            # Try registration
            response = self.client.post("/api/v1/auth/register", json=credentials)
            
            if response.status_code == 201:
                # Registration successful, now login
                login_data = {
                    "username": credentials["username"],
                    "password": credentials["password"]
                }
                
                login_response = self.client.post("/api/v1/auth/login", json=login_data)
                
                if login_response.status_code == 200:
                    try:
                        data = login_response.json()
                        self.access_token = data.get("access_token")
                        self.refresh_token = data.get("refresh_token")
                        
                        if self.access_token:
                            self.auth_headers = {"Authorization": f"Bearer {self.access_token}"}
                            return True
                    except json.JSONDecodeError:
                        continue
            
            elif response.status_code != 400:  # 400 means user exists, retry
                break  # Other errors, don't retry
        
        return False
    
    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using the refresh token.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.refresh_token:
            return False
        
        refresh_data = {"refresh_token": self.refresh_token}
        response = self.client.post("/api/v1/auth/refresh", json=refresh_data)
        
        if response.status_code == 200:
            try:
                data = response.json()
                new_access_token = data.get("access_token")
                new_refresh_token = data.get("refresh_token")
                
                if new_access_token:
                    self.access_token = new_access_token
                    self.auth_headers = {"Authorization": f"Bearer {new_access_token}"}
                    
                    if new_refresh_token:
                        self.refresh_token = new_refresh_token
                    
                    return True
            except json.JSONDecodeError:
                pass
        
        return False
    
    def logout(self) -> bool:
        """
        Logout the current user.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.refresh_token:
            return True  # Already logged out
        
        logout_data = {"refresh_token": self.refresh_token}
        response = self.client.post("/api/v1/auth/logout", json=logout_data)
        
        # Clear tokens regardless of response
        self.access_token = None
        self.refresh_token = None
        self.auth_headers = None
        
        return response.status_code == 200
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        return self.auth_headers is not None
    
    def handle_auth_error(self, response) -> bool:
        """
        Handle authentication errors by trying to refresh token.
        
        Args:
            response: HTTP response that might have auth error
            
        Returns:
            True if auth was refreshed successfully, False otherwise
        """
        if response.status_code == 401 and self.refresh_token:
            return self.refresh_access_token()
        return False


class ResponseValidator:
    """Helper class for validating API responses in performance tests."""
    
    @staticmethod
    def validate_health_response(data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate health check response format.
        
        Args:
            data: JSON response data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Response is not a JSON object"
        
        required_fields = ["status", "timestamp", "components"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        if data["status"] not in valid_statuses:
            return False, f"Invalid status: {data['status']}"
        
        return True, ""
    
    @staticmethod
    def validate_auth_response(data: Dict[str, Any], response_type: str) -> tuple[bool, str]:
        """
        Validate authentication response format.
        
        Args:
            data: JSON response data
            response_type: Type of auth response ('login', 'profile', 'token')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Response is not a JSON object"
        
        if response_type == "login":
            required_fields = ["access_token", "refresh_token", "token_type"]
        elif response_type == "profile":
            required_fields = ["username", "email", "is_active"]
        elif response_type == "token":
            required_fields = ["access_token", "refresh_token", "token_type"]
        else:
            return False, f"Unknown response type: {response_type}"
        
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        return True, ""
    
    @staticmethod
    def validate_tasks_response(data: Dict[str, Any], response_type: str) -> tuple[bool, str]:
        """
        Validate tasks endpoint response format.
        
        Args:
            data: JSON response data
            response_type: Type of tasks response ('health', 'stats', 'active')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Response is not a JSON object"
        
        if response_type == "health":
            required_fields = ["status", "celery_available"]
        elif response_type == "stats":
            # Worker stats can vary, but should be a dict
            return True, ""
        elif response_type == "active":
            required_fields = ["active", "scheduled", "reserved"]
        else:
            return False, f"Unknown response type: {response_type}"
        
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        return True, ""


class LoadTestMetrics:
    """Helper class for collecting and analyzing load test metrics."""
    
    def __init__(self):
        self.request_counts = {}
        self.error_counts = {}
        self.response_times = {}
    
    def record_request(self, endpoint: str, response_time: float, success: bool):
        """Record a request for metrics analysis."""
        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = 0
            self.error_counts[endpoint] = 0
            self.response_times[endpoint] = []
        
        self.request_counts[endpoint] += 1
        if not success:
            self.error_counts[endpoint] += 1
        
        self.response_times[endpoint].append(response_time)
    
    def get_error_rate(self, endpoint: str) -> float:
        """Get error rate for a specific endpoint."""
        if endpoint not in self.request_counts:
            return 0.0
        
        total_requests = self.request_counts[endpoint]
        if total_requests == 0:
            return 0.0
        
        return self.error_counts[endpoint] / total_requests
    
    def get_average_response_time(self, endpoint: str) -> float:
        """Get average response time for a specific endpoint."""
        if endpoint not in self.response_times or not self.response_times[endpoint]:
            return 0.0
        
        return sum(self.response_times[endpoint]) / len(self.response_times[endpoint])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {}
        
        for endpoint in self.request_counts:
            summary[endpoint] = {
                "requests": self.request_counts[endpoint],
                "errors": self.error_counts[endpoint],
                "error_rate": self.get_error_rate(endpoint),
                "avg_response_time": self.get_average_response_time(endpoint)
            }
        
        return summary


def generate_test_data() -> Dict[str, Any]:
    """Generate realistic test data for various endpoints."""
    return {
        "user_update": {
            "first_name": f"Updated_{random.randint(1000, 9999)}",
            "last_name": f"User_{random.randint(1000, 9999)}"
        },
        "password_change": {
            "current_password": "OldPassword123!",
            "new_password": f"NewPassword123!_{random.randint(1000, 9999)}"
        },
        "email_request": {
            "email": f"test_{random.randint(1000, 9999)}@example.com"
        }
    }


def is_development_mode(client) -> bool:
    """
    Check if the API is running in development mode.
    
    Args:
        client: HTTP client to test with
        
    Returns:
        True if development mode, False otherwise
    """
    try:
        response = client.get("/")
        if response.status_code == 200:
            data = response.json()
            return data.get("debug", False) or data.get("docs") != "disabled"
    except:
        pass
    
    return False