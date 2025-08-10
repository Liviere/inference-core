"""
Locust performance test suite for FastAPI Backend Template.

This module contains comprehensive performance tests covering:
- Health check endpoints
- Authentication flows
- Task monitoring endpoints
- Database health checks

Excludes LLM endpoints to avoid API costs.
"""

import json
import random
import string
import uuid
from typing import Dict, Any, Optional

from locust import HttpUser, task, between, events
from locust.env import Environment

# Handle imports for both direct execution and module import
try:
    from .config import get_profile, get_host_url
except ImportError:
    from config import get_profile, get_host_url


class BaseUser(HttpUser):
    """Base user class with common functionality."""
    
    abstract = True
    wait_time = between(1, 3)
    
    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.auth_headers: Optional[Dict[str, str]] = None
        self.refresh_token: Optional[str] = None
        self.user_credentials: Optional[Dict[str, str]] = None
    
    def generate_random_string(self, length: int = 8) -> str:
        """Generate a random string for unique usernames/emails."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    def check_response(self, response, endpoint_name: str):
        """Helper to check response and log errors appropriately."""
        if response.status_code >= 400:
            print(f"[{endpoint_name}] Error {response.status_code}: {response.text}")
        return response.status_code < 400


class HealthCheckUser(BaseUser):
    """
    User that performs health check operations.
    
    Tests all health-related endpoints including root, general health,
    database health, and ping endpoints.
    """
    
    weight = 3
    
    @task(5)
    def health_check(self):
        """Test the main health check endpoint."""
        with self.client.get(
            "/api/v1/health/",
            catch_response=True,
            name="Health Check"
        ) as response:
            if self.check_response(response, "Health Check"):
                try:
                    data = response.json()
                    if data.get("status") in ["healthy", "degraded"]:
                        response.success()
                    else:
                        response.failure(f"Unexpected health status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
    
    @task(3)
    def database_health(self):
        """Test database-specific health check."""
        with self.client.get(
            "/api/v1/health/database",
            catch_response=True,
            name="Database Health"
        ) as response:
            if self.check_response(response, "Database Health"):
                try:
                    data = response.json()
                    if data.get("status") in ["healthy", "unhealthy", "degraded"]:
                        response.success()
                    else:
                        response.failure(f"Unexpected DB status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
    
    @task(4)
    def ping(self):
        """Test simple ping endpoint."""
        with self.client.get(
            "/api/v1/health/ping",
            catch_response=True,
            name="Ping"
        ) as response:
            if self.check_response(response, "Ping"):
                try:
                    data = response.json()
                    if data.get("message") == "pong":
                        response.success()
                    else:
                        response.failure(f"Unexpected ping response: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
    
    @task(2)
    def root_endpoint(self):
        """Test root endpoint."""
        with self.client.get(
            "/",
            catch_response=True,
            name="Root"
        ) as response:
            if self.check_response(response, "Root"):
                try:
                    data = response.json()
                    if "message" in data and "version" in data:
                        response.success()
                    else:
                        response.failure("Missing expected fields in root response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
    
    @task(1)
    def docs_endpoint_if_available(self):
        """Test docs endpoint if available (development mode only)."""
        with self.client.get(
            "/docs",
            catch_response=True,
            name="Docs (Dev Only)"
        ) as response:
            # In production, docs might be disabled (404), which is acceptable
            if response.status_code == 404:
                response.success()  # Expected in production
            elif response.status_code == 200:
                if "swagger" in response.text.lower() or "openapi" in response.text.lower():
                    response.success()
                else:
                    response.failure("Docs endpoint doesn't contain expected content")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class AuthUserFlow(BaseUser):
    """
    User that performs complete authentication flows.
    
    Tests registration, login, profile operations, password changes,
    token refresh, and logout functionality.
    """
    
    weight = 8
    
    def on_start(self):
        """Initialize user with unique credentials."""
        random_suffix = self.generate_random_string()
        self.user_credentials = {
            "username": f"testuser_{random_suffix}",
            "email": f"test_{random_suffix}@example.com",
            "password": "SecurePass123!",
            "first_name": "Test",
            "last_name": "User"
        }
    
    @task(10)
    def complete_auth_flow(self):
        """Perform complete authentication flow."""
        # Step 1: Register
        if not self._register():
            return
        
        # Step 2: Login
        if not self._login():
            return
        
        # Step 3: Get profile
        if not self._get_profile():
            return
        
        # Step 4: Update profile
        if not self._update_profile():
            return
        
        # Step 5: Change password (optional, 30% chance)
        if random.random() < 0.3:
            if not self._change_password():
                return
        
        # Step 6: Refresh tokens (50% chance)
        if random.random() < 0.5 and self.refresh_token:
            if not self._refresh_tokens():
                return
        
        # Step 7: Logout
        self._logout()
    
    def _register(self) -> bool:
        """Register a new user."""
        with self.client.post(
            "/api/v1/auth/register",
            json=self.user_credentials,
            catch_response=True,
            name="Auth - Register"
        ) as response:
            if response.status_code == 201:
                return True
            elif response.status_code == 400:
                # User might already exist, try with new credentials
                random_suffix = self.generate_random_string()
                self.user_credentials["username"] = f"testuser_{random_suffix}"
                self.user_credentials["email"] = f"test_{random_suffix}@example.com"
                
                # Retry registration
                with self.client.post(
                    "/api/v1/auth/register",
                    json=self.user_credentials,
                    catch_response=True,
                    name="Auth - Register Retry"
                ) as retry_response:
                    return retry_response.status_code == 201
            else:
                response.failure(f"Registration failed: {response.status_code}")
                return False
    
    def _login(self) -> bool:
        """Login with user credentials."""
        login_data = {
            "username": self.user_credentials["username"],
            "password": self.user_credentials["password"]
        }
        
        with self.client.post(
            "/api/v1/auth/login",
            json=login_data,
            catch_response=True,
            name="Auth - Login"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    access_token = data.get("access_token")
                    self.refresh_token = data.get("refresh_token")
                    
                    if access_token:
                        self.auth_headers = {"Authorization": f"Bearer {access_token}"}
                        return True
                    else:
                        response.failure("No access token in login response")
                        return False
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in login response")
                    return False
            else:
                response.failure(f"Login failed: {response.status_code}")
                return False
    
    def _get_profile(self) -> bool:
        """Get current user profile."""
        if not self.auth_headers:
            return False
        
        with self.client.get(
            "/api/v1/auth/me",
            headers=self.auth_headers,
            catch_response=True,
            name="Auth - Get Profile"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "username" in data and "email" in data:
                        return True
                    else:
                        response.failure("Missing fields in profile response")
                        return False
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in profile response")
                    return False
            elif response.status_code == 401:
                # Token might be expired, try to refresh
                if self.refresh_token and self._refresh_tokens():
                    return self._get_profile()  # Retry
                else:
                    response.failure("Authentication failed")
                    return False
            else:
                response.failure(f"Get profile failed: {response.status_code}")
                return False
    
    def _update_profile(self) -> bool:
        """Update user profile."""
        if not self.auth_headers:
            return False
        
        update_data = {
            "first_name": f"Updated_{self.generate_random_string(4)}"
        }
        
        with self.client.put(
            "/api/v1/auth/me",
            headers=self.auth_headers,
            json=update_data,
            catch_response=True,
            name="Auth - Update Profile"
        ) as response:
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                # Token might be expired, try to refresh
                if self.refresh_token and self._refresh_tokens():
                    return self._update_profile()  # Retry
                else:
                    response.failure("Authentication failed")
                    return False
            else:
                response.failure(f"Update profile failed: {response.status_code}")
                return False
    
    def _change_password(self) -> bool:
        """Change user password."""
        if not self.auth_headers:
            return False
        
        new_password = f"NewSecurePass123!_{self.generate_random_string(4)}"
        password_data = {
            "current_password": self.user_credentials["password"],
            "new_password": new_password
        }
        
        with self.client.post(
            "/api/v1/auth/change-password",
            headers=self.auth_headers,
            json=password_data,
            catch_response=True,
            name="Auth - Change Password"
        ) as response:
            if response.status_code == 200:
                self.user_credentials["password"] = new_password
                return True
            elif response.status_code == 401:
                # Token might be expired, try to refresh
                if self.refresh_token and self._refresh_tokens():
                    return self._change_password()  # Retry
                else:
                    response.failure("Authentication failed")
                    return False
            else:
                response.failure(f"Change password failed: {response.status_code}")
                return False
    
    def _refresh_tokens(self) -> bool:
        """Refresh access tokens."""
        if not self.refresh_token:
            return False
        
        refresh_data = {"refresh_token": self.refresh_token}
        
        with self.client.post(
            "/api/v1/auth/refresh",
            json=refresh_data,
            catch_response=True,
            name="Auth - Refresh Tokens"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    access_token = data.get("access_token")
                    new_refresh_token = data.get("refresh_token")
                    
                    if access_token:
                        self.auth_headers = {"Authorization": f"Bearer {access_token}"}
                        if new_refresh_token:
                            self.refresh_token = new_refresh_token
                        return True
                    else:
                        response.failure("No access token in refresh response")
                        return False
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in refresh response")
                    return False
            else:
                response.failure(f"Token refresh failed: {response.status_code}")
                return False
    
    def _logout(self) -> bool:
        """Logout user."""
        if not self.refresh_token:
            return True  # Already logged out
        
        logout_data = {"refresh_token": self.refresh_token}
        
        with self.client.post(
            "/api/v1/auth/logout",
            json=logout_data,
            catch_response=True,
            name="Auth - Logout"
        ) as response:
            # Clear tokens regardless of response
            self.auth_headers = None
            self.refresh_token = None
            
            if response.status_code == 200:
                return True
            else:
                response.failure(f"Logout failed: {response.status_code}")
                return False
    
    @task(3)
    def forgot_password_request(self):
        """Test forgot password request (no email sending required)."""
        forgot_data = {
            "email": f"forgot_{self.generate_random_string()}@example.com"
        }
        
        with self.client.post(
            "/api/v1/auth/forgot-password",
            json=forgot_data,
            catch_response=True,
            name="Auth - Forgot Password"
        ) as response:
            # This endpoint should always return success for security reasons
            # (don't reveal if email exists)
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Forgot password failed: {response.status_code}")


class TasksMonitoringUser(BaseUser):
    """
    User that monitors task system health and statistics.
    
    Tests task health, worker stats, and active tasks endpoints.
    Requires Redis and Celery workers to be running for meaningful data.
    """
    
    weight = 4
    
    @task(5)
    def tasks_health(self):
        """Check task system health."""
        with self.client.get(
            "/api/v1/tasks/health",
            catch_response=True,
            name="Tasks - Health"
        ) as response:
            if self.check_response(response, "Tasks Health"):
                try:
                    data = response.json()
                    if "status" in data and "celery_available" in data:
                        response.success()
                    else:
                        response.failure("Missing expected fields in tasks health response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
    
    @task(3)
    def worker_stats(self):
        """Get worker statistics."""
        with self.client.get(
            "/api/v1/tasks/workers/stats",
            catch_response=True,
            name="Tasks - Worker Stats"
        ) as response:
            if self.check_response(response, "Worker Stats"):
                try:
                    data = response.json()
                    # Response should have stats, ping, and registered fields
                    if isinstance(data, dict):
                        response.success()
                    else:
                        response.failure("Worker stats response is not a dictionary")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
    
    @task(2)
    def active_tasks(self):
        """Get active tasks information."""
        with self.client.get(
            "/api/v1/tasks/active",
            catch_response=True,
            name="Tasks - Active"
        ) as response:
            if self.check_response(response, "Active Tasks"):
                try:
                    data = response.json()
                    # Response should have active, scheduled, and reserved fields
                    expected_fields = ["active", "scheduled", "reserved"]
                    if all(field in data for field in expected_fields):
                        response.success()
                    else:
                        response.failure(f"Missing expected fields: {expected_fields}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")


class DatabaseHealthUser(BaseUser):
    """
    User focused specifically on database health monitoring.
    
    Provides dedicated load on database health endpoints
    to test database performance under concurrent access.
    """
    
    weight = 3
    
    @task(1)
    def database_health_focused(self):
        """Focused database health check."""
        with self.client.get(
            "/api/v1/health/database",
            catch_response=True,
            name="DB - Health Focus"
        ) as response:
            if self.check_response(response, "DB Health Focus"):
                try:
                    data = response.json()
                    if "status" in data and "details" in data:
                        response.success()
                    else:
                        response.failure("Missing expected fields in DB health response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")


# Configure user weights from profile
def get_user_classes():
    """Get user classes with weights from current profile."""
    profile = get_profile()
    
    classes = []
    for class_name, weight in profile.weight_config.items():
        if class_name == "HealthCheckUser":
            HealthCheckUser.weight = weight
            classes.append(HealthCheckUser)
        elif class_name == "AuthUserFlow":
            AuthUserFlow.weight = weight
            classes.append(AuthUserFlow)
        elif class_name == "TasksMonitoringUser":
            TasksMonitoringUser.weight = weight
            classes.append(TasksMonitoringUser)
        elif class_name == "DatabaseHealthUser":
            DatabaseHealthUser.weight = weight
            classes.append(DatabaseHealthUser)
    
    return classes


# Event handlers for enhanced reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start with configuration."""
    profile = get_profile()
    print(f"\n🚀 Starting performance test with profile: {profile.name}")
    print(f"📋 Description: {profile.description}")
    print(f"👥 Users: {profile.users}")
    print(f"⚡ Spawn rate: {profile.spawn_rate}/s")
    print(f"⏱️  Duration: {profile.run_time}")
    print(f"🎯 Target: {get_host_url()}")
    print(f"📊 Weights: {profile.weight_config}")
    print("-" * 50)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion."""
    profile = get_profile()
    print(f"\n✅ Performance test completed: {profile.name}")
    print(f"📈 Report available at: reports/performance/{profile.name}_load_report.html")
    print("-" * 50)


# This allows Locust to automatically discover user classes
# when the file is run directly
if __name__ == "__main__":
    # Set user classes dynamically based on profile
    user_classes = get_user_classes()
    print(f"Loaded user classes: {[cls.__name__ for cls in user_classes]}")