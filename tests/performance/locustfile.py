"""
Locust performance test suite for FastAPI Backend Template.

This module contains comprehensive performance tests covering:
- Health check endpoints
- Authentication flows
- Task monitoring endpoints
- Database health checks
- No-cost LLM mock workflows with agent instances, embeddings, and vector search

LLM-heavy workflows are available only through the ``llm_mock`` profile and
must be run against an environment configured with LLM emulation and fake or
local no-cost embeddings.
"""

import json
import os
import random
import string
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

# Handle imports for both direct execution and module import
from config import (
    format_llm_mock_embedding_backends,
    get_host_url,
    get_profile,
    is_llm_mock_embedding_backend,
)
from jose import jwt
from locust import HttpUser, between, events, task
from locust.env import Environment

AUTH_DEPENDENT_USER_CLASSES = frozenset({"AuthUserFlow", "LLMMockWorkspaceUser"})


class BaseUser(HttpUser):
    """Base user class with common functionality."""

    abstract = True
    wait_time = between(1, 3)
    access_token_refresh_skew_seconds = 30
    refresh_cookie_name = os.getenv("AUTH_REFRESH_COOKIE_NAME", "refresh_token")

    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.access_token: Optional[str] = None
        self.access_token_exp: Optional[int] = None
        self.auth_headers: Optional[Dict[str, str]] = None
        self.refresh_token: Optional[str] = None
        self.refresh_token_present: bool = False
        self.is_authenticated: bool = False
        self.user_credentials: Optional[Dict[str, str]] = None
        self.registered: bool = False

    def generate_random_string(self, length: int = 8) -> str:
        """Generate a random string for unique usernames/emails."""
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def check_response(self, response, endpoint_name: str):
        """Helper to check response and log errors appropriately."""
        if response.status_code >= 400:
            print(f"[{endpoint_name}] Error {response.status_code}: {response.text}")
        return response.status_code < 400

    def reset_auth_state(self, clear_cookies: bool = False) -> None:
        """Clear the simulated user's auth state.

        WHY: Login, refresh, and logout all need the same cleanup semantics so
        scenarios can recover from expired or revoked sessions predictably.
        """
        self.access_token = None
        self.access_token_exp = None
        self.auth_headers = None
        self.is_authenticated = False
        self.refresh_token = None
        self.refresh_token_present = False
        if clear_cookies:
            self._clear_refresh_cookie()

    def _clear_refresh_cookie(self) -> None:
        """Remove the refresh cookie from the Locust client's cookie jar.

        WHY: The cookie jar is the closest analogue to a browser session, so a
        revoked session should be removed there instead of only from Python state.
        """
        cookies_to_clear = [
            cookie
            for cookie in self.client.cookies
            if cookie.name == self.refresh_cookie_name
        ]
        for cookie in cookies_to_clear:
            self.client.cookies.clear(
                domain=cookie.domain,
                path=cookie.path,
                name=cookie.name,
            )

    def _get_refresh_cookie_value(self) -> Optional[str]:
        """Read the current refresh cookie value from the client's cookie jar.

        WHY: The server writes refresh tokens to an HttpOnly cookie, so Locust's
        session cookie jar should be treated as the source of truth.
        """
        for cookie in self.client.cookies:
            if cookie.name == self.refresh_cookie_name:
                return str(cookie.value)
        return None

    def _sync_refresh_token_state(self) -> None:
        """Mirror the cookie jar refresh token into lightweight user state.

        WHY: Scenario code sometimes needs a cheap boolean about refresh-session
        availability without duplicating cookie-jar parsing in every request path.
        """
        refresh_cookie = self._get_refresh_cookie_value()
        self.refresh_token = refresh_cookie
        self.refresh_token_present = bool(refresh_cookie)

    def _extract_access_token_exp(self, access_token: str) -> Optional[int]:
        """Decode the access token expiry without verification.

        WHY: Locust only needs the ``exp`` claim to refresh proactively before a
        protected request; verifying the JWT client-side adds no value here.
        """
        try:
            claims = jwt.get_unverified_claims(access_token)
        except Exception:
            return None
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)
        if isinstance(exp, str) and exp.isdigit():
            return int(exp)
        return None

    def _store_access_token(self, access_token: str) -> None:
        """Store an access token and derived session metadata.

        WHY: Every successful login or refresh should update the same fields so
        auth-dependent scenarios can rely on one shared contract.
        """
        self.access_token = access_token
        self.access_token_exp = self._extract_access_token_exp(access_token)
        self.auth_headers = {"Authorization": f"Bearer {access_token}"}
        self.is_authenticated = True
        self._sync_refresh_token_state()

    def _access_token_is_fresh(self) -> bool:
        """Return whether the cached access token is still usable.

        WHY: Long-lived Locust users need a cheap TTL check so they can refresh
        before the next authenticated request instead of failing with avoidable 401s.
        """
        if not self.access_token or not self.auth_headers:
            return False
        if self.access_token_exp is None:
            return True
        return int(time.time()) < (
            self.access_token_exp - self.access_token_refresh_skew_seconds
        )

    def has_refresh_session(self) -> bool:
        """Return whether the client still carries a refresh session cookie.

        WHY: Refreshability depends on the cookie jar state, not only on whether
        a previous login stored a string in a Python attribute.
        """
        self._sync_refresh_token_state()
        return self.refresh_token_present

    def refresh_access_token(self) -> bool:
        """Rotate the current session into a fresh access token.

        WHY: Authenticated scenarios should survive short access-token TTLs by
        exercising the same refresh path that real browser clients use.
        """
        if not self.has_refresh_session():
            return False

        with self.client.post(
            "/api/v1/auth/refresh",
            catch_response=True,
            name="Auth - Refresh Tokens",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Token refresh failed: {response.status_code}")
                self.reset_auth_state(clear_cookies=True)
                return False
            try:
                data = response.json()
            except json.JSONDecodeError:
                response.failure("Invalid JSON in refresh response")
                self.reset_auth_state(clear_cookies=True)
                return False

            access_token = data.get("access_token")
            if not access_token:
                response.failure("No access token in refresh response")
                self.reset_auth_state(clear_cookies=True)
                return False

            self._store_access_token(access_token)
            response.success()
            return True

    def ensure_authenticated(
        self,
        *,
        force_refresh: bool = False,
        force_login: bool = False,
    ) -> bool:
        """Guarantee that the user has a valid access token before a request.

        WHY: Scenario code should ask for an authenticated session, not hand-roll
        token expiry checks and recovery logic for every endpoint.
        """
        if force_login:
            self.reset_auth_state(clear_cookies=True)
            return self._login_current_user()
        if not force_refresh and self._access_token_is_fresh():
            return True
        if self.refresh_access_token():
            return True
        return self._login_current_user()

    def authenticate_unique_user(self, name_prefix: str = "locust") -> bool:
        """Register and login a unique user for stateful workload simulation.

        WHY: Agent instances, preferences, and vector collections are user-owned
        resources. A realistic Locust user needs stable credentials for the
        lifetime of the simulated browser session.
        """
        random_suffix = self.generate_random_string(12)
        self.user_credentials = {
            "username": f"{name_prefix}{random_suffix}",
            "email": f"{name_prefix}_{random_suffix}@example.com",
            "password": "SecurePass123!",
            "first_name": "Load",
            "last_name": "Tester",
        }

        if not self._register_current_user():
            return False
        self.registered = self._login_current_user()
        return self.registered

    def _register_current_user(self) -> bool:
        """Create the current credentials once for this Locust user.

        WHY: Keeping registration in the base class lets realistic user flows
        share authentication setup without inheriting unrelated task weights.
        """
        if not self.user_credentials:
            return False

        with self.client.post(
            "/api/v1/auth/register",
            json=self.user_credentials,
            catch_response=True,
            name="Auth - Register",
        ) as response:
            if response.status_code == 201:
                response.success()
                return True
            response.failure(f"Registration failed: {response.status_code}")
            return False

    def _login_current_user(self) -> bool:
        """Login the current credentials and store JWT/cookie state.

        WHY: Most realistic workloads need an Authorization header while refresh
        and logout depend on the refresh cookie returned by the API.
        """
        if not self.user_credentials:
            return False

        login_data = {
            "username": self.user_credentials["username"],
            "password": self.user_credentials["password"],
        }

        with self.client.post(
            "/api/v1/auth/login",
            json=login_data,
            catch_response=True,
            name="Auth - Login",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Login failed: {response.status_code}")
                self.reset_auth_state()
                return False
            try:
                data = response.json()
            except json.JSONDecodeError:
                response.failure("Invalid JSON in login response")
                self.reset_auth_state()
                return False

            access_token = data.get("access_token")
            if not access_token:
                response.failure("No access token in login response")
                self.reset_auth_state()
                return False

            self._store_access_token(access_token)
            response.success()
            return True

    def logout_current_user(self) -> bool:
        """Logout the simulated user if a refresh token is available.

        WHY: Long-running Locust runs should exercise session cleanup and avoid
        leaving every simulated user with an active refresh session.
        """
        if not self.has_refresh_session():
            self.reset_auth_state(clear_cookies=True)
            return True

        with self.client.post(
            "/api/v1/auth/logout",
            catch_response=True,
            name="Auth - Logout",
        ) as response:
            self.reset_auth_state(clear_cookies=True)
            if response.status_code == 200:
                response.success()
                return True
            response.failure(f"Logout failed: {response.status_code}")
            return False

    def authenticated_headers(self) -> Dict[str, str]:
        """Return Authorization headers, re-authenticating if needed.

        WHY: Locust users live longer than a single request and can outlive short
        access-token TTLs in endurance profiles.
        """
        self.ensure_authenticated()
        return self.auth_headers or {}


class HealthCheckUser(BaseUser):
    """
    User that performs health check operations.

    Tests all health-related endpoints including root, general health,
    database health, and ping endpoints.
    """

    weight = 30

    @task(2)
    def database_health(self):
        """Test database-specific health check."""
        with self.client.get(
            "/api/v1/health/database", catch_response=True, name="Database Health"
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

    @task(5)
    def ping(self):
        """Test simple ping endpoint."""
        with self.client.get(
            "/api/v1/health/ping", catch_response=True, name="Ping"
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

    @task(5)
    def root_endpoint(self):
        """Test root endpoint."""
        with self.client.get("/", catch_response=True, name="Root") as response:
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
            "/docs", catch_response=True, name="Docs (Dev Only)"
        ) as response:
            # In production, docs might be disabled (404), which is acceptable
            if response.status_code == 404:
                response.success()  # Expected in production
            elif response.status_code == 200:
                if (
                    "swagger" in response.text.lower()
                    or "openapi" in response.text.lower()
                ):
                    response.success()
                else:
                    response.failure("Docs endpoint doesn't contain expected content")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")


class HealthCheckFullUser(BaseUser):
    """
    User that performs full health check operation.

    Tests main health endpoint which is a combination of all health checks.
    This is pretty heavy and should be used sparingly.
    """

    weight = 1

    @task(1)
    def health_check(self):
        """Test the main health check endpoint."""
        with self.client.get(
            "/api/v1/health/", catch_response=True, name="Health Check"
        ) as response:
            if self.check_response(response, "Health Check"):
                try:
                    data = response.json()
                    if data.get("status") in ["healthy", "degraded"]:
                        response.success()
                    else:
                        response.failure(
                            f"Unexpected health status: {data.get('status')}"
                        )
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")


class AuthUserFlow(BaseUser):
    """
    User that performs complete authentication flows.

    Tests registration, login, profile operations, password changes,
    token refresh, and logout functionality.
    """

    weight = 8

    def on_start(self):
        """Initialize user with unique credentials and authenticate once."""
        self.registered = self.authenticate_unique_user("testuser")

    def on_stop(self):
        """Logout at the end of the user's life."""
        self.logout_current_user()

    @task(6)
    def authenticated_profile_flow(self):
        """Exercise normal authenticated profile operations.

        WHY: Most real sessions spend their time reading or mutating profile data
        while relying on the shared session layer to keep auth state valid.
        """
        if not self._ensure_registered_session():
            return

        if not self._get_profile():
            return
        if not self._update_profile():
            return

    @task(2)
    def refresh_session_flow(self):
        """Exercise explicit refresh-token rotation during a live session.

        WHY: Short-lived access tokens need dedicated load coverage; otherwise a
        green auth profile can still miss broken refresh semantics.
        """
        if not self._ensure_registered_session():
            return
        self.refresh_access_token()

    @task(1)
    def password_rotation_flow(self):
        """Change the password within an authenticated session.

        WHY: Password change is a stateful write that validates session handling
        under a more realistic authenticated workflow than profile reads alone.
        """
        if not self._ensure_registered_session():
            return
        self._change_password()

    @task(1)
    def logout_and_relogin_flow(self):
        """Exercise session teardown and recovery within a user's lifetime.

        WHY: Logging out only in ``on_stop`` misses the production path where a
        user explicitly ends a session and starts another one later.
        """
        if not self._ensure_registered_session():
            return
        if not self.logout_current_user():
            return
        self.ensure_authenticated(force_login=True)

    @task(1)
    def revoked_refresh_token_is_rejected(self):
        """Verify that a logged-out refresh token cannot be reused.

        WHY: The backend revokes refresh sessions in Redis, so Locust should
        assert that the old cookie no longer works after logout.
        """
        if not self._ensure_registered_session():
            return

        self._sync_refresh_token_state()
        stale_refresh_token = self.refresh_token
        if not stale_refresh_token:
            return

        if not self.logout_current_user():
            return

        with self.client.post(
            "/api/v1/auth/refresh",
            cookies={self.refresh_cookie_name: stale_refresh_token},
            catch_response=True,
            name="Auth - Refresh Revoked",
        ) as response:
            if response.status_code == 401:
                response.success()
            else:
                response.failure(
                    "Expected revoked refresh token to be rejected, got "
                    f"{response.status_code}"
                )

        self.ensure_authenticated(force_login=True)

    def _ensure_registered_session(self) -> bool:
        """Ensure this Locust user has a recoverable authenticated session.

        WHY: Auth tasks need one shared entry point that can rebuild the session
        after logout, refresh revocation, or access-token expiry.
        """
        if not self.user_credentials:
            return False
        if not self.registered:
            self.registered = self._register()
            if not self.registered:
                return False
        return self.ensure_authenticated()

    def _register(self) -> bool:
        """Register a new user."""
        with self.client.post(
            "/api/v1/auth/register",
            json=self.user_credentials,
            catch_response=True,
            name="Auth - Register",
        ) as response:
            if response.status_code == 201:
                return True
            elif response.status_code == 400:
                # User might already exist, try with new credentials
                random_suffix = self.generate_random_string()
                self.user_credentials["username"] = f"testuser{random_suffix}"
                self.user_credentials["email"] = f"test{random_suffix}@example.com"

                # Retry registration
                with self.client.post(
                    "/api/v1/auth/register",
                    json=self.user_credentials,
                    catch_response=True,
                    name="Auth - Register Retry",
                ) as retry_response:
                    if retry_response.status_code == 201:
                        retry_response.success()
                        return True
                    retry_response.failure(
                        f"Registration retry failed: {retry_response.status_code}"
                    )
                    return False
            else:
                response.failure(f"Registration failed: {response.status_code}")
                return False

    def _login(self) -> bool:
        """Login with user credentials."""
        return self._login_current_user()

    def _get_profile(self) -> bool:
        """Get current user profile."""
        headers = self.authenticated_headers()
        if not headers:
            return False

        with self.client.get(
            "/api/v1/auth/me",
            headers=headers,
            catch_response=True,
            name="Auth - Get Profile",
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
                self.reset_auth_state()
                if self.ensure_authenticated(force_refresh=True):
                    return self._get_profile()  # Retry
                response.failure("Authentication failed")
                return False
            else:
                response.failure(f"Get profile failed: {response.status_code}")
                return False

    def _update_profile(self) -> bool:
        """Update user profile."""
        headers = self.authenticated_headers()
        if not headers:
            return False

        update_data = {"first_name": f"Updated_{self.generate_random_string(4)}"}

        with self.client.put(
            "/api/v1/auth/me",
            headers=headers,
            json=update_data,
            catch_response=True,
            name="Auth - Update Profile",
        ) as response:
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                self.reset_auth_state()
                if self.ensure_authenticated(force_refresh=True):
                    return self._update_profile()  # Retry
                response.failure("Authentication failed")
                return False
            else:
                response.failure(f"Update profile failed: {response.status_code}")
                return False

    def _change_password(self) -> bool:
        """Change user password."""
        headers = self.authenticated_headers()
        if not headers:
            return False

        new_password = f"NewSecurePass123!_{self.generate_random_string(4)}"
        password_data = {
            "current_password": self.user_credentials["password"],
            "new_password": new_password,
        }

        with self.client.post(
            "/api/v1/auth/change-password",
            headers=headers,
            json=password_data,
            catch_response=True,
            name="Auth - Change Password",
        ) as response:
            if response.status_code == 200:
                self.user_credentials["password"] = new_password
                return True
            elif response.status_code == 401:
                self.reset_auth_state()
                if self.ensure_authenticated(force_refresh=True):
                    return self._change_password()  # Retry
                response.failure("Authentication failed")
                return False
            else:
                response.failure(f"Change password failed: {response.status_code}")
                return False

    def _refresh_tokens(self) -> bool:
        """Refresh access tokens."""
        return self.refresh_access_token()

    def _logout(self) -> bool:
        """Logout user."""
        return self.logout_current_user()

    @task(1)
    def forgot_password_request(self):
        """Test forgot password request (no email sending required)."""
        forgot_data = {"email": f"forgot_{self.generate_random_string()}@example.com"}

        with self.client.post(
            "/api/v1/auth/forgot-password",
            json=forgot_data,
            catch_response=True,
            name="Auth - Forgot Password",
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

    weight = 1

    @task(5)
    def tasks_health(self):
        """Check task system health."""
        with self.client.get(
            "/api/v1/tasks/health", catch_response=True, name="Tasks - Health"
        ) as response:
            if self.check_response(response, "Tasks Health"):
                try:
                    data = response.json()
                    if "status" in data and "celery_available" in data:
                        response.success()
                    else:
                        response.failure(
                            "Missing expected fields in tasks health response"
                        )
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")

    @task(1)
    def worker_stats(self):
        """Get worker statistics."""
        with self.client.get(
            "/api/v1/tasks/workers/stats",
            catch_response=True,
            name="Tasks - Worker Stats",
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

    @task(1)
    def active_tasks(self):
        """Get active tasks information."""
        with self.client.get(
            "/api/v1/tasks/active", catch_response=True, name="Tasks - Active"
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


class LLMMockWorkspaceUser(BaseUser):
    """Realistic no-cost user working with agents and a small knowledge base."""

    weight = 0
    wait_time = between(1, 4)

    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.template_name: Optional[str] = None
        self.template_primary_model: Optional[str] = None
        self.agent_instance_id: Optional[str] = None
        self.agent_instance_name: Optional[str] = None
        self.vector_collection: Optional[str] = None
        self.vector_available: bool = False

    def on_start(self):
        """Create the user session and seed resources used by later tasks.

        WHY: Real user traffic usually reuses an agent and a knowledge base
        across many interactions instead of recreating them on every request.
        """
        if not self.authenticate_unique_user("llmuser"):
            return

        suffix = self.generate_random_string(10)
        self.vector_collection = f"locust_{suffix}"
        self.agent_instance_name = f"locust-agent-{suffix}"

        self._load_agent_template()
        self._create_agent_instance()
        self._seed_vector_collection()

    def on_stop(self):
        """Clean up resources owned by this simulated user where practical.

        WHY: Performance runs create many short-lived users; soft-deleting agent
        instances and logging out keeps the test database closer to normal usage.
        """
        if self.agent_instance_id:
            self.client.delete(
                f"/api/v1/agent-instances/{self.agent_instance_id}",
                headers=self.authenticated_headers(),
                name="Agent Instances - Delete",
            )
        self.logout_current_user()

    @task(6)
    def chat_with_agent(self):
        """Run the user's configured agent instance with varied inputs."""
        if not self._ensure_agent_instance():
            return

        prompt = random.choice(
            [
                "Summarize what changed in my project today.",
                "Draft a short implementation checklist for a backend feature.",
                "Explain the tradeoffs of using fake embeddings in tests.",
                "Suggest the next validation step for an API change.",
            ]
        )
        payload = {
            "user_input": prompt,
            "system_prompt": "You are a concise backend engineering assistant.",
        }

        for attempt in range(2):
            with self.client.post(
                f"/api/v1/agent-instances/{self.agent_instance_id}/run",
                headers=self.authenticated_headers(),
                json=payload,
                catch_response=True,
                name="Agent Instances - Run Emulated",
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    response.success()
                    self.reset_auth_state()
                    if self.ensure_authenticated(force_refresh=True):
                        continue
                    response.failure("Agent run authentication recovery failed")
                    return
                if response.status_code != 200:
                    response.failure(f"Agent run failed: {response.status_code}")
                    return
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in agent run response")
                    return
                if "result" in data and "model_name" in data:
                    response.success()
                else:
                    response.failure("Missing agent run fields")
                return

    @task(4)
    def browse_agent_workspace(self):
        """Read templates, own instances, and current instance metadata."""
        headers = self.authenticated_headers()
        self._get_json(
            "GET",
            "/api/v1/agent-instances/templates",
            name="Agent Instances - Templates",
            headers=headers,
            expected_fields=("templates", "available_models"),
        )
        self._get_json(
            "GET",
            "/api/v1/agent-instances",
            name="Agent Instances - List",
            headers=headers,
            expected_fields=("instances", "total"),
        )
        if self.agent_instance_id:
            self._get_json(
                "GET",
                f"/api/v1/agent-instances/{self.agent_instance_id}",
                name="Agent Instances - Detail",
                headers=headers,
                expected_fields=("id", "instance_name", "base_agent_name"),
            )

    @task(3)
    def query_knowledge_base(self):
        """Search and list the user's test knowledge base collection."""
        if not self.vector_available or not self.vector_collection:
            return

        query_payload = {
            "query": random.choice(
                [
                    "agent emulation",
                    "performance test checklist",
                    "vector search smoke test",
                    "security scan preparation",
                ]
            ),
            "k": 3,
            "collection": self.vector_collection,
        }
        self._post_json(
            "/api/v1/vector/query",
            query_payload,
            name="Vector - Query",
            headers=self.authenticated_headers(),
            expected_fields=("documents", "count", "collection"),
        )

        list_payload = {
            "collection": self.vector_collection,
            "limit": 10,
            "offset": 0,
            "filters": {"source": "locust"},
        }
        self._post_json(
            "/api/v1/vector/list",
            list_payload,
            name="Vector - List",
            headers=self.authenticated_headers(),
            expected_fields=("documents", "count", "collection"),
        )

    @task(2)
    def add_small_knowledge_update(self):
        """Ingest a small synchronous document batch into the user's collection."""
        if not self.vector_available or not self.vector_collection:
            return

        marker = self.generate_random_string(6)
        payload = {
            "texts": [
                f"Locust knowledge update {marker}: agent runs are emulated.",
                f"Locust knowledge update {marker}: embeddings are deterministic.",
            ],
            "metadatas": [
                {"source": "locust", "marker": marker, "kind": "agent"},
                {"source": "locust", "marker": marker, "kind": "embedding"},
            ],
            "collection": self.vector_collection,
            "async_mode": False,
        }
        self._post_json(
            "/api/v1/vector/ingest",
            payload,
            name="Vector - Ingest Sync",
            headers=self.authenticated_headers(),
            expected_fields=("task_id", "collection", "estimated_count"),
        )

    @task(2)
    def generate_embeddings(self):
        """Exercise the public embedding endpoint under a safe test backend."""
        payload = {
            "texts": [
                "User asks the agent to summarize a document.",
                "User searches their small project knowledge base.",
                "User opens an agent configuration panel.",
            ]
        }
        for attempt in range(2):
            with self.client.post(
                "/api/v1/embeddings/generate",
                headers=self.authenticated_headers(),
                json=payload,
                catch_response=True,
                name="Embeddings - Generate",
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    response.success()
                    self.reset_auth_state()
                    if self.ensure_authenticated(force_refresh=True):
                        continue
                    response.failure("Embedding authentication recovery failed")
                    return
                if response.status_code != 200:
                    response.failure(
                        f"Embedding generation failed: {response.status_code}"
                    )
                    return
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in embedding response")
                    return
                backend = str(data.get("backend", "")).lower()
                if is_llm_mock_embedding_backend(backend) and data.get("count") == len(
                    payload["texts"]
                ):
                    response.success()
                else:
                    response.failure(
                        "Embedding backend is not one of "
                        f"{format_llm_mock_embedding_backends()} or count does not match. "
                        "Run llm_mock only with a safe no-cost embedding backend."
                    )
                return

    @task(1)
    def update_agent_settings(self):
        """Patch non-costly metadata on the user's agent instance."""
        if not self.agent_instance_id:
            return

        payload = {
            "display_name": f"Locust Agent {self.generate_random_string(4)}",
            "system_prompt_append": "Keep answers concise and test-friendly.",
        }
        self._patch_json(
            f"/api/v1/agent-instances/{self.agent_instance_id}",
            payload,
            name="Agent Instances - Patch",
            headers=self.authenticated_headers(),
            expected_fields=("id", "display_name"),
        )

    @task(1)
    def vector_collection_stats(self):
        """Read collection stats to mimic dashboard refresh behaviour."""
        if not self.vector_available or not self.vector_collection:
            return
        self._get_json(
            "GET",
            f"/api/v1/vector/collections/{self.vector_collection}/stats",
            name="Vector - Collection Stats",
            headers=self.authenticated_headers(),
            expected_fields=("name", "count", "dimension"),
        )

    def _load_agent_template(self) -> bool:
        data = self._get_json(
            "GET",
            "/api/v1/agent-instances/templates",
            name="Agent Instances - Templates",
            headers=self.authenticated_headers(),
            expected_fields=("templates", "available_models"),
        )
        if not data or not data.get("templates"):
            return False
        template = data["templates"][0]
        self.template_name = template.get("agent_name")
        self.template_primary_model = template.get("primary_model")
        return bool(self.template_name)

    def _create_agent_instance(self) -> bool:
        if not self.template_name or not self.agent_instance_name:
            return False

        payload = {
            "instance_name": self.agent_instance_name,
            "display_name": "Locust Workspace Agent",
            "base_agent_name": self.template_name,
            "description": "Agent instance created by the llm_mock Locust profile.",
            "system_prompt_append": "Prefer short answers suitable for load tests.",
        }
        data = self._post_json(
            "/api/v1/agent-instances",
            payload,
            name="Agent Instances - Create",
            headers=self.authenticated_headers(),
            expected_fields=("id", "instance_name", "base_agent_name"),
            success_statuses={201},
        )
        if not data:
            return False
        self.agent_instance_id = data.get("id")
        return bool(self.agent_instance_id)

    def _ensure_agent_instance(self) -> bool:
        if self.agent_instance_id:
            return True
        if not self.template_name and not self._load_agent_template():
            return False
        return self._create_agent_instance()

    def _seed_vector_collection(self) -> bool:
        if not self.vector_collection:
            return False

        health = self._get_json(
            "GET",
            "/api/v1/vector/health",
            name="Vector - Health",
            headers=self.authenticated_headers(),
            expected_fields=("status",),
            success_statuses={200, 503},
        )
        if not health or health.get("status") not in {"healthy", "ok"}:
            self.vector_available = False
            return False

        payload = {
            "texts": [
                "The backend uses emulated chat models for no-cost tests.",
                "Fake embeddings make vector search deterministic during load tests.",
                "Agent instances let users customize prompts and model settings.",
                "Locust should exercise realistic read and write API workflows.",
            ],
            "metadatas": [
                {"source": "locust", "kind": "llm"},
                {"source": "locust", "kind": "embedding"},
                {"source": "locust", "kind": "agent"},
                {"source": "locust", "kind": "performance"},
            ],
            "collection": self.vector_collection,
            "async_mode": False,
        }
        data = self._post_json(
            "/api/v1/vector/ingest",
            payload,
            name="Vector - Seed Sync",
            headers=self.authenticated_headers(),
            expected_fields=("task_id", "collection", "estimated_count"),
        )
        self.vector_available = bool(data)
        return self.vector_available

    def _get_json(
        self,
        method: str,
        path: str,
        *,
        name: str,
        headers: Optional[Dict[str, str]] = None,
        expected_fields: tuple[str, ...] = (),
        success_statuses: set[int] | None = None,
    ) -> Optional[Dict[str, Any]]:
        success_statuses = success_statuses or {200}
        for attempt in range(2):
            with self.client.request(
                method,
                path,
                headers=headers,
                catch_response=True,
                name=name,
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    response.success()
                    self.reset_auth_state()
                    if self.ensure_authenticated(force_refresh=True):
                        headers = self.authenticated_headers()
                        continue
                    response.failure(f"{name} authentication recovery failed")
                    return None
                return self._parse_json_response(
                    response, name, expected_fields, success_statuses
                )
        return None

    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        name: str,
        headers: Optional[Dict[str, str]] = None,
        expected_fields: tuple[str, ...] = (),
        success_statuses: set[int] | None = None,
    ) -> Optional[Dict[str, Any]]:
        success_statuses = success_statuses or {200}
        for attempt in range(2):
            with self.client.post(
                path,
                headers=headers,
                json=payload,
                catch_response=True,
                name=name,
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    response.success()
                    self.reset_auth_state()
                    if self.ensure_authenticated(force_refresh=True):
                        headers = self.authenticated_headers()
                        continue
                    response.failure(f"{name} authentication recovery failed")
                    return None
                return self._parse_json_response(
                    response, name, expected_fields, success_statuses
                )
        return None

    def _patch_json(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        name: str,
        headers: Optional[Dict[str, str]] = None,
        expected_fields: tuple[str, ...] = (),
    ) -> Optional[Dict[str, Any]]:
        for attempt in range(2):
            with self.client.patch(
                path,
                headers=headers,
                json=payload,
                catch_response=True,
                name=name,
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    response.success()
                    self.reset_auth_state()
                    if self.ensure_authenticated(force_refresh=True):
                        headers = self.authenticated_headers()
                        continue
                    response.failure(f"{name} authentication recovery failed")
                    return None
                return self._parse_json_response(response, name, expected_fields, {200})
        return None

    def _parse_json_response(
        self,
        response,
        name: str,
        expected_fields: tuple[str, ...],
        success_statuses: set[int],
    ) -> Optional[Dict[str, Any]]:
        if response.status_code not in success_statuses:
            response.failure(f"{name} failed: {response.status_code}")
            return None
        try:
            data = response.json()
        except json.JSONDecodeError:
            response.failure(f"{name} returned invalid JSON")
            return None
        missing = [field for field in expected_fields if field not in data]
        if missing:
            response.failure(f"{name} missing fields: {missing}")
            return None
        response.success()
        return data


def _all_selectable_user_classes() -> tuple[type[HttpUser], ...]:
    """Return every Locust user class controlled by load profiles.

    WHY: Locust auto-discovers top-level ``HttpUser`` subclasses, so profile
    selection must actively disable classes that are not part of the current run.
    """
    return (
        HealthCheckUser,
        HealthCheckFullUser,
        AuthUserFlow,
        TasksMonitoringUser,
        LLMMockWorkspaceUser,
    )


# Configure user weights from profile
def get_user_classes():
    """Get user classes with weights from current profile."""
    profile = get_profile()

    classes = []
    for class_name, weight in profile.weight_config.items():
        if class_name == "HealthCheckUser":
            HealthCheckUser.weight = weight
            classes.append(HealthCheckUser)
        elif class_name == "HealthCheckFullUser":
            HealthCheckFullUser.weight = weight
            classes.append(HealthCheckFullUser)
        elif class_name == "AuthUserFlow":
            AuthUserFlow.weight = weight
            classes.append(AuthUserFlow)
        elif class_name == "LLMMockWorkspaceUser":
            LLMMockWorkspaceUser.weight = weight
            classes.append(LLMMockWorkspaceUser)
        elif class_name == "TasksMonitoringUser":
            TasksMonitoringUser.weight = weight
            classes.append(TasksMonitoringUser)

    active_class_names = {user_class.__name__ for user_class in classes}
    for user_class in _all_selectable_user_classes():
        user_class.abstract = user_class.__name__ not in active_class_names

    return classes


def _env_flag_enabled(name: str) -> bool:
    """Return whether an environment flag is explicitly enabled.

    WHY: Locust startup checks need a shared parser for opt-out flags so the
    preflight behaviour stays predictable across CI and local runs.
    """
    return os.getenv(name, "false").lower() in {"1", "true", "yes", "on"}


def _profile_has_auth_workloads(profile) -> bool:
    """Return whether the active profile includes auth-dependent users.

    WHY: Pure health-check profiles should stay cheap and not create test users,
    while auth-based profiles should fail fast if the target is not ready.
    """
    return any(
        class_name in AUTH_DEPENDENT_USER_CLASSES and weight > 0
        for class_name, weight in profile.weight_config.items()
    )


def _target_host(environment) -> str:
    """Resolve the effective host used by Locust for startup checks.

    WHY: Locust can receive the host from CLI, environment variables, or web UI;
    the preflight must interrogate the same target that workers will exercise.
    """
    return str(environment.host or get_host_url()).rstrip("/")


def _post_json_preflight(url: str, payload: Dict[str, str]) -> tuple[int, str]:
    """Send a small JSON POST request for startup validation.

    WHY: Register/login readiness needs a synchronous check before spawning many
    users, but that check should remain independent from Locust user instances.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="ignore")


def _validate_auth_preflight(environment, profile) -> None:
    """Fail fast when auth-based workloads target an unprepared environment.

    WHY: Without this check Locust spends the whole run reporting secondary 401s
    and 500s when the real issue is a missing migration or a broken auth setup.
    """
    if not _profile_has_auth_workloads(profile) or _env_flag_enabled(
        "LOCUST_SKIP_AUTH_PREFLIGHT"
    ):
        return

    host = _target_host(environment)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    username = f"locustpreflight{suffix}"
    password = "SecurePass123!"

    register_status, register_body = _post_json_preflight(
        f"{host}/api/v1/auth/register",
        {
            "username": username,
            "email": f"{username}@example.com",
            "password": password,
            "first_name": "Locust",
            "last_name": "Preflight",
        },
    )
    if register_status != 201:
        detail = register_body.strip() or "no response body"
        raise RuntimeError(
            "Auth preflight failed on register with HTTP "
            f"{register_status}. The target environment is not ready for "
            "auth-dependent Locust scenarios. Common causes: missing database "
            "migrations, missing `users` table, or broken auth dependencies. "
            "For a fresh local test database, run "
            "`ENVIRONMENT=testing poetry run python scripts/bootstrap_test_db.py`; "
            "for an existing migrated database, use `poetry run alembic upgrade head`, "
            f"then retry. Response: {detail}"
        )

    login_status, login_body = _post_json_preflight(
        f"{host}/api/v1/auth/login",
        {"username": username, "password": password},
    )
    if login_status != 200:
        detail = login_body.strip() or "no response body"
        raise RuntimeError(
            "Auth preflight failed on login with HTTP "
            f"{login_status}. Registration succeeded but interactive auth is not "
            "usable for Locust. Check auth settings such as active/verified-user "
            f"requirements and Redis/session configuration. Response: {detail}"
        )


def _abort_test_start(environment, message: str) -> None:
    """Stop the Locust run immediately with a clear startup failure reason.

    WHY: Event handler exceptions are only logged by Locust, so an explicit abort
    is needed to prevent user spawning after a failed environment preflight.
    """
    print(f"\n❌ {message}")
    environment.process_exit_code = 1
    if environment.runner is not None:
        environment.runner.quit()
    raise SystemExit(message)


# Bind the active classes at import time so Locust does not auto-discover every
# top-level HttpUser subclass and accidentally run profiles outside the selected one.
user_classes = get_user_classes()


# Event handlers for enhanced reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start with configuration."""
    profile = get_profile()
    target_host = _target_host(environment)
    try:
        if profile.name == "llm_mock":
            _validate_llm_mock_guard()
        _validate_auth_preflight(environment, profile)
    except RuntimeError as exc:
        _abort_test_start(environment, str(exc))
    print(f"\n🚀 Starting performance test with profile: {profile.name}")
    print(f"📋 Description: {profile.description}")
    print(f"👥 Users: {profile.users}")
    print(f"⚡ Spawn rate: {profile.spawn_rate}/s")
    print(f"⏱️  Duration: {profile.run_time}")
    print(f"🎯 Target: {target_host}")
    print(f"📊 Weights: {profile.weight_config}")
    print("-" * 50)


def _validate_llm_mock_guard() -> None:
    """Fail fast when local no-cost guardrails are not explicitly enabled.

    WHY: The ``llm_mock`` profile intentionally exercises agent and embedding
    endpoints. Requiring explicit emulation variables prevents accidental runs
    against paid providers when Locust and the API share the same environment.
    """
    if os.getenv("LOCUST_ALLOW_UNSAFE_LLM_TRAFFIC", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    missing = []
    if os.getenv("LLM_EMULATION_ENABLED", "").lower() != "true":
        missing.append("LLM_EMULATION_ENABLED=true")

    embedding_backend = os.getenv("EMBEDDING_BACKEND", "")
    if not is_llm_mock_embedding_backend(embedding_backend):
        missing.append(
            "EMBEDDING_BACKEND in {" + format_llm_mock_embedding_backends() + "}"
        )

    if missing:
        raise RuntimeError(
            "LOAD_PROFILE=llm_mock requires local no-cost guardrails: "
            + ", ".join(missing)
            + ". Set them for the API/Locust environment, or set "
            "LOCUST_ALLOW_UNSAFE_LLM_TRAFFIC=true only when you intentionally "
            "target a separately guarded test server."
        )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion."""
    profile = get_profile()
    print(f"\n✅ Performance test completed: {profile.name}")
    print(
        f"📈 Report available at: reports/performance/{profile.name}_load_report.html"
    )
    print("-" * 50)


# This allows Locust to automatically discover user classes
# when the file is run directly
if __name__ == "__main__":
    print(f"Loaded user classes: {[cls.__name__ for cls in user_classes]}")
