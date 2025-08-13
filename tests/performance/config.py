"""
Performance test configuration for different load profiles.

This module defines various load testing scenarios for the FastAPI backend,
ranging from light smoke tests to heavy stress testing.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Type


@dataclass
class LoadProfile:
    """Configuration for a specific load testing profile."""

    name: str
    description: str
    users: int
    spawn_rate: float  # users per second
    run_time: str  # duration like "30s", "5m", "1h"
    weight_config: Dict[str, int]  # User class weights


# Define load profiles
LOAD_PROFILES = {
    "light": LoadProfile(
        name="light",
        description="Light load for smoke testing and development",
        users=10,
        spawn_rate=1.0,
        run_time="1m",
        weight_config={
            "HealthCheckUser": 3,
            "HealthCheckFullUser": 1,
            "AuthUserFlow": 4,
            "TasksMonitoringUser": 1,
        },
    ),
    "medium": LoadProfile(
        name="medium",
        description="Medium load for regular performance testing",
        users=20,
        spawn_rate=2.0,
        run_time="5m",
        weight_config={
            "HealthCheckUser": 6,
            "HealthCheckFullUser": 1,
            "AuthUserFlow": 9,
            "TasksMonitoringUser": 1,
        },
    ),
    "heavy": LoadProfile(
        name="heavy",
        description="Heavy load for stress testing",
        users=50,
        spawn_rate=5.0,
        run_time="10m",
        weight_config={
            "HealthCheckUser": 19,
            "HealthCheckFullUser": 1,
            "AuthUserFlow": 25,
            "TasksMonitoringUser": 1,
        },
    ),
    "spike": LoadProfile(
        name="spike",
        description="Spike testing with rapid user ramp-up",
        users=100,
        spawn_rate=10.0,
        run_time="3m",
        weight_config={
            "HealthCheckUser": 28,
            "HealthCheckFullUser": 2,
            "AuthUserFlow": 50,
            "TasksMonitoringUser": 2,
        },
    ),
    "endurance": LoadProfile(
        name="endurance",
        description="Long-running endurance test",
        users=50,
        spawn_rate=1.0,
        run_time="30m",
        weight_config={
            "HealthCheckUser": 19,
            "HealthCheckFullUser": 1,
            "AuthUserFlow": 25,
            "TasksMonitoringUser": 1,
        },
    ),
}


def get_profile(profile_name: str = None) -> LoadProfile:
    """
    Get load profile by name or from environment variable.

    Args:
        profile_name: Name of the profile to get

    Returns:
        LoadProfile configuration

    Raises:
        ValueError: If profile name is invalid
    """
    if profile_name is None:
        profile_name = os.getenv("LOAD_PROFILE", "light")

    if profile_name not in LOAD_PROFILES:
        available = ", ".join(LOAD_PROFILES.keys())
        raise ValueError(f"Invalid profile '{profile_name}'. Available: {available}")

    return LOAD_PROFILES[profile_name]


def get_host_url() -> str:
    """
    Get the target host URL for testing.

    Returns:
        Base URL for the API server
    """
    return os.getenv("TARGET_HOST", "http://localhost:8000")


def get_report_path(profile_name: str = None) -> str:
    """
    Get the path for HTML report output.

    Args:
        profile_name: Load profile name for report naming

    Returns:
        Path to save the HTML report
    """
    if profile_name is None:
        profile_name = os.getenv("LOAD_PROFILE", "light")

    return f"reports/performance/{profile_name}_load_report.html"


# Performance thresholds for quick regression detection
PERFORMANCE_THRESHOLDS = {
    "health_p95_ms": 100,  # Health endpoints should be fast
    "auth_p95_ms": 500,  # Auth operations can be slower due to password hashing
    "tasks_p95_ms": 200,  # Task monitoring should be responsive
    "overall_failure_rate": 0.01,  # Less than 1% failure rate
}


def validate_performance(stats: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate performance against defined thresholds.

    Args:
        stats: Performance statistics from Locust

    Returns:
        Dictionary of threshold check results
    """
    results = {}

    # This is a placeholder for threshold validation
    # In a real implementation, you would parse the stats
    # and check against PERFORMANCE_THRESHOLDS

    for threshold_name in PERFORMANCE_THRESHOLDS:
        results[threshold_name] = True  # Placeholder

    return results
