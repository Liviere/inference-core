"""Tests for the demo tool-calling provider helpers."""

import json

import pytest

from inference_core.agents.tools.demo_tool_calling import _safe_eval, calculate


def test_safe_eval_allows_basic_arithmetic():
    """Basic arithmetic stays available for the demo calculator."""
    assert _safe_eval("(5 + 3) * 12") == 96
    assert _safe_eval("-5 + 2.5") == pytest.approx(-2.5)


def test_safe_eval_rejects_code_injection_payloads():
    """Non-arithmetic syntax is rejected before any Python object access."""
    with pytest.raises(ValueError, match="Invalid expression"):
        _safe_eval("__import__('os').system('id')")


def test_safe_eval_rejects_unsupported_operators():
    """Operators that increase cost or semantics are rejected explicitly."""
    with pytest.raises(ValueError, match="Unsupported operator"):
        _safe_eval("2**10")

    with pytest.raises(ValueError, match="Unsupported operator"):
        _safe_eval("7 // 2")


def test_safe_eval_rejects_overly_long_expressions():
    """Length bounds keep the demo calculator computationally small."""
    with pytest.raises(ValueError, match="too long"):
        _safe_eval("1+" * 80 + "1")


def test_calculate_returns_json_result_payload():
    """The tool keeps the same JSON contract for successful calculations."""
    payload = json.loads(calculate.func("8 / 2"))

    assert payload == {"expression": "8 / 2", "result": 4.0}


def test_calculate_returns_json_error_payload():
    """The tool serializes parser failures into the existing error shape."""
    payload = json.loads(calculate.func("2**1000"))

    assert payload["expression"] == "2**1000"
    assert "Unsupported operator" in payload["error"]
