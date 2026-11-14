"""Unit tests for the anti-hallucination grounding guard."""
from __future__ import annotations

from src.vision.grounding import GroundingGuard
from src.browser.dom import AccessibilityNode
from src.vision.vlm import VLMResponse


def test_validate_valid_click() -> None:
    nodes = [
        AccessibilityNode(role="button", name="Buy", element_id="e1"),
        AccessibilityNode(role="link", name="Home", element_id="e2"),
    ]
    guard = GroundingGuard(nodes)
    response = VLMResponse(
        action="click", element_id="e1", text=None,
        reasoning="User wants to buy", confidence=0.9,
    )
    result = guard.validate(response)
    assert result.validated is True
    assert result.action == "click"
    assert result.element_id == "e1"


def test_validate_hallucinated_element_rejected() -> None:
    nodes = [
        AccessibilityNode(role="button", name="Buy", element_id="e1"),
    ]
    guard = GroundingGuard(nodes)
    response = VLMResponse(
        action="click", element_id="e99", text=None,
        reasoning="Fake element", confidence=0.8,
    )
    result = guard.validate(response)
    assert result.validated is False
    assert result.action == "stop"  # guard overrides to stop
    assert "not in accessibility tree" in result.reasoning


def test_validate_wait_no_element_needed() -> None:
    nodes = []
    guard = GroundingGuard(nodes)
    response = VLMResponse(
        action="wait", element_id=None, text=None,
        reasoning="Page loading", confidence=0.5,
    )
    result = guard.validate(response)
    assert result.validated is True
    assert result.action == "wait"


def test_validate_stop_no_element_needed() -> None:
    nodes = []
    guard = GroundingGuard(nodes)
    response = VLMResponse(
        action="stop", element_id=None, text=None,
        reasoning="Goal complete", confidence=1.0,
    )
    result = guard.validate(response)
    assert result.validated is True
    assert result.action == "stop"


def test_validate_missing_element_id_for_click() -> None:
    nodes = [AccessibilityNode(role="button", name="Buy", element_id="e1")]
    guard = GroundingGuard(nodes)
    response = VLMResponse(
        action="click", element_id=None, text=None,
        reasoning="", confidence=0.3,
    )
    result = guard.validate(response)
    assert result.validated is False
    assert result.action == "stop"