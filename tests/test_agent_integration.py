"""Integration tests for Agent with mocked dependencies."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent import Agent
from src.browser.controller import BrowserController, BrowserConfig
from src.browser.dom import AccessibilityNode
from src.config import AppConfig
from src.executor.executor import Executor, VerificationResult
from src.planner.planner import Planner
from src.safety.guard import SafetyGuard
from src.state import State
from src.stt.listener import Listener
from src.vision.grounding import GroundedAction, GroundingGuard
from src.vision.vlm import VLMClient, VLMResponse


@pytest.fixture
def config() -> AppConfig:
    return AppConfig.from_file("config.yaml")


@pytest.fixture
def mock_listener() -> Listener:
    m = MagicMock(spec=Listener)
    m.listen_once.return_value = "add widget to cart"
    return m


@pytest.fixture
def mock_planner() -> Planner:
    return Planner(
        store_url="http://127.0.0.1:8765/store/index.html",
        dashboard_url="http://127.0.0.1:8765/dashboard/index.html",
    )


@pytest.fixture
def mock_browser() -> BrowserController:
    m = MagicMock(spec=BrowserController)
    m._page = MagicMock()
    m.get_url.return_value = "http://127.0.0.1:8765/store/index.html"
    m.get_title.return_value = "Mock Store - Products"
    m.goto.return_value = "http://127.0.0.1:8765/store/index.html"
    m.screenshot.return_value = b"fake-image-bytes"
    return m


@pytest.fixture
def mock_executor() -> Executor:
    m = MagicMock(spec=Executor)
    m.execute.return_value = VerificationResult(
        success=True, url_changed=True, dom_changed=True,
        expected_text_found=True, detail="Navigated",
    )
    return m


@pytest.fixture
def mock_vlm() -> VLMClient:
    m = MagicMock(spec=VLMClient)
    m.ground.return_value = VLMResponse(
        action="click", element_id="e1", text=None,
        reasoning="Add to cart button", confidence=0.95,
    )
    return m


@pytest.fixture
def mock_grounding_guard() -> GroundingGuard:
    nodes = [AccessibilityNode(role="button", name="Add to Cart", element_id="e1")]
    guard = GroundingGuard(nodes)
    return guard


@pytest.fixture
def mock_safety() -> SafetyGuard:
    guard = SafetyGuard(
        allowlist_domains=["127.0.0.1"],
        max_steps=50,
        require_confirmation_for=["purchase"],
    )
    guard.confirm_action = MagicMock(return_value=True)
    return guard


def test_agent_runs_full_loop(
    config, mock_listener, mock_planner, mock_browser,
    mock_executor, mock_vlm, mock_grounding_guard, mock_safety,
) -> None:
    agent = Agent(
        config=config, state=State(),
        listener=mock_listener, planner=mock_planner,
        browser=mock_browser, executor=mock_executor,
        vlm=mock_vlm, grounding_guard=mock_grounding_guard,
        safety=mock_safety,
    )

    # Run with a max of 5 steps to avoid infinite loop
    agent.max_steps = 5

    # Patch _ground to return a valid grounded action (skip actual VLM)
    agent._ground = MagicMock(return_value=GroundedAction(
        action="click", element_id="e1", text=None,
        reasoning="test", confidence=0.9, validated=True,
    ))

    # Patch _resolve_selector to return a valid selector
    agent._resolve_selector = MagicMock(return_value='[role="button"]')

    state = agent.run("add widget to cart")

    assert state.last_action is not None
    assert len(state.completed_steps) > 0
    mock_listener.listen_once.assert_called_once()
    mock_browser.start.assert_called_once()
    mock_browser.close.assert_called_once()


def test_agent_handles_no_command(
    config, mock_listener, mock_planner, mock_browser,
    mock_executor, mock_vlm, mock_grounding_guard, mock_safety,
) -> None:
    mock_listener.listen_once.return_value = None  # silence

    agent = Agent(
        config=config, state=State(),
        listener=mock_listener, planner=mock_planner,
        browser=mock_browser, executor=mock_executor,
        vlm=mock_vlm, grounding_guard=mock_grounding_guard,
        safety=mock_safety,
    )
    agent.max_steps = 5

    state = agent.run("test")
    assert state.last_action == "stop"