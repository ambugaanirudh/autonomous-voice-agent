"""Unit tests for the Agent core loop."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent import Action, Agent, LoopResult
from src.browser.controller import BrowserController, BrowserConfig
from src.config import AppConfig
from src.executor.executor import Executor
from src.planner.planner import Planner
from src.safety.guard import SafetyGuard
from src.state import State
from src.stt.listener import Listener
from src.vision.grounding import GroundingGuard
from src.vision.vlm import VLMClient


@pytest.fixture
def config() -> AppConfig:
    return AppConfig.from_file("config.yaml")


@pytest.fixture
def mock_dependencies() -> dict:
    return {
        "listener": MagicMock(spec=Listener),
        "planner": MagicMock(spec=Planner),
        "browser": MagicMock(spec=BrowserController),
        "executor": MagicMock(spec=Executor),
        "vlm": MagicMock(spec=VLMClient),
        "grounding_guard": MagicMock(spec=GroundingGuard),
        "safety": MagicMock(spec=SafetyGuard),
    }


@pytest.fixture
def agent(config: AppConfig, mock_dependencies: dict) -> Agent:
    return Agent(config=config, state=State(), **mock_dependencies)


def test_agent_initial_state(agent: Agent) -> None:
    assert agent.state.goal == ""
    assert agent.state.plan == []
    assert agent.state.retry_count == 0
    assert agent.max_steps == 50


def test_action_enum_values() -> None:
    assert Action.GOTO == "goto"
    assert Action.CLICK == "click"
    assert Action.TYPE == "type"
    assert Action.STOP == "stop"


def test_loop_result_defaults() -> None:
    r = LoopResult(success=True, action=Action.CLICK)
    assert r.success is True
    assert r.action == Action.CLICK
    assert r.target is None
    assert r.detail == ""


def test_record_step_appends_to_state(agent: Agent) -> None:
    agent._record_step(1, "click", "button#buy", True, "Clicked buy")
    assert len(agent.state.completed_steps) == 1
    assert agent.state.completed_steps[0].action == "click"
    assert agent.state.completed_steps[0].success is True