"""Unit tests for the State object."""
from __future__ import annotations

import os
import tempfile

from src.state import State, StepRecord


def test_state_defaults() -> None:
    s = State()
    assert s.goal == ""
    assert s.plan == []
    assert s.completed_steps == []
    assert s.current_url is None
    assert s.last_action is None
    assert s.retry_count == 0
    assert s.max_retries == 3


def test_state_save_and_load_roundtrip() -> None:
    s = State(
        goal="Test goal",
        plan=["step one", "step two"],
        max_retries=5,
    )
    s.completed_steps.append(
        StepRecord(
            step=1,
            action="click",
            target="button#buy",
            success=True,
            detail="Clicked buy button",
            timestamp="2024-01-01T00:00:00Z",
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        s.save(tmp_path)
        loaded = State.load(tmp_path)

        assert loaded.goal == "Test goal"
        assert loaded.plan == ["step one", "step two"]
        assert loaded.max_retries == 5
        assert len(loaded.completed_steps) == 1
        assert loaded.completed_steps[0].action == "click"
        assert loaded.completed_steps[0].success is True
    finally:
        os.unlink(tmp_path)


def test_state_load_missing_file_returns_empty() -> None:
    s = State.load("nonexistent_state.json")
    assert s.goal == ""
    assert s.plan == []