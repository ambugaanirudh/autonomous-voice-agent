"""Pytest shared fixtures."""
from __future__ import annotations

import pytest

from src.state import State, StepRecord


@pytest.fixture
def empty_state() -> State:
    return State()


@pytest.fixture
def sample_state() -> State:
    return State(
        goal="Buy a widget from the mock store",
        plan=["Go to store", "Add widget to cart", "Checkout"],
        completed_steps=[
            StepRecord(
                step=1,
                action="goto",
                target="store",
                success=True,
                detail="Loaded store page",
                timestamp="2024-01-01T00:00:00Z",
            )
        ],
        current_url="http://127.0.0.1:8765/store/index.html",
        last_action="goto",
        retry_count=0,
    )