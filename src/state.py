"""Explicit agent state object.

State is persisted independently of the model context so the agent never
relies on the LLM's memory for goal tracking, retry counts, or completion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
import json


@dataclass
class StepRecord:
    """A single completed (or attempted) step in the plan."""
    step: int
    action: str
    target: Optional[str]
    success: bool
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class State:
    goal: str = ""
    plan: list[str] = field(default_factory=list)
    completed_steps: list[StepRecord] = field(default_factory=list)
    current_url: Optional[str] = None
    last_action: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist state to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def load(cls, path: str | Path) -> "State":
        """Load state from a JSON file. Returns an empty State if missing."""
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        steps = [StepRecord(**s) for s in data.pop("completed_steps", [])]
        return cls(completed_steps=steps, **data)