"""Core agent loop skeleton.

Defines the Action enum, the Agent class, and the seven-step loop:
listen -> plan -> observe -> ground -> act -> verify -> replan.

No model calls are made here. Each step delegates to a module that will be
implemented in later phases (STT, planner, browser, vision, executor, safety).
Stub methods raise NotImplementedError so missing pieces are obvious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

from loguru import logger

from src.config import AppConfig
from src.state import State, StepRecord


class Action(StrEnum):
    """Actions the agent can take."""
    GOTO = "goto"
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    CONFIRM = "confirm"   # safety-gated irreversible action
    REPLAN = "replan"
    STOP = "stop"


@dataclass
class LoopResult:
    """Outcome of a single loop iteration."""
    success: bool
    action: Action
    target: Optional[str] = None
    detail: str = ""


class Agent:
    """Autonomous voice-to-action web agent.

    State lives in self.state (independent of model context). The core loop
    runs until the goal is complete, max_steps is reached, or a STOP action
    is returned.
    """

    def __init__(self, config: AppConfig, state: State) -> None:
        self.config = config
        self.state = state
        self.max_steps = config.safety.max_steps
        self.logger = logger

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(self, goal: str) -> State:
        """Execute the core loop until completion or limit."""
        self.state.goal = goal
        self.state.plan = []
        self.state.completed_steps = []
        self.state.retry_count = 0
        self.state.current_url = None
        self.state.last_action = None

        self.logger.info("Agent starting. Goal: {goal}", goal=goal)

        step = 0
        while step < self.max_steps:
            step += 1
            self.logger.debug("Loop iteration {step}/{max_steps}", step=step, max_steps=self.max_steps)

            # 1. Listen (STT + wake word) — Phase 2
            command = self._listen()
            if command is None:
                # No wake word / silence / false trigger — keep listening
                self.logger.debug("No command heard; continuing to listen")
                continue
            if command.lower() in ("cancel", "stop", "abort"):
                self.logger.info("User cancelled the task")
                self.state.last_action = "cancel"
                self._record_step(step, "cancel", None, True, "User cancelled")
                break

            # 2. Plan (VLM intent -> plan) — Phase 5
            plan = self._plan(command)
            self.state.plan = plan
            self.logger.info("Plan: {plan}", plan=" -> ".join(plan))

            # 3. Observe (screenshot + accessibility tree) — Phase 3
            observation = self._observe()
            self.state.current_url = observation.get("url")
            self.logger.debug("Observed URL: {url}", url=self.state.current_url)

            # 4. Ground (VLM picks element ID via Set-of-Marks) — Phase 4
            grounded = self._ground(command, plan, observation)
            if grounded is None:
                self.logger.warning("Grounding failed — no element selected")
                self.state.retry_count += 1
                if self.state.retry_count >= self.state.max_retries:
                    self.logger.error("Max retries reached; stopping")
                    break
                continue

            # 5. Act (Playwright click/type) — Phase 6
            result = self._act(grounded.action, grounded.target, grounded.detail)
            self.state.last_action = grounded.action.value
            self._record_step(step, grounded.action.value, grounded.target, result.success, result.detail)

            # 6. Verify (URL, DOM diff, expected text, screenshot) — Phase 6
            verified = self._verify(grounded, result)
            if not verified:
                self.logger.warning("Verification failed; replanning")
                self.state.retry_count += 1
                continue

            # 7. Replan / check completion — Phase 5
            self.state.retry_count = 0  # reset on success
            if self._is_goal_complete(observation):
                self.logger.info("Goal complete")
                self.state.last_action = "stop"
                self._record_step(step, "stop", None, True, "Goal verified complete")
                break

        self.logger.info("Agent finished. Steps taken: {step}", step=step)
        return self.state

    # ------------------------------------------------------------------
    # Stub steps — implemented in later phases
    # ------------------------------------------------------------------

    def _listen(self) -> Optional[str]:
        """Phase 2: STT + wake word + VAD. Returns command text or None."""
        raise NotImplementedError("STT module not yet implemented (Phase 2)")

    def _plan(self, command: str) -> list[str]:
        """Phase 5: VLM intent -> ordered list of actions."""
        raise NotImplementedError("Planner not yet implemented (Phase 5)")

    def _observe(self) -> dict:
        """Phase 3: Playwright screenshot + accessibility tree extraction."""
        raise NotImplementedError("Browser observe not yet implemented (Phase 3)")

    def _ground(self, command: str, plan: list[str], observation: dict) -> Optional[LoopResult]:
        """Phase 4: Set-of-Marks overlay + VLM element selection."""
        raise NotImplementedError("Grounding not yet implemented (Phase 4)")

    def _act(self, action: Action, target: Optional[str], detail: str) -> LoopResult:
        """Phase 6: Playwright click/type + safety confirmation."""
        raise NotImplementedError("Executor not yet implemented (Phase 6)")

    def _verify(self, grounded: LoopResult, result: LoopResult) -> bool:
        """Phase 6: URL change, DOM diff, expected text, screenshot comparison."""
        raise NotImplementedError("Verification not yet implemented (Phase 6)")

    def _is_goal_complete(self, observation: dict) -> bool:
        """Phase 5: Check if goal is satisfied based on observation."""
        raise NotImplementedError("Goal completion check not yet implemented (Phase 5)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_step(self, step: int, action: str, target: Optional[str],
                     success: bool, detail: str) -> None:
        record = StepRecord(
            step=step,
            action=action,
            target=target,
            success=success,
            detail=detail,
        )
        self.state.completed_steps.append(record)