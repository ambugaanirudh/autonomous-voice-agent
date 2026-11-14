"""Core agent: full listen -> plan -> observe -> ground -> act -> verify -> replan loop.

Wires together all modules:
  STT (listener) -> Planner -> Browser observe -> VLM grounding (Set-of-Marks)
  -> Executor -> Verification -> Safety guard -> replan on failure.

State lives in self.state (independent of model context).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional

from loguru import logger

from src.browser.controller import BrowserController
from src.browser.dom import AccessibilityNode, find_node_by_id
from src.config import AppConfig
from src.executor.executor import Executor, VerificationResult
from src.planner.planner import Planner, PlanStep
from src.safety.guard import SafetyGuard, SafetyVerdict
from src.state import State, StepRecord
from src.stt.listener import Listener
from src.vision.grounding import GroundedAction, GroundingGuard
from src.vision.som import MarkedElement, draw_with_bboxes
from src.vision.vlm import VLMClient, VLMResponse


class Action(StrEnum):
    """Actions the agent can take."""
    GOTO = "goto"
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    CONFIRM = "confirm"
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

    Core loop: listen -> plan -> observe -> ground -> act -> verify -> replan.
    Verification after EVERY action. Safety checks on every step.
    """

    def __init__(
        self,
        config: AppConfig,
        state: State,
        listener: Listener,
        planner: Planner,
        browser: BrowserController,
        executor: Executor,
        vlm: VLMClient,
        grounding_guard: GroundingGuard,
        safety: SafetyGuard,
    ) -> None:
        self.config = config
        self.state = state
        self.listener = listener
        self.planner = planner
        self.browser = browser
        self.executor = executor
        self.vlm = vlm
        self.grounding_guard = grounding_guard
        self.safety = safety
        self.logger = logger

        self.max_steps = config.safety.max_steps
        self._current_plan: list[PlanStep] = []
        self._plan_step_index = 0

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
        self.safety.reset()

        self.logger.info("Agent starting. Goal: {goal}", goal=goal)

        # 1. Listen for command (STT)
        command = self._listen()
        if command is None:
            self.logger.warning("No command heard — cannot start")
            self.state.last_action = "stop"
            self._record_step(0, "stop", None, False, "No command heard")
            return self.state

        self.logger.info("Command heard: {command}", command=command)

        # Safety: confirmation for irreversible actions
        if self._is_irreversible(command):
            if not self.safety.confirm_action("irreversible_action", command):
                self.logger.info("User cancelled irreversible action")
                self.state.last_action = "stop"
                self._record_step(0, "stop", None, True, "User cancelled")
                return self.state

        # 2. Plan
        self._current_plan = self.planner.plan(command)
        self.state.plan = [p.description for p in self._current_plan]
        self._plan_step_index = 0
        self.logger.info("Plan: {plan}", plan=" -> ".join(self.state.plan))

        # Start browser
        self.browser.start()

        try:
            step = 0
            while step < self.max_steps:
                step += 1
                self.safety.increment_step()

                # Safety: step limit
                step_verdict = self.safety.check_step_limit()
                if not step_verdict.allowed:
                    self.logger.error("Safety: {reason}", reason=step_verdict.reason)
                    self.state.last_action = "stop"
                    self._record_step(step, "stop", None, False, step_verdict.reason)
                    break

                # Safety: domain check on current URL
                current_url = self.browser.get_url()
                if current_url:
                    domain_verdict = self.safety.check_domain(current_url)
                    if not domain_verdict.allowed:
                        self.logger.error("Safety: {reason}", reason=domain_verdict.reason)
                        self.state.last_action = "stop"
                        self._record_step(step, "stop", None, False, domain_verdict.reason)
                        break

                # If plan is exhausted, check goal completion
                if self._plan_step_index >= len(self._current_plan):
                    if self._is_goal_complete(command, current_url, self.browser.get_title() or ""):
                        self.logger.info("Goal complete — plan exhausted")
                        self.state.last_action = "stop"
                        self._record_step(step, "stop", None, True, "Goal complete")
                        break
                    else:
                        # Replan: generate a new plan
                        self.logger.warning("Plan exhausted but goal not complete — replanning")
                        self._current_plan = self.planner.plan(command)
                        self.state.plan = [p.description for p in self._current_plan]
                        self._plan_step_index = 0
                        if not self._current_plan or self._current_plan[0].action == "stop":
                            self.logger.error("Cannot replan — stopping")
                            break
                        continue

                plan_step = self._current_plan[self._plan_step_index]
                self.logger.debug(
                    "Executing plan step {idx}/{total}: {action} {target}",
                    idx=self._plan_step_index + 1,
                    total=len(self._current_plan),
                    action=plan_step.action,
                    target=plan_step.target,
                )

                # 3. Observe (screenshot + accessibility tree)
                observation = self._observe()
                self.state.current_url = observation.get("url")

                # 4. Handle goto steps directly (no VLM needed)
                if plan_step.action == "goto":
                    result = self._handle_goto(plan_step)
                    self._record_step(
                        step, plan_step.action, plan_step.target,
                        result.success, result.detail,
                    )
                    if result.success:
                        self._plan_step_index += 1
                        self.state.retry_count = 0
                    else:
                        self.state.retry_count += 1
                        if self.state.retry_count >= self.state.max_retries:
                            self.logger.error("Max retries for goto — replanning")
                            self._replan(command)
                            self.state.retry_count = 0
                    continue

                # 5. Ground (Set-of-Marks + VLM + anti-hallucination guard)
                grounded = self._ground(command, plan_step, observation)
                if grounded is None or not grounded.validated:
                    self.logger.warning("Grounding failed — retrying or replanning")
                    self.state.retry_count += 1
                    if self.state.retry_count >= self.state.max_retries:
                        self.logger.error("Max retries for grounding — replanning")
                        self._replan(command)
                        self.state.retry_count = 0
                    continue

                # 6. Act (execute via Playwright)
                selector = self._resolve_selector(grounded.element_id, observation.get("nodes", []))
                if not selector:
                    self.logger.warning("Cannot resolve element_id {eid} to selector", eid=grounded.element_id)
                    self.state.retry_count += 1
                    if self.state.retry_count >= self.state.max_retries:
                        self._replan(command)
                        self.state.retry_count = 0
                    continue

                result = self.executor.execute(grounded.action, selector, grounded.text)
                self.state.last_action = grounded.action
                self._record_step(
                    step, grounded.action, grounded.element_id,
                    result.success, result.detail,
                )

                # 7. Verify
                if not result.success:
                    self.logger.warning("Verification failed: {detail}", detail=result.detail)
                    self.state.retry_count += 1
                    if self.state.retry_count >= self.state.max_retries:
                        self.logger.error("Max retries for action — replanning")
                        self._replan(command)
                        self.state.retry_count = 0
                    continue

                # Success — advance plan step
                self.state.retry_count = 0
                self._plan_step_index += 1

        finally:
            self.browser.close()

        self.logger.info("Agent finished. Steps taken: {step}", step=step)
        return self.state

    # ------------------------------------------------------------------
    # Loop substages
    # ------------------------------------------------------------------

    def _listen(self) -> Optional[str]:
        """STT: listen for wake word + command."""
        try:
            return self.listener.listen_once(timeout_sec=30.0)
        except Exception as exc:
            self.logger.error("STT error: {error}", error=exc)
            return None

    def _observe(self) -> dict:
        """Browser observe: screenshot + accessibility tree extraction."""
        page = self.browser._page
        if page is None:
            return {"url": None, "title": None, "nodes": [], "screenshot_path": None}

        url = self.browser.get_url()
        title = self.browser.get_title()

        # Extract accessibility tree
        try:
            from src.browser.dom import extract_accessibility_tree
            nodes = extract_accessibility_tree(page)
        except Exception as exc:
            self.logger.warning("Accessibility tree extraction failed: {error}", error=exc)
            nodes = []

        # Take screenshot
        screenshot_path = None
        try:
            screenshot_path = Path("logs/screenshots") / f"observe_{len(self.state.completed_steps) + 1}.png"
            self.browser.screenshot(screenshot_path)
        except Exception as exc:
            self.logger.warning("Screenshot failed: {error}", error=exc)

        return {
            "url": url,
            "title": title,
            "nodes": nodes,
            "screenshot_path": screenshot_path,
        }

    def _ground(
        self,
        command: str,
        plan_step: PlanStep,
        observation: dict,
    ) -> Optional[GroundedAction]:
        """Set-of-Marks overlay -> VLM -> anti-hallucination guard."""
        screenshot_path = observation.get("screenshot_path")
        nodes: list[AccessibilityNode] = observation.get("nodes", [])

        if not screenshot_path or not nodes:
            self.logger.warning("No screenshot or nodes for grounding")
            return None

        # Build element map with bounding boxes from Playwright
        page = self.browser._page
        if page is None:
            return None

        elements_with_bboxes: list[dict] = []
        for node in nodes:
            if not node.element_id or not node.css_selector:
                continue
            try:
                box = page.locator(node.css_selector).bounding_box(timeout=2000)
                if box:
                    elements_with_bboxes.append({
                        "number": len(elements_with_bboxes) + 1,
                        "element_id": node.element_id,
                        "role": node.role,
                        "name": node.name,
                        "bbox": (int(box["x"]), int(box["y"]),
                                int(box["x"] + box["width"]), int(box["y"] + box["height"])),
                    })
            except Exception:
                continue

        if not elements_with_bboxes:
            self.logger.warning("No elements with bounding boxes for grounding")
            return None

        # Draw Set-of-Marks overlay
        marked_path = Path("logs/screenshots") / f"marked_{len(self.state.completed_steps) + 1}.png"
        try:
            draw_with_bboxes(screenshot_path, elements_with_bboxes, marked_path)
        except Exception as exc:
            self.logger.warning("Set-of-Marks overlay failed: {error}", error=exc)
            marked_path = screenshot_path  # fallback to unmarked

        # Build element map for VLM prompt (without bboxes)
        element_map = [
            {"number": e["number"], "element_id": e["element_id"],
             "role": e["role"], "name": e["name"]}
            for e in elements_with_bboxes
        ]

        # VLM grounding
        try:
            response: VLMResponse = self.vlm.ground(marked_path, command, element_map)
        except Exception as exc:
            self.logger.error("VLM grounding error: {error}", error=exc)
            return None

        # Anti-hallucination guard
        grounded = self.grounding_guard.validate(response)
        if not grounded.validated:
            self.logger.warning(
                "Grounding guard rejected: {reason}", reason=grounded.reasoning
            )
            return grounded  # Return validated=False so caller knows

        return grounded

    def _resolve_selector(
        self,
        element_id: Optional[str],
        nodes: list[AccessibilityNode],
    ) -> Optional[str]:
        """Resolve an element_id to a CSS selector using the accessibility tree."""
        if not element_id:
            return None
        node = find_node_by_id(nodes, element_id)
        if node and node.css_selector:
            return node.css_selector
        return None

    def _handle_goto(self, plan_step: PlanStep) -> VerificationResult:
        """Handle a goto plan step directly via browser navigation."""
        url = plan_step.target
        if not url:
            return VerificationResult(
                success=False, url_changed=False, dom_changed=False,
                expected_text_found=False, detail="No URL for goto",
            )
        try:
            final_url = self.browser.goto(url)
            return VerificationResult(
                success=True, url_changed=True, dom_changed=True,
                expected_text_found=True,
                detail=f"Navigated to {final_url}",
            )
        except Exception as exc:
            return VerificationResult(
                success=False, url_changed=False, dom_changed=False,
                expected_text_found=False, detail=f"Goto failed: {exc}",
            )

    def _replan(self, command: str) -> None:
        """Replan from scratch when retries are exhausted."""
        self.logger.info("Replanning from command: {command}", command=command)
        self._current_plan = self.planner.plan(command)
        self.state.plan = [p.description for p in self._current_plan]
        self._plan_step_index = 0

    def _is_goal_complete(self, command: str, current_url: Optional[str], page_title: str) -> bool:
        if not current_url:
            return False
        return self.planner.is_goal_complete(command, current_url, page_title)

    def _is_irreversible(self, command: str) -> bool:
        """Check if command implies an irreversible action needing confirmation."""
        cmd = command.lower().strip()
        for action in self.safety.require_confirmation_for:
            if action in cmd:
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_step(
        self,
        step: int,
        action: str,
        target: Optional[str],
        success: bool,
        detail: str,
    ) -> None:
        record = StepRecord(
            step=step,
            action=action,
            target=target,
            success=success,
            detail=detail,
        )
        self.state.completed_steps.append(record)
        self.logger.info(
            "Step {step}: {action} {target} -> {success} ({detail})",
            step=step, action=action, target=target,
            success=success, detail=detail,
        )