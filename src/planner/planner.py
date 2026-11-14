"""Planner: converts voice commands into ordered action plans.

Rule-based planner for predictability. Maps high-level intents to sequences
of actions the executor can carry out. Replace with VLM-based planner for
more complex domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlanStep:
    """A single step in the plan."""
    action: str           # "goto", "click", "type", "wait", "stop"
    target: Optional[str] = None   # element_id or URL
    value: Optional[str] = None    # text to type
    description: str = ""


class Planner:
    def __init__(self, store_url: str, dashboard_url: str) -> None:
        self.store_url = store_url
        self.dashboard_url = dashboard_url

    def plan(self, command: str) -> list[PlanStep]:
        """Convert a voice command into an ordered list of PlanSteps."""
        cmd = command.lower().strip()

        # --- Mock store flows ---
        if "add" in cmd and "cart" in cmd:
            if "widget" in cmd:
                target = "widget-001"
            elif "gadget" in cmd:
                target = "gadget-002"
            elif "doohickey" in cmd:
                target = "doohickey-003"
            else:
                target = None
            return [
                PlanStep("goto", self.store_url, None, "Navigate to store"),
                PlanStep("click", target, None, f"Add {target or 'product'} to cart"),
            ]

        if "checkout" in cmd or "purchase" in cmd or "buy" in cmd:
            return [
                PlanStep("goto", self.store_url, None, "Navigate to store"),
                PlanStep("click", None, None, "Add item to cart"),
                PlanStep("click", None, None, "Go to cart"),
                PlanStep("click", None, None, "Proceed to checkout"),
            ]

        if "cart" in cmd:
            cart_url = self.store_url.rsplit("/", 1)[0] + "/cart.html"
            return [PlanStep("goto", cart_url, None, "Open cart")]

        # --- Dashboard flows ---
        if "dashboard" in cmd or "metrics" in cmd or "revenue" in cmd:
            return [PlanStep("goto", self.dashboard_url, None, "Open dashboard")]

        # --- Generic navigation ---
        if "go to" in cmd or "navigate to" in cmd:
            return [PlanStep("goto", self.store_url, None, "Navigate to store")]

        # Unknown command
        return [PlanStep("stop", None, None, "Unknown command")]

    def is_goal_complete(self, command: str, current_url: str, page_title: str) -> bool:
        """Heuristic check: has the goal been satisfied based on current page?"""
        cmd = command.lower().strip()

        if "cart" in cmd and "/cart" in current_url:
            return True
        if ("checkout" in cmd or "purchase" in cmd or "buy" in cmd) and "/checkout" in current_url:
            return True
        if "dashboard" in cmd and "/dashboard" in current_url:
            return True
        if "store" in cmd and "/store" in current_url:
            return True

        return False