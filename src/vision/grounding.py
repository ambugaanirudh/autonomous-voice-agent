"""Anti-hallucination grounding guard.

Validates that VLM-chosen element IDs exist in the current accessibility tree.
Rejects any element ID that doesn't exist — this is the core anti-hallucination
mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.browser.dom import AccessibilityNode, find_node_by_id


@dataclass
class GroundedAction:
    """A validated action ready for the executor."""
    action: str           # "click", "type", "wait", "stop"
    element_id: Optional[str]
    text: Optional[str]
    reasoning: str
    confidence: float
    validated: bool       # True if element_id exists in the tree (or is None for stop/wait)


class GroundingGuard:
    def __init__(self, nodes: list[AccessibilityNode]) -> None:
        self.nodes = nodes

    def validate(self, response) -> GroundedAction:
        """Check that the VLM's chosen element_id exists in the accessibility tree.

        Returns a GroundedAction with validated=True only if the element exists
        (or the action doesn't require an element, like wait/stop).
        """
        from src.vision.vlm import VLMResponse

        if not isinstance(response, VLMResponse):
            action = getattr(response, "action", "stop")
            element_id = getattr(response, "element_id", None)
            text = getattr(response, "text", None)
            reasoning = getattr(response, "reasoning", "")
            confidence = getattr(response, "confidence", 0.0)
        else:
            action = response.action
            element_id = response.element_id
            text = response.text
            reasoning = response.reasoning
            confidence = response.confidence

        # Actions that don't need an element
        if action in ("wait", "stop"):
            return GroundedAction(
                action=action,
                element_id=None,
                text=None,
                reasoning=reasoning,
                confidence=confidence,
                validated=True,
            )

        # click/type require a valid element_id
        if not element_id:
            return GroundedAction(
                action="stop",
                element_id=None,
                text=None,
                reasoning="No element_id provided for action " + action,
                confidence=0.0,
                validated=False,
            )

        if not find_node_by_id(self.nodes, element_id):
            return GroundedAction(
                action="stop",
                element_id=element_id,
                text=text,
                reasoning=f"Hallucinated element_id '{element_id}' not in accessibility tree",
                confidence=confidence,
                validated=False,
            )

        return GroundedAction(
            action=action,
            element_id=element_id,
            text=text,
            reasoning=reasoning,
            confidence=confidence,
            validated=True,
        )