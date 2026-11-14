"""Vision package: Set-of-Marks overlay + VLM grounding."""
from __future__ import annotations

from src.vision.som import draw_set_of_marks, draw_with_bboxes, save_marked_image
from src.vision.vlm import VLMClient, VLMResponse
from src.vision.grounding import GroundingGuard, GroundedAction

__all__ = [
    "draw_set_of_marks",
    "draw_with_bboxes",
    "save_marked_image",
    "VLMClient",
    "VLMResponse",
    "GroundingGuard",
    "GroundedAction",
]