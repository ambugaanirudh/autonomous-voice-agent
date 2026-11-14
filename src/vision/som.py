"""Set-of-Marks overlay: draw numbered bounding boxes on interactive elements.

The overlay converts the accessibility tree into a visual numbering scheme.
Each interactive element gets a number drawn on the screenshot. The VLM returns
the number instead of raw pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from src.browser.dom import AccessibilityNode


@dataclass
class MarkedElement:
    """An element with its Set-of-Marks number and bounding box."""
    number: int
    element_id: str
    role: str
    name: str
    bbox: tuple[int, int, int, int]  # left, top, right, bottom


def _collect_leaf_interactive(
    node: AccessibilityNode, out: list[AccessibilityNode]
) -> None:
    """Recursively collect interactive leaf nodes from the accessibility tree."""
    if not node.children:
        if node.element_id:
            out.append(node)
        return
    for child in node.children:
        _collect_leaf_interactive(child, out)


def draw_with_bboxes(
    image_path: str | Path,
    elements: list[dict],
    output_path: str | Path,
    font_size: int = 18,
) -> list[MarkedElement]:
    """Draw numbered markers using explicit bounding boxes from Playwright.

    Each element dict should have: element_id, role, name, bbox (left, top, right, bottom).
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    marked: list[MarkedElement] = []
    for idx, el in enumerate(elements, start=1):
        left, top, right, bottom = el["bbox"]
        number = idx

        # Draw red bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # Draw number label with white background
        label = str(number)
        bbox = draw.textbbox((left, top), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_h = bbox[3] - bbox[1]
        label_bg = [left, top - label_h - 4, left + label_w + 4, top]
        draw.rectangle(label_bg, fill="red")
        draw.text((left + 2, top - label_h - 2), label, fill="white", font=font)

        marked.append(
            MarkedElement(
                number=number,
                element_id=el["element_id"],
                role=el.get("role", ""),
                name=el.get("name", ""),
                bbox=el["bbox"],
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return marked


def draw_set_of_marks(
    image_path: str | Path,
    nodes: list[AccessibilityNode],
    output_path: str | Path,
    font_size: int = 18,
) -> list[MarkedElement]:
    """Draw numbered markers on interactive elements from an accessibility tree.

    Note: This assigns sequential numbers but does not have real bounding boxes
    (the accessibility tree doesn't include pixel coordinates). Use
    draw_with_bboxes() with Playwright bounding boxes for accurate marking.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    leaves: list[AccessibilityNode] = []
    for node in nodes:
        _collect_leaf_interactive(node, leaves)

    marked: list[MarkedElement] = []
    for idx, node in enumerate(leaves, start=1):
        marked.append(
            MarkedElement(
                number=idx,
                element_id=node.element_id or f"e{idx}",
                role=node.role,
                name=node.name,
                bbox=(0, 0, 0, 0),  # placeholder; real bboxes from Playwright
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return marked


def save_marked_image(
    image_bytes: bytes,
    output_path: str | Path,
) -> Path:
    """Save raw screenshot bytes to a file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(image_bytes)
    return p