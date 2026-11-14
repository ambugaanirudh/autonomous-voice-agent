"""Unit tests for Set-of-Marks overlay."""
from __future__ import annotations

from PIL import Image
import pytest

from src.vision.som import draw_with_bboxes, MarkedElement


def test_draw_with_bboxes_creates_marked_elements(tmp_path) -> None:
    # Create a simple test image
    img_path = tmp_path / "screenshot.png"
    Image.new("RGB", (100, 100), color="white").save(img_path)

    elements = [
        {"element_id": "e1", "role": "button", "name": "Buy", "bbox": (10, 10, 50, 40)},
        {"element_id": "e2", "role": "link", "name": "Home", "bbox": (60, 60, 90, 80)},
    ]
    out_path = tmp_path / "marked.png"
    marked = draw_with_bboxes(img_path, elements, out_path)

    assert len(marked) == 2
    assert marked[0].number == 1
    assert marked[0].element_id == "e1"
    assert marked[1].number == 2
    assert marked[1].element_id == "e2"
    assert out_path.exists()


def test_marked_element_dataclass() -> None:
    me = MarkedElement(
        number=5,
        element_id="e42",
        role="button",
        name="Submit",
        bbox=(100, 200, 200, 250),
    )
    assert me.number == 5
    assert me.element_id == "e42"
    assert me.role == "button"
    assert me.name == "Submit"
    assert me.bbox == (100, 200, 200, 250)