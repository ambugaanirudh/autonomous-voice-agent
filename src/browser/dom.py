"""Accessibility tree extraction from Playwright pages.

Extracts ARIA roles, names, and IDs for interactive elements.
This is the primary perception layer — preferred over VLM grounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AccessibilityTree:
    """The full accessibility tree for a page."""
    nodes: list["AccessibilityNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes]}


@dataclass
class AccessibilityNode:
    """A single node in the accessibility tree."""
    role: str
    name: str
    element_id: Optional[str] = None
    css_selector: Optional[str] = None
    children: list["AccessibilityNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "name": self.name,
            "element_id": self.element_id,
            "css_selector": self.css_selector,
            "children": [c.to_dict() for c in self.children],
        }


def extract_accessibility_tree(page) -> list[AccessibilityNode]:
    """Extract the accessibility tree from a Playwright page.

    Uses page.accessibility.snapshot() to get the AX tree, then filters
    to interactive elements (buttons, links, inputs, etc.) and assigns
    sequential element IDs for Set-of-Marks grounding.

    Returns:
        List of top-level AccessibilityNode objects.
    """
    snapshot = page.accessibility.snapshot()
    if snapshot is None:
        return []

    interactive_roles = {
        "button", "link", "textbox", "combobox", "checkbox",
        "radio", "listbox", "menuitem", "tab", "slider",
        "searchbox", "spinbutton",
    }

    nodes: list[AccessibilityNode] = []
    counter = [0]  # mutable counter for sequential IDs

    def build_node(ax_node: dict) -> Optional[AccessibilityNode]:
        role = ax_node.get("role", "").lower()
        name = ax_node.get("name") or ""

        is_interactive = role in interactive_roles
        has_name = bool(name.strip())

        if not is_interactive and not has_name:
            # Still recurse into children
            children = []
            for child in ax_node.get("children", []) or []:
                child_node = build_node(child)
                if child_node:
                    children.append(child_node)
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            return AccessibilityNode(role="container", name="", children=children)

        if is_interactive or has_name:
            counter[0] += 1
            element_id = f"e{counter[0]}"
        else:
            element_id = None

        selector_parts = []
        ax_role = ax_node.get("role", "")
        if ax_role:
            selector_parts.append(f'[role="{ax_role}"]')
        if name:
            selector_parts.append(f'[aria-label="{name}"]')
        css_selector = ", ".join(selector_parts) if selector_parts else None

        node = AccessibilityNode(
            role=role,
            name=name,
            element_id=element_id,
            css_selector=css_selector,
        )

        for child in ax_node.get("children", []) or []:
            child_node = build_node(child)
            if child_node:
                node.children.append(child_node)

        return node

    for child in snapshot.get("children", []) or []:
        node = build_node(child)
        if node:
            nodes.append(node)

    return nodes


def find_node_by_id(nodes: list[AccessibilityNode], element_id: str) -> Optional[AccessibilityNode]:
    """Find a node by its element ID (e.g., "e42")."""
    for node in nodes:
        found = _search(node, element_id)
        if found:
            return found
    return None


def _search(node: AccessibilityNode, element_id: str) -> Optional[AccessibilityNode]:
    if node.element_id == element_id:
        return node
    for child in node.children:
        found = _search(child, element_id)
        if found:
            return found
    return None


def validate_element_id(nodes: list[AccessibilityNode], element_id: str) -> bool:
    """Return True if element_id exists in the accessibility tree (anti-hallucination guard)."""
    return find_node_by_id(nodes, element_id) is not None