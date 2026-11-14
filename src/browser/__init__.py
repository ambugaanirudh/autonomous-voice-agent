"""Browser package: Playwright wrapper + accessibility tree extraction."""
from __future__ import annotations

from src.browser.controller import BrowserController, BrowserConfig
from src.browser.dom import AccessibilityTree, extract_accessibility_tree

__all__ = ["BrowserController", "BrowserConfig", "AccessibilityTree", "extract_accessibility_tree"]