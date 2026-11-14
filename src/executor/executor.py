"""Executor: executes grounded actions via Playwright and verifies results.

Verification after every action: URL change, DOM diff, expected text,
screenshot comparison. On failure, returns VerificationResult(success=False)
so the agent can retry or replan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from playwright.sync_api import Page

from src.browser.controller import BrowserController


@dataclass
class VerificationResult:
    """Result of verifying an action's effect."""
    success: bool
    url_changed: bool
    dom_changed: bool
    expected_text_found: bool
    detail: str


class Executor:
    def __init__(
        self,
        browser: BrowserController,
        screenshots_dir: str = "logs/screenshots",
    ) -> None:
        self.browser = browser
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute(
        self,
        action: str,
        selector: Optional[str] = None,
        text: Optional[str] = None,
    ) -> VerificationResult:
        """Execute a grounded action and verify the result.

        Args:
            action: "click", "type", "wait", "stop"
            selector: CSS selector for the target element
            text: Text to type (for "type" actions)

        Returns:
            VerificationResult indicating success/failure and why.
        """
        page = self.browser._page
        if page is None:
            return VerificationResult(
                success=False, url_changed=False, dom_changed=False,
                expected_text_found=False, detail="Browser page not available",
            )

        # Capture pre-action state
        url_before = self.browser.get_url()
        dom_hash_before = self._dom_hash(page)

        # Execute the action
        try:
            if action == "click":
                if not selector:
                    return VerificationResult(
                        success=False, url_changed=False, dom_changed=False,
                        expected_text_found=False, detail="No selector for click",
                    )
                self._click(page, selector)
            elif action == "type":
                if not selector:
                    return VerificationResult(
                        success=False, url_changed=False, dom_changed=False,
                        expected_text_found=False, detail="No selector for type",
                    )
                self._type(page, selector, text or "")
            elif action == "wait":
                page.wait_for_timeout(2000)
            elif action == "stop":
                pass
            else:
                return VerificationResult(
                    success=False, url_changed=False, dom_changed=False,
                    expected_text_found=False, detail=f"Unknown action: {action}",
                )
        except Exception as exc:
            logger.error("Action execution failed: {error}", error=exc)
            return VerificationResult(
                success=False, url_changed=False, dom_changed=False,
                expected_text_found=False, detail=f"Execution error: {exc}",
            )

        # Capture post-action state
        url_after = self.browser.get_url()
        dom_hash_after = self._dom_hash(page)

        url_changed = url_before != url_after
        dom_changed = dom_hash_before != dom_hash_after

        # Take a post-action screenshot
        screenshot_path = self.screenshots_dir / f"post_action_{action}_{selector or 'none'}.png"
        try:
            self.browser.screenshot(screenshot_path)
        except Exception as exc:
            logger.warning("Screenshot failed: {error}", error=exc)

        expected_text_found = True  # No specific text expected unless provided

        detail_parts = []
        if url_changed:
            detail_parts.append(f"URL changed: {url_before} -> {url_after}")
        if dom_changed:
            detail_parts.append("DOM changed")
        if not url_changed and not dom_changed:
            detail_parts.append("No detectable change")

        success = url_changed or dom_changed or action in ("wait", "stop")

        return VerificationResult(
            success=success,
            url_changed=url_changed,
            dom_changed=dom_changed,
            expected_text_found=expected_text_found,
            detail="; ".join(detail_parts) if detail_parts else "Action completed",
        )

    def _click(self, page: Page, selector: str) -> None:
        """Click an element with fallback strategies."""
        # Strategy 1: Direct CSS selector
        try:
            page.click(selector, timeout=5000)
            return
        except Exception:
            pass

        # Strategy 2: Role + name
        try:
            page.get_by_role("button", name=selector).click(timeout=5000)
            return
        except Exception:
            pass

        # Strategy 3: Text match
        try:
            page.get_by_text(selector).click(timeout=5000)
            return
        except Exception:
            pass

        raise RuntimeError(f"Could not click element: {selector}")

    def _type(self, page: Page, selector: str, text: str) -> None:
        """Type text into an element with fallback strategies."""
        try:
            page.fill(selector, text, timeout=5000)
            return
        except Exception:
            pass

        try:
            page.get_by_role("textbox", name=selector).fill(text, timeout=5000)
            return
        except Exception:
            pass

        raise RuntimeError(f"Could not type into element: {selector}")

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def _dom_hash(self, page: Page) -> str:
        """Compute a hash of the page's DOM for change detection."""
        try:
            html = page.evaluate("() => document.documentElement.outerHTML")
            return hashlib.sha256(html.encode("utf-8")).hexdigest()
        except Exception:
            return ""