"""Playwright browser controller.

Handles browser lifecycle, navigation, screenshots, clicks, and typing.
All methods are synchronous for simplicity and debuggability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext


@dataclass
class BrowserConfig:
    headless: bool = False
    viewport_width: int = 1280
    viewport_height: int = 800
    timeout_ms: int = 30000


class BrowserController:
    def __init__(self, config: BrowserConfig) -> None:
        self.config = config
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    def start(self) -> None:
        """Launch Chromium and open a new page."""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.config.headless,
        )
        self._context = self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            }
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.config.timeout_ms)
        logger.info("Browser started. Headless: {headless}", headless=self.config.headless)

    def goto(self, url: str) -> str:
        """Navigate to a URL. Returns the final URL after redirects."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        self.logger.info("Navigating to {url}", url=url)
        response = self._page.goto(url, wait_until="domcontentloaded")
        final_url = self._page.url
        self.logger.info(
            "Navigated to {final_url} (status: {status})",
            final_url=final_url,
            status=response.status if response else None,
        )
        return final_url

    def screenshot(self, path: str | Path) -> bytes:
        """Take a screenshot and save to path. Returns raw bytes."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bytes_data = self._page.screenshot(path=str(p), full_page=True)
        self.logger.debug("Screenshot saved to {path}", path=p.resolve())
        return bytes_data

    def click(self, selector: str) -> None:
        """Click the element matching the CSS selector."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        self.logger.debug("Clicking {selector}", selector=selector)
        self._page.click(selector)

    def type_text(self, selector: str, text: str) -> None:
        """Type text into the element matching the CSS selector."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        self.logger.debug("Typing into {selector}", selector=selector)
        self._page.fill(selector, text)

    def wait_for_selector(self, selector: str, timeout_ms: Optional[int] = None) -> None:
        """Wait for an element to appear in the DOM."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        self._page.wait_for_selector(selector, timeout=timeout_ms or self.config.timeout_ms)

    def get_url(self) -> Optional[str]:
        """Return the current page URL."""
        if self._page is None:
            return None
        return self._page.url

    def get_title(self) -> Optional[str]:
        """Return the current page title."""
        if self._page is None:
            return None
        return self._page.title()

    def evaluate(self, script: str) -> object:
        """Run JavaScript in the page context and return the result."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page.evaluate(script)

    def close(self) -> None:
        """Close the browser and release resources."""
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        logger.info("Browser closed")

    def __enter__(self) -> "BrowserController":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()