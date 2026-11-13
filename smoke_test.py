"""Phase 0 smoke test.

Verifies:
  1. Python version is 3.11+
  2. config.yaml loads and validates via Pydantic
  3. State object saves and loads correctly
  4. Playwright can launch Chromium and navigate to the mock store

Prerequisites:
  - Mock site server running in another terminal:  python scripts/serve_sites.py
"""

from __future__ import annotations

import os
import sys
import tempfile


def check_python() -> bool:
    print(f"Python version: {sys.version}")
    ok = sys.version_info >= (3, 11)
    print(f"  Python >= 3.11: {'OK' if ok else 'FAIL'}")
    return ok


def check_config() -> bool:
    from src.config import AppConfig

    cfg = AppConfig.from_file("config.yaml")
    print(f"Config loaded: {cfg.app.name} v{cfg.app.version}")
    print(f"  Log level: {cfg.app.log_level}")
    print(f"  Store URL: {cfg.sites.get('mock_store_url')}")
    print(f"  Dashboard URL: {cfg.sites.get('mock_dashboard_url')}")
    print(f"  Max steps: {cfg.safety.max_steps}")
    print(f"  Allowlist: {cfg.safety.allowlist_domains}")
    print("  Config validation: OK")
    return True


def check_state() -> bool:
    from src.state import State, StepRecord

    s = State(goal="Test goal", plan=["step one", "step two"], max_retries=5)
    s.completed_steps.append(
        StepRecord(
            step=1,
            action="click",
            target="button#buy",
            success=True,
            detail="Clicked buy button",
            timestamp="2024-01-01T00:00:00Z",
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        s.save(tmp_path)
        loaded = State.load(tmp_path)
        assert loaded.goal == "Test goal"
        assert loaded.plan == ["step one", "step two"]
        assert loaded.max_retries == 5
        assert len(loaded.completed_steps) == 1
        assert loaded.completed_steps[0].success is True
        print("State save/load roundtrip: OK")
        return True
    finally:
        os.unlink(tmp_path)


def check_playwright() -> bool:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto("http://127.0.0.1:8765/store/index.html", timeout=10000)
            title = page.title()
            print(f"Playwright reached mock store. Title: {title!r}")
            assert "Mock Store" in title
            print("Playwright navigation: OK")
            return True
        except Exception as exc:
            print(f"Playwright navigation failed: {exc}")
            print("  (Is the mock site server running? Start it with: python scripts/serve_sites.py)")
            return False
        finally:
            browser.close()


def main() -> int:
    print("=" * 50)
    print("Phase 0 Smoke Test")
    print("=" * 50)

    results = []
    results.append(("Python version", check_python()))
    results.append(("Config load", check_config()))
    results.append(("State roundtrip", check_state()))
    results.append(("Playwright navigation", check_playwright()))

    print("\n" + "=" * 50)
    all_ok = True
    for name, ok in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'}")
        all_ok = all_ok and ok
    print("=" * 50)
    print(f"Overall: {'PASSED' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())