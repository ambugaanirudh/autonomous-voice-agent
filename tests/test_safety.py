"""Unit tests for the SafetyGuard."""
from __future__ import annotations

from src.safety.guard import SafetyGuard, SafetyVerdict


def test_check_domain_allowlisted() -> None:
    guard = SafetyGuard(
        allowlist_domains=["127.0.0.1", "localhost"],
        max_steps=50,
        require_confirmation_for=["purchase"],
    )
    v = guard.check_domain("http://127.0.0.1:8765/store/index.html")
    assert v.allowed is True


def test_check_domain_not_allowlisted() -> None:
    guard = SafetyGuard(
        allowlist_domains=["127.0.0.1"],
        max_steps=50,
        require_confirmation_for=["purchase"],
    )
    v = guard.check_domain("http://evil-site.com")
    assert v.allowed is False
    assert "NOT on allowlist" in v.reason


def test_check_step_limit_ok() -> None:
    guard = SafetyGuard(
        allowlist_domains=["127.0.0.1"],
        max_steps=50,
        require_confirmation_for=["purchase"],
    )
    v = guard.check_step_limit()
    assert v.allowed is True


def test_check_step_limit_exceeded() -> None:
    guard = SafetyGuard(
        allowlist_domains=["127.0.0.1"],
        max_steps=3,
        require_confirmation_for=["purchase"],
    )
    guard.steps_taken = 3
    v = guard.check_step_limit()
    assert v.allowed is False
    assert "Max steps" in v.reason


def test_requires_confirmation() -> None:
    guard = SafetyGuard(
        allowlist_domains=["127.0.0.1"],
        max_steps=50,
        require_confirmation_for=["purchase", "delete"],
    )
    assert guard.requires_confirmation("purchase") is True
    assert guard.requires_confirmation("delete") is True
    assert guard.requires_confirmation("click") is False