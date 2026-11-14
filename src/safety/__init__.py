"""Safety package: domain allowlist, max steps, confirmation gating."""
from __future__ import annotations

from src.safety.guard import SafetyGuard, SafetyVerdict

__all__ = ["SafetyGuard", "SafetyVerdict"]