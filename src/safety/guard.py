"""Safety guard: domain allowlist, max steps, confirmation gating.

Treats all page content as untrusted (prompt-injection protection).
Requires spoken/typed confirmation before irreversible actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class SafetyVerdict:
    allowed: bool
    reason: str
    requires_confirmation: bool = False


class SafetyGuard:
    def __init__(
        self,
        allowlist_domains: list[str],
        max_steps: int,
        require_confirmation_for: list[str],
    ) -> None:
        self.allowlist_domains = [d.lower() for d in allowlist_domains]
        self.max_steps = max_steps
        self.require_confirmation_for = [a.lower() for a in require_confirmation_for]
        self.steps_taken = 0

    def check_domain(self, url: str) -> SafetyVerdict:
        """Verify the URL domain is on the allowlist."""
        try:
            domain = urlparse(url).hostname or ""
        except Exception:
            return SafetyVerdict(
                allowed=False, reason="Cannot parse URL domain", requires_confirmation=False
            )

        domain_lower = domain.lower()
        for allowed in self.allowlist_domains:
            if domain_lower == allowed or domain_lower.endswith("." + allowed):
                return SafetyVerdict(
                    allowed=True, reason=f"Domain {domain} is allowlisted",
                    requires_confirmation=False,
                )

        return SafetyVerdict(
            allowed=False,
            reason=f"Domain {domain} is NOT on allowlist: {self.allowlist_domains}",
            requires_confirmation=False,
        )

    def check_step_limit(self) -> SafetyVerdict:
        """Check if we've exceeded the max steps limit."""
        if self.steps_taken >= self.max_steps:
            return SafetyVerdict(
                allowed=False,
                reason=f"Max steps ({self.max_steps}) reached — stopping to prevent runaway loop",
                requires_confirmation=False,
            )
        return SafetyVerdict(
            allowed=True, reason=f"Step {self.steps_taken + 1}/{self.max_steps}",
            requires_confirmation=False,
        )

    def increment_step(self) -> None:
        self.steps_taken += 1

    def requires_confirmation(self, action: str) -> bool:
        """Check if an action requires user confirmation before execution."""
        return action.lower() in self.require_confirmation_for

    def confirm_action(self, action: str, detail: str) -> bool:
        """Prompt the user for confirmation before an irreversible action.

        In a voice interface, this would be spoken; here we use input().
        Returns True if confirmed, False if cancelled.
        """
        print(f"\n{'=' * 50}")
        print(f"SAFETY CONFIRMATION REQUIRED")
        print(f"Action: {action}")
        print(f"Detail: {detail}")
        print(f"This is an irreversible action.")
        print(f"{'=' * 50}")
        response = input("Type 'yes' to confirm, anything else to cancel: ").strip().lower()
        return response == "yes"

    def reset(self) -> None:
        self.steps_taken = 0