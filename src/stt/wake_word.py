"""Wake word detection via transcription keyword spotting.

Transcribes audio buffers and checks for the configured wake word.
No separate wake-word model is needed; this keeps the dependency footprint
small and runs entirely on CPU.
"""

from __future__ import annotations

from typing import Optional


class WakeWordDetector:
    def __init__(self, wake_word: str = "computer") -> None:
        self.wake_word = wake_word.lower().strip()
        if not self.wake_word:
            raise ValueError("wake_word must not be empty")

    def contains_wake_word(self, text: str) -> bool:
        """Return True if the wake word appears anywhere in the text."""
        return self.wake_word in text.lower()

    def extract_command(self, text: str) -> Optional[str]:
        """Return the command text after the wake word, or None if not found.

        Example:
            wake_word="computer", text="computer open the cart"
            returns "open the cart"
        """
        text_lower = text.lower()
        idx = text_lower.find(self.wake_word)
        if idx == -1:
            return None
        command = text[idx + len(self.wake_word) :].strip()
        return command if command else None