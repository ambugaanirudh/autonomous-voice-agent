"""Manual STT test script.

Usage:
    .venv\Scripts\python.exe scripts/manual_stt_test.py

Speak after the wake word (default: "computer"). Say e.g. "computer open the cart".
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `src` package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stt.listener import Listener


def main() -> None:
    listener = Listener(wake_word="computer", model_size="tiny")
    print(f"Listening (wake word: '{listener.wake_word.wake_word}')...")
    print("Say something like: computer open the cart")
    print("Waiting up to 30 seconds...")
    command = listener.listen_once(timeout_sec=30)
    print(f"Result: {command!r}")


if __name__ == "__main__":
    main()