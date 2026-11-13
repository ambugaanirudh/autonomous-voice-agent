"""Unit tests for wake word detection."""
from __future__ import annotations

import pytest

from src.stt.wake_word import WakeWordDetector


def test_contains_wake_word() -> None:
    wwd = WakeWordDetector(wake_word="computer")
    assert wwd.contains_wake_word("computer open the cart") is True
    assert wwd.contains_wake_word("Open the computer please") is True
    assert wwd.contains_wake_word("open the cart") is False
    assert wwd.contains_wake_word("Computer") is True  # case insensitive


def test_extract_command() -> None:
    wwd = WakeWordDetector(wake_word="computer")
    assert wwd.extract_command("computer open the cart") == "open the cart"
    assert wwd.extract_command("computer") is None  # wake word only
    assert wwd.extract_command("open the cart") is None  # no wake word
    assert wwd.extract_command("") is None


def test_different_wake_word() -> None:
    wwd = WakeWordDetector(wake_word="hey assistant")
    assert wwd.contains_wake_word("hey assistant play music") is True
    assert wwd.extract_command("hey assistant play music") == "play music"


def test_empty_wake_word_rejected() -> None:
    with pytest.raises(ValueError):
        WakeWordDetector(wake_word="")