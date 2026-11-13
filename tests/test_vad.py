"""Unit tests for VAD."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from src.stt.vad import VAD


@pytest.fixture
def vad() -> VAD:
    return VAD(aggressiveness=2, sample_rate=16000, frame_duration_ms=30)


def test_vad_frame_size(vad: VAD) -> None:
    # 30ms at 16kHz = 480 samples = 960 bytes int16
    assert vad.frame_size == 480


def test_vad_float_to_pcm16(vad: VAD) -> None:
    audio = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    pcm = vad.float_to_pcm16(audio)
    assert len(pcm) == 4 * 2  # 4 samples * 2 bytes
    values = struct.unpack("<4h", pcm)
    assert values[1] == 32767
    assert values[2] == -32767


def test_vad_is_speech_rejects_wrong_size(vad: VAD) -> None:
    with pytest.raises(ValueError):
        vad.is_speech(b"\x00\x00")  # too short for a 30ms frame


def test_vad_invalid_frame_duration() -> None:
    with pytest.raises(ValueError):
        VAD(frame_duration_ms=15)  # not 10, 20, or 30