"""Unit tests for Transcriber structure (model download skipped in CI)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.stt.transcriber import Transcriber


@patch("src.stt.transcriber.WhisperModel")
def test_transcriber_init(mock_model: MagicMock) -> None:
    t = Transcriber(model_size="small", language="en", device="cpu", compute_type="int8")
    assert t.language == "en"
    assert t.device == "cpu"
    assert t.compute_type == "int8"
    mock_model.assert_called_once_with("small", device="cpu", compute_type="int8")


@patch("src.stt.transcriber.WhisperModel")
def test_transcribe_calls_model(mock_model: MagicMock, tmp_path) -> None:
    mock_instance = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_instance.transcribe.return_value = ([mock_segment], MagicMock())
    mock_model.return_value = mock_instance

    t = Transcriber()
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake")

    result = t.transcribe(str(audio_file))
    assert result == "hello world"
    mock_instance.transcribe.assert_called_once()