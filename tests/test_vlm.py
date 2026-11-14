"""Unit tests for VLM client structure (model download skipped)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from PIL import Image
from src.vision.vlm import VLMClient, VLMResponse


@patch("src.vision.vlm.Llama", create=True)
def test_vlm_client_init(mock_llama: MagicMock, tmp_path) -> None:
    model_file = tmp_path / "model.gguf"
    mmproj_file = tmp_path / "mmproj.gguf"
    model_file.write_bytes(b"fake")
    mmproj_file.write_bytes(b"fake")

    client = VLMClient(
        model_path=model_file,
        mmproj_path=mmproj_file,
        n_gpu_layers=20,
        ctx_size=4096,
    )
    assert client.n_gpu_layers == 20
    assert client.ctx_size == 4096
    assert client._model is None  # lazy load


@patch("src.vision.vlm.Llama", create=True)
def test_vlm_client_missing_model_raises(tmp_path) -> None:
    missing = tmp_path / "nonexistent.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"fake")

    with pytest.raises(FileNotFoundError):
        VLMClient(model_path=missing, mmproj_path=mmproj)


@patch("src.vision.vlm.Llama", create=True)
def test_ground_parses_valid_json(mock_llama: MagicMock, tmp_path) -> None:
    model_file = tmp_path / "model.gguf"
    mmproj_file = tmp_path / "mmproj.gguf"
    model_file.write_bytes(b"fake")
    mmproj_file.write_bytes(b"fake")

    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"action": "click", "element_id": "e3", "text": null, "reasoning": "Buy button", "confidence": 0.95}'
                }
            }
        ]
    }
    mock_llama.return_value = mock_instance

    client = VLMClient(model_path=model_file, mmproj_path=mmproj_file)
    client._model = mock_instance  # skip lazy load

    image_path = tmp_path / "marked.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    element_map = [
        {"number": 1, "element_id": "e1", "role": "button", "name": "Home"},
        {"number": 3, "element_id": "e3", "role": "button", "name": "Buy"},
    ]

    response = client.ground(image_path, "open the cart", element_map)
    assert response.action == "click"
    assert response.element_id == "e3"
    assert response.confidence == 0.95
    assert "Buy button" in response.reasoning


@patch("src.vision.vlm.Llama", create=True)
def test_ground_invalid_json_returns_stop(mock_llama: MagicMock, tmp_path) -> None:
    model_file = tmp_path / "model.gguf"
    mmproj_file = tmp_path / "mmproj.gguf"
    model_file.write_bytes(b"fake")
    mmproj_file.write_bytes(b"fake")

    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "not json at all"}}]
    }
    mock_llama.return_value = mock_instance

    client = VLMClient(model_path=model_file, mmproj_path=mmproj_file)
    client._model = mock_instance

    image_path = tmp_path / "marked.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    response = client.ground(image_path, "do something", [])
    assert response.action == "stop"
    assert response.element_id is None
    assert response.confidence == 0.0