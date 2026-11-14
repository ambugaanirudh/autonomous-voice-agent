"""VLM client using Qwen2-VL-2B INT4 GGUF via llama-cpp-python.

Forces structured JSON output using a system prompt.
Runs on CPU or CUDA depending on config (n_gpu_layers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment,misc]


@dataclass
class VLMResponse:
    """Structured response from the VLM."""
    action: str           # "click", "type", "wait", "stop", etc.
    element_id: Optional[str]
    text: Optional[str]   # text to type, if action is "type"
    reasoning: str        # VLM's explanation (for debugging/logs)
    confidence: float     # 0.0 to 1.0


class VLMClient:
    def __init__(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        n_gpu_layers: int = 20,
        ctx_size: int = 4096,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.verbose = verbose
        self._model = None

        # Validate model files exist (fail fast with clear error)
        if not self.model_path.exists():
            raise FileNotFoundError(f"VLM model not found: {self.model_path.resolve()}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"VLM mmproj not found: {self.mmproj_path.resolve()}")

    def _load(self) -> None:
        """Lazy-load the llama-cpp-python model."""
        if self._model is not None:
            return
        if Llama is None:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        self._model = Llama(
            model_path=str(self.model_path),
            mmproj=str(self.mmproj_path),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.ctx_size,
            verbose=self.verbose,
        )
        logger.info(
            "VLM loaded: model={model} mmproj={mmproj} gpu_layers={layers}",
            model=self.model_path.name,
            mmproj=self.mmproj_path.name,
            layers=self.n_gpu_layers,
        )

    def ground(
        self,
        image_path: str | Path,
        command: str,
        element_map: list[dict],
    ) -> VLMResponse:
        """Ask the VLM to pick an element ID for the given command.

        Args:
            image_path: Path to the Set-of-Marks marked screenshot.
            command: User command (e.g., "open the cart").
            element_map: List of {number, element_id, role, name} for the prompt.

        Returns:
            VLMResponse with action, element_id, text, reasoning, confidence.
        """
        self._load()

        # Build the element list for the prompt
        element_lines = []
        for el in element_map:
            element_lines.append(
                f"  {el['number']}. [{el['role']}] '{el['name']}' (id: {el['element_id']})"
            )
        element_list = "\n".join(element_lines)

        # System prompt: force JSON output, Set-of-Marks grounding
        system_prompt = """You are a web navigation agent. You receive a screenshot with numbered interactive elements (Set-of-Marks).
Your job is to pick the ONE element that best matches the user's command.

Rules:
- Return ONLY valid JSON, no markdown, no extra text.
- Pick an element by its number from the list below.
- action must be one of: "click", "type", "wait", "stop".
- If action is "type", include the text to type in the "text" field.
- confidence must be 0.0 to 1.0.
- If no element matches, set action to "stop" and element_id to null.
- Never invent element IDs or numbers not in the list.

Available elements:
{element_list}

Respond with this JSON schema:
{{"action": "click|type|wait|stop", "element_id": "e1|null", "text": "string|null", "reasoning": "string", "confidence": 0.0-1.0}}"""

        user_prompt = f"Command: {command}\n\nSelect the best element:"

        try:
            response = self._model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt.format(element_list=element_list)},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": str(image_path)},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=256,
            )
        except Exception as exc:
            logger.error("VLM inference failed: {error}", error=exc)
            return VLMResponse(
                action="stop",
                element_id=None,
                text=None,
                reasoning=f"VLM error: {exc}",
                confidence=0.0,
            )

        raw_text = response["choices"][0]["message"]["content"].strip()
        logger.debug("VLM raw response: {text}", text=raw_text[:200])

        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> VLMResponse:
        """Parse and validate the VLM's JSON response."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)
            if len(cleaned) >= 3:
                cleaned = cleaned[1]
                cleaned = "\n".join(cleaned.split("\n")[1:]) if cleaned.startswith(("json", "JSON")) else cleaned
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("VLM returned invalid JSON: {error}", error=exc)
            return VLMResponse(
                action="stop",
                element_id=None,
                text=None,
                reasoning=f"Invalid JSON: {raw_text[:100]}",
                confidence=0.0,
            )

        return VLMResponse(
            action=str(data.get("action", "stop")).lower(),
            element_id=data.get("element_id"),
            text=data.get("text"),
            reasoning=str(data.get("reasoning", "")),
            confidence=float(data.get("confidence", 0.0)),
        )