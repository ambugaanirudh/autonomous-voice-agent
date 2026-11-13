"""Application configuration loader.

Reads config.yaml and validates it with Pydantic. All paths are resolved
relative to the project root; there are no hardcoded absolute paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml


class BrowserConfig(BaseModel):
    headless: bool = False
    viewport_width: int = 1280
    viewport_height: int = 800
    timeout_ms: int = 30000


class SafetyConfig(BaseModel):
    max_steps: int = 50
    allowlist_domains: list[str] = Field(default_factory=list)
    require_confirmation_for: list[str] = Field(default_factory=list)


class AppSection(BaseModel):
    name: str
    version: str
    log_level: str = "INFO"


class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    sites: dict[str, Any] = Field(default_factory=dict)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    stt: dict[str, Any] = Field(default_factory=dict)
    vision: dict[str, Any] = Field(default_factory=dict)
    tts: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        """Load and validate configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p.resolve()}")
        with p.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**raw)