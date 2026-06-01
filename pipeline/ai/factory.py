from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from .providers import (
    HttpAIAnalyzer,
    LocalCommandAIAnalyzer,
    LocalHeuristicAIAnalyzer,
    NoOpAIAnalyzer,
    OpenAICompatibleAnalyzer,
    PokeVisionProvider,
)


def build_ai_analyzer(config: PipelineConfig, overrides: dict[str, Any] | None = None):
    overrides = overrides or {}
    provider = (overrides.get("ai_provider") or config.ai_provider or os.getenv("BEACH_MAP_AI_PROVIDER", "noop")).strip().lower()
    endpoint = overrides.get("ai_endpoint") or config.ai_endpoint or os.getenv("BEACH_MAP_AI_ENDPOINT")
    model = overrides.get("ai_model") or config.ai_model or os.getenv("BEACH_MAP_AI_MODEL")
    api_key = os.getenv(overrides.get("ai_api_key_env") or config.ai_api_key_env)

    if provider in {"", "noop", "none", "off", "disabled"}:
        return NoOpAIAnalyzer()
    if provider in {"local", "heuristic", "rule", "local_heuristic"}:
        return LocalHeuristicAIAnalyzer()
    if provider in {"local_command", "command"}:
        command = overrides.get("ai_command") or os.getenv("BEACH_MAP_AI_COMMAND")
        if not command:
            raise ValueError("BEACH_MAP_AI_COMMAND is required for local_command provider")
        return LocalCommandAIAnalyzer(command=command.split())
    if provider in {"poke", "poke_vision", "vision"}:
        image_cache_dir = Path(config.cache_dir / "vision")
        return PokeVisionProvider(
            image_cache_dir=image_cache_dir,
            endpoint=endpoint or "http://127.0.0.1:11434/api/chat",
            model=model,
            api_key_env=config.ai_api_key_env,
            timeout_seconds=config.ai_timeout_seconds,
        )
    if provider in {"api", "http", "openai_compatible", "openai"}:
        if not endpoint:
            raise ValueError("BEACH_MAP_AI_ENDPOINT is required for api provider")
        return OpenAICompatibleAnalyzer(endpoint=endpoint, api_key=api_key, model=model)
    if provider in {"http_generic"}:
        if not endpoint:
            raise ValueError("BEACH_MAP_AI_ENDPOINT is required for http_generic provider")
        return HttpAIAnalyzer(endpoint=endpoint, api_key=api_key, model=model)
    raise ValueError(f"Unknown AI provider: {provider}")
