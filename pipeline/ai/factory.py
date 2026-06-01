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


def _resolved_provider(config: PipelineConfig, overrides: dict[str, Any]) -> str:
    return (
        overrides.get("ai_provider")
        or config.ai_provider
        or os.getenv("BEACH_MAP_AI_PROVIDER", "noop")
    ).strip().lower()


def build_ai_analyzer(config: PipelineConfig, overrides: dict[str, Any] | None = None):
    overrides = overrides or {}
    provider = _resolved_provider(config, overrides)
    endpoint = overrides.get("ai_endpoint") or config.ai_endpoint or os.getenv("BEACH_MAP_AI_ENDPOINT")
    model = overrides.get("ai_model") or config.ai_model or os.getenv("BEACH_MAP_AI_MODEL")
    api_key_env = overrides.get("ai_api_key_env") or config.ai_api_key_env
    api_key = os.getenv(api_key_env)

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
            api_key_env=api_key_env,
            timeout_seconds=config.ai_timeout_seconds,
            max_retries=config.ai_max_retries,
            retry_backoff_seconds=config.ai_retry_backoff_seconds,
            max_output_tokens=config.ai_max_output_tokens,
            image_width=config.vision_image_width,
            image_height=config.vision_image_height,
        )

    if provider in {"openai_compatible", "openai", "api"}:
        if not endpoint:
            raise ValueError("BEACH_MAP_AI_ENDPOINT is required for openai_compatible provider")
        if provider == "openai" and endpoint and "generativelanguage.googleapis.com" in endpoint:
            raise ValueError("Ambiguous provider config: provider=openai with Gemini endpoint. Use provider=poke_vision or set matching endpoint.")
        return OpenAICompatibleAnalyzer(endpoint=endpoint, api_key=api_key, model=model)

    if provider in {"http", "http_generic"}:
        if not endpoint:
            raise ValueError("BEACH_MAP_AI_ENDPOINT is required for http provider")
        return HttpAIAnalyzer(endpoint=endpoint, api_key=api_key, model=model)

    raise ValueError(f"Unknown AI provider: {provider}")
