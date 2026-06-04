from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from ..vision.imagery import download_segment_satellite_image
from .base import AIClassification

PokeVision_LLM_ENDPOINT = "http://127.0.0.1:11434/api/chat"
PokeVision_LLM_MODEL = "qwen2.5:7b"
GEMINI_VISION_MODEL = "gemini-2.5-flash"
GEMINI_VISION_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
VISION_LABELS = ("rock", "pebble", "cobble", "sand", "built", "unknown")
GEMINI_USAGE_LOG_FILENAME = "gemini_usage.log"


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _gemini_usage_log_path() -> Path:
    return Path(__file__).resolve().parents[2] / GEMINI_USAGE_LOG_FILENAME


def _usage_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _normalize_gemini_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _usage_int(usage, "promptTokenCount", "prompt_tokens", "promptTokens")
    completion_tokens = _usage_int(usage, "candidatesTokenCount", "completion_tokens", "completionTokens", "candidateTokenCount")
    total_tokens = _usage_int(usage, "totalTokenCount", "total_tokens", "totalTokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}


def _extract_gemini_usage(data: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any] | None:
    usage = _normalize_gemini_usage(data.get("usageMetadata") or data.get("usage_metadata") or data.get("usage"))
    if usage:
        usage["source"] = "response.usageMetadata"
        return usage
    if isinstance(parsed, dict):
        usage = _normalize_gemini_usage(parsed.get("usage") or parsed.get("usageMetadata") or parsed.get("token_usage"))
        if usage:
            usage["source"] = "parsed.response"
            return usage
    return None


def _append_gemini_usage_log(*, segment_id: str | None, model: str | None, endpoint: str, usage: dict[str, Any], response: dict[str, Any], parsed: dict[str, Any]) -> None:
    path = _gemini_usage_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cumulative = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    previous = json.loads(line)
                except Exception:
                    continue
                previous_cumulative = previous.get("cumulative")
                if isinstance(previous_cumulative, dict):
                    for key in cumulative:
                        value = previous_cumulative.get(key)
                        if value is not None:
                            try:
                                cumulative[key] = int(value)
                            except Exception:
                                pass
                else:
                    previous_usage = previous.get("usage")
                    if isinstance(previous_usage, dict):
                        for key in cumulative:
                            value = previous_usage.get(key)
                            if value is not None:
                                try:
                                    cumulative[key] += int(value)
                                except Exception:
                                    pass
        except Exception:
            pass
    for key in cumulative:
        value = usage.get(key)
        if value is not None:
            try:
                cumulative[key] += int(value)
            except Exception:
                pass
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "segment_id": segment_id,
        "model": model,
        "endpoint": endpoint,
        "usage_source": usage.get("source"),
        "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "cumulative": cumulative,
        "finish_reason": (response.get("candidates") or [{}])[0].get("finishReason") if isinstance(response, dict) else None,
        "parsed_label": parsed.get("label") if isinstance(parsed, dict) else None,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_json_payload(content: str):
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    first_curly = text.find("{")
    first_square = text.find("[")
    if first_curly == -1 and first_square == -1:
        return json.loads(text)
    if first_square != -1 and (first_curly == -1 or first_square < first_curly):
        start = first_square
        end = text.rfind("]")
    else:
        start = first_curly
        end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _segment_payload(segment: dict[str, Any]) -> dict[str, Any]:
    geom = segment.get("geometry")
    payload = {
        "segment_id": segment.get("segment_id"),
        "surface_class": segment.get("surface_class"),
        "matched_beach_osm_id": segment.get("matched_beach_osm_id"),
        "source_osm_id": segment.get("source_osm_id"),
        "needs_ai": segment.get("needs_ai"),
        "segment_length_m": segment.get("segment_length_m"),
        "pipeline_version": segment.get("pipeline_version"),
        "ai_vision_version": segment.get("ai_vision_version"),
    }
    if geom is not None and not geom.is_empty:
        centroid = geom.centroid
        payload["centroid_lon"] = float(centroid.x)
        payload["centroid_lat"] = float(centroid.y)
        payload["bounds"] = list(map(float, geom.bounds))
    return payload


def _parse_llm_json(text: str) -> dict[str, Any]:
    def _unwrap(value: Any) -> Any:
        current = value
        for _ in range(3):
            if isinstance(current, dict):
                return current
            if not isinstance(current, str):
                return current
            candidate = current.strip()
            if candidate.startswith("```"):
                lines = candidate.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                candidate = "\n".join(lines).strip()
            try:
                current = json.loads(candidate)
            except Exception:
                return current
        return current

    try:
        parsed = _unwrap(_extract_json_payload(text))
    except Exception:
        return {"label": None, "confidence": None, "rationale": text or "invalid json from model"}
    if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list) and parsed["results"]:
        parsed = _unwrap(parsed["results"][0])
    if isinstance(parsed, str):
        parsed = _unwrap(parsed)
    if not isinstance(parsed, dict):
        return {"label": None, "confidence": None, "rationale": text or "invalid json from model"}
    return parsed


def _normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        label = str(label)
    value = label.strip().lower()
    return value if value in VISION_LABELS else None




def _validate_structured_output(parsed: dict[str, Any]) -> dict[str, Any]:
    label = _normalize_label(parsed.get("label"))
    confidence = parsed.get("confidence")
    rationale = parsed.get("rationale")
    if label is None or confidence is None or rationale in (None, ""):
        return {"label": "unknown", "confidence": 0.0, "rationale": "invalid structured output from model"}
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(rationale)[:1024]
    return {"label": label, "confidence": confidence, "rationale": rationale}


def _post_with_retry(url: str, *, json_payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 120.0, max_retries: int = 2, retry_backoff_seconds: float = 1.5):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()
            response = requests.post(url, json=json_payload, headers=headers, timeout=timeout)
            latency_ms = int((time.perf_counter() - started) * 1000)
            response.raise_for_status()
            return response, attempt, latency_ms
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(retry_backoff_seconds * (2 ** attempt))
    raise last_exc

def _label_from_scores(scores: dict[str, float]) -> tuple[str, float]:
    label = max(scores, key=scores.get)
    total = sum(max(v, 0.0) for v in scores.values())
    confidence = 0.55 if total <= 0 else min(0.98, 0.45 + scores[label] / (total + 1e-9) * 0.5)
    return label, confidence


def _heuristic_image_classification(image_path: Path, segment: dict[str, Any]) -> AIClassification:
    try:
        from PIL import Image
    except Exception as exc:
        return AIClassification(
            label=None,
            confidence=None,
            rationale=f"pillow unavailable: {exc}",
            provider="poke_vision",
            raw={"image_path": str(image_path), "segment_id": segment.get("segment_id")},
        )

    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB").resize((128, 128))
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
    except Exception as exc:
        return AIClassification(
            label=None,
            confidence=None,
            rationale=f"unable to inspect image: {exc}",
            provider="poke_vision",
            raw={"image_path": str(image_path), "segment_id": segment.get("segment_id")},
        )

    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    warm = float((arr[:, :, 0] - arr[:, :, 2]).mean())
    green = float(arr[:, :, 1].mean())
    edge_x = float(np.abs(np.diff(gray, axis=1)).mean())
    edge_y = float(np.abs(np.diff(gray, axis=0)).mean())
    edge = (edge_x + edge_y) / 2.0
    texture = float(np.abs(gray - gray.mean()).mean())

    scores = {
        "sand": 1.50 * brightness + 0.60 * warm + 0.25 * green - 0.95 * edge - 0.25 * contrast,
        "pebble": 0.45 * brightness + 0.65 * texture + 0.80 * edge + 0.20 * (1.0 - saturation),
        "rock": 1.10 * contrast + 1.05 * edge + 0.40 * (1.0 - brightness) + 0.20 * (1.0 - warm),
        "built": 0.95 * edge + 0.80 * (1.0 - saturation) + 0.45 * contrast + 0.20 * (1.0 - green),
    }
    label, confidence = _label_from_scores(scores)
    rationale = (
        f"heuristic image stats brightness={brightness:.3f}, contrast={contrast:.3f}, saturation={saturation:.3f}, "
        f"warmth={warm:.3f}, edge={edge:.3f}, texture={texture:.3f}"
    )
    return AIClassification(
        label=label,
        confidence=round(confidence, 3),
        rationale=rationale,
        provider="poke_vision",
        raw={"scores": scores, "image_path": str(image_path), "segment_id": segment.get("segment_id")},
    )


@dataclass
class NoOpAIAnalyzer:
    name: str = "noop"

    def classify_segment(self, segment: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        return AIClassification(label=None, confidence=None, rationale="no ai provider configured", provider=self.name, raw={"segment_id": segment.get("segment_id")})

    def classify_batch(self, segments: list[dict[str, Any]], context: dict[str, Any] | None = None) -> list[AIClassification]:
        return [self.classify_segment(segment, context) for segment in segments]


@dataclass
class LocalHeuristicAIAnalyzer:
    name: str = "local_heuristic"

    def classify_segment(self, segment: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        surface = segment.get("surface_class")
        if isinstance(surface, str) and surface.strip().lower() in VISION_LABELS:
            label = surface.strip().lower()
            return AIClassification(label=label, confidence=0.70, rationale="surface_class prior", provider=self.name, raw={"segment_id": segment.get("segment_id")})
        return AIClassification(label="unknown", confidence=0.0, rationale="no surface prior available", provider=self.name, raw={"segment_id": segment.get("segment_id")})

    def classify_batch(self, segments: list[dict[str, Any]], context: dict[str, Any] | None = None) -> list[AIClassification]:
        return [self.classify_segment(segment, context) for segment in segments]


@dataclass
class LocalCommandAIAnalyzer:
    command: list[str]
    name: str = "local_command"

    def classify_segment(self, segment: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        payload = json.dumps({"segment": segment, "context": context or {}}, ensure_ascii=False)
        result = subprocess.run(
            self.command,
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return AIClassification(
                label=None,
                confidence=None,
                rationale=f"local command failed: {result.stderr.strip()}",
                provider=self.name,
                raw={"returncode": result.returncode, "stderr": result.stderr},
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return AIClassification(
                label=None,
                confidence=None,
                rationale="local command returned invalid json",
                provider=self.name,
                raw={"stdout": result.stdout},
            )
        return AIClassification(
            label=data.get("label"),
            confidence=data.get("confidence"),
            rationale=data.get("rationale"),
            provider=self.name,
            raw=data,
        )

    def classify_batch(self, segments: list[dict[str, Any]], context: dict[str, Any] | None = None) -> list[AIClassification]:
        return [self.classify_segment(segment, context) for segment in segments]


@dataclass
class HttpAIAnalyzer:
    endpoint: str
    api_key: str | None = None
    model: str | None = None
    name: str = "http"
    timeout_seconds: float = 120.0

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def classify_segment(self, segment: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        payload = {
            "model": self.model,
            "segment": segment,
            "context": context or {},
        }
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return AIClassification(
            label=data.get("label"),
            confidence=data.get("confidence"),
            rationale=data.get("rationale"),
            provider=self.name,
            raw=data,
        )

    def classify_batch(self, segments: list[dict[str, Any]], context: dict[str, Any] | None = None) -> list[AIClassification]:
        return [self.classify_segment(segment, context) for segment in segments]


@dataclass
class OpenAICompatibleAnalyzer(HttpAIAnalyzer):
    name: str = "openai_compatible"


@dataclass
class PokeVisionProvider:
    image_cache_dir: Path
    endpoint: str | None = None
    model: str | None = None
    api_key_env: str = "BEACH_MAP_VISION_API_KEY"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
    max_output_tokens: int = 384
    allow_heuristic_fallback: bool = False
    image_width: int = 640
    image_height: int = 640
    image_zoom: int = 17
    image_extent_m: float = 500.0
    name: str = "poke_vision"

    def _api_key(self) -> str | None:
        api_key = (
            os.getenv(self.api_key_env)
            or os.getenv("BEACH_MAP_AI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if api_key:
            return api_key.strip().strip('\"').strip("'")
        env_path = self.image_cache_dir.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() in {self.api_key_env, "BEACH_MAP_AI_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"}:
                    return value.strip().strip('\"').strip("'")
        return None

    def _resolved_model(self) -> str | None:
        return (
            self.model
            or os.getenv("BEACH_MAP_VISION_MODEL")
            or os.getenv("BEACH_MAP_AI_MODEL")
            or (GEMINI_VISION_MODEL if self._api_key() else None)
            or PokeVision_LLM_MODEL
        )

    def _endpoint(self) -> str:
        return (
            self.endpoint
            or os.getenv("BEACH_MAP_VISION_ENDPOINT")
            or (GEMINI_VISION_ENDPOINT.format(model=self._resolved_model() or GEMINI_VISION_MODEL) if self._api_key() else PokeVision_LLM_ENDPOINT)
        )

    def _vision_prompt(self, segment: dict[str, Any], image_meta: dict[str, Any], context: dict[str, Any] | None = None) -> str:
        prompt = {
            "task": "Classify the coastal satellite image.",
            "labels": list(VISION_LABELS),
            "instructions": "Return only JSON with label, confidence, rationale, and usage. Usage should include prompt_tokens, completion_tokens, and total_tokens if available. Keep rationale <= 200 chars. label must be one of: rock, pebble, cobble, sand, built, unknown. Strict rocky policy: pebble and cobble are not rock. Label rules: rock = exposed bedrock, cliffs, jagged outcrops, steep hard shoreline, or dark irregular stone at the waterline; sand: light smooth beach/sediment; pebble/cobble: loose rounded stones, not rock; built = clearly man-made harbor, seawall, pier, ramp, buildings, or paved waterfront; unknown only when image quality/occlusion prevents a call.",
            "segment_id": segment.get("segment_id"),
        }
        return json.dumps(_jsonable(prompt), ensure_ascii=False)

    def _image_b64(self, image_path: str) -> str:
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")

    def _normalize_result(self, data: dict[str, Any], *, provider: str, image_meta: dict[str, Any]) -> AIClassification:
        label = _normalize_label(data.get("label"))
        confidence = data.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None
        rationale = data.get("rationale") or data.get("reason") or "vision model returned empty response"
        if label is None:
            label = "unknown"
        return AIClassification(
            label=label,
            confidence=confidence,
            rationale=rationale,
            provider=provider,
            raw={"parsed": data, "image_meta": image_meta},
        )

    def _call_ollama(self, segment: dict[str, Any], image_meta: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        model = self._resolved_model()
        if not model:
            raise ValueError("vision model not configured")
        prompt = self._vision_prompt(segment, image_meta, context)
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
            "messages": [
                {"role": "system", "content": "You are a coastal satellite-image classifier. Output only valid JSON."},
                {"role": "user", "content": prompt, "images": [self._image_b64(image_meta["image_path"])]},
            ],
        }
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response, retries_used, latency_ms = _post_with_retry(self._endpoint(), json_payload=payload, headers=headers, timeout=self.timeout_seconds, max_retries=self.max_retries, retry_backoff_seconds=self.retry_backoff_seconds)
        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        parsed = _validate_structured_output(_parse_llm_json(content))
        classification = self._normalize_result(parsed, provider=self.name, image_meta=image_meta)
        classification.raw.update({"endpoint": self._endpoint(), "model": model, "ollama": data, "retries_used": retries_used, "latency_ms": latency_ms, "request_id": str(uuid.uuid4())})
        return classification

    def _call_openai_compatible(self, segment: dict[str, Any], image_meta: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        model = self._resolved_model()
        if not model:
            raise ValueError("vision model not configured")
        prompt = self._vision_prompt(segment, image_meta, context)
        data_url = f"data:image/jpeg;base64,{self._image_b64(image_meta['image_path'])}"
        payload = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "You are a coastal satellite-image classifier. Output only valid JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        endpoint = self._endpoint().rstrip("/")
        if endpoint.endswith("/responses"):
            response, retries_used, latency_ms = _post_with_retry(endpoint, json_payload={"model": model, "input": payload["messages"], "temperature": 0.1}, headers=headers, timeout=self.timeout_seconds, max_retries=self.max_retries, retry_backoff_seconds=self.retry_backoff_seconds)
            data = response.json()
            content = ""
            if isinstance(data.get("output"), list):
                for item in data["output"]:
                    for part in item.get("content", []):
                        if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                            content += part.get("text", "")
            parsed = _validate_structured_output(_parse_llm_json(content))
            classification = self._normalize_result(parsed, provider=self.name, image_meta=image_meta)
            classification.raw.update({"endpoint": endpoint, "model": model, "responses": data, "retries_used": retries_used, "latency_ms": latency_ms, "request_id": str(uuid.uuid4())})
            return classification
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        response, retries_used, latency_ms = _post_with_retry(endpoint, json_payload={"model": model, "temperature": 0.1, "messages": payload["messages"]}, headers=headers, timeout=self.timeout_seconds, max_retries=self.max_retries, retry_backoff_seconds=self.retry_backoff_seconds)
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _validate_structured_output(_parse_llm_json(content))
        classification = self._normalize_result(parsed, provider=self.name, image_meta=image_meta)
        classification.raw.update({"endpoint": endpoint, "model": model, "openai": data, "retries_used": retries_used, "latency_ms": latency_ms, "request_id": str(uuid.uuid4())})
        return classification

    def _call_gemini(self, segment: dict[str, Any], image_meta: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        model = self._resolved_model()
        if not model:
            raise ValueError("vision model not configured")
        prompt = self._vision_prompt(segment, image_meta, context)
        api_key = self._api_key()
        if not api_key:
            raise ValueError("Gemini vision requires an API key")
        endpoint = self._endpoint().rstrip("/")
        if "/models/" not in endpoint:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        if endpoint.endswith(":generateContent"):
            url = f"{endpoint}?key={api_key}" if "key=" not in endpoint else endpoint
        else:
            url = f"{endpoint}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": self._image_b64(image_meta["image_path"])}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max(self.max_output_tokens, 1024),
                "thinkingConfig": {"thinkingBudget": 0},
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING", "enum": list(VISION_LABELS)},
                        "confidence": {"type": "NUMBER"},
                        "rationale": {"type": "STRING"},
                        "usage": {
                            "type": "OBJECT",
                            "properties": {
                                "prompt_tokens": {"type": "NUMBER"},
                                "completion_tokens": {"type": "NUMBER"},
                                "total_tokens": {"type": "NUMBER"},
                            },
                        },
                    },
                    "required": ["label", "confidence", "rationale"],
                },
            },
        }
        response, retries_used, latency_ms = _post_with_retry(url, json_payload=payload, timeout=self.timeout_seconds, max_retries=self.max_retries, retry_backoff_seconds=self.retry_backoff_seconds)
        data = response.json()
        candidates = data.get("candidates") or []
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        parsed = _validate_structured_output(_parse_llm_json(content))
        classification = self._normalize_result(parsed, provider=self.name, image_meta=image_meta)
        usage = _extract_gemini_usage(data, parsed)
        if usage:
            _append_gemini_usage_log(
                segment_id=segment.get("segment_id"),
                model=model,
                endpoint=url,
                usage=usage,
                response=data,
                parsed=parsed,
            )
            classification.raw["usage"] = usage
        classification.raw.update({"endpoint": url, "model": model, "gemini": data, "retries_used": retries_used, "latency_ms": latency_ms, "request_id": str(uuid.uuid4())})
        return classification

    def classify_segment(self, segment: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        logger = context.get("logger") if context else None
        image_meta = download_segment_satellite_image(
            segment,
            self.image_cache_dir,
            logger=logger,
            width=self.image_width,
            height=self.image_height,
            zoom=self.image_zoom,
            extent_m=self.image_extent_m,
            force=False,
        )
        try:
            model = self._resolved_model()
            if not model:
                return AIClassification(
                    label=None,
                    confidence=None,
                    rationale="vision model not configured",
                    provider=self.name,
                    raw={"image_meta": image_meta, "fallback": "none"},
                )
            endpoint = self._endpoint().lower()
            if "generativelanguage.googleapis.com" in endpoint or str(model).lower().startswith("gemini"):
                return self._call_gemini(segment, image_meta, context=context)
            if "openai" in endpoint or endpoint.endswith("/responses") or str(model).lower().startswith(("gpt-", "o1", "o3")):
                return self._call_openai_compatible(segment, image_meta, context=context)
            return self._call_ollama(segment, image_meta, context=context)
        except Exception as exc:
            if self.allow_heuristic_fallback:
                fallback = _heuristic_image_classification(Path(image_meta["image_path"]), segment)
                fallback.rationale = f"{fallback.rationale}; vision backend failed: {exc}"
                fallback.raw["image_meta"] = image_meta
                fallback.raw["vision_backend_error"] = str(exc)
                fallback.raw["fallback"] = "heuristic"
                return fallback
            return AIClassification(
                label=None,
                confidence=None,
                rationale=f"vision backend failed: {exc}",
                provider=self.name,
                raw={"image_meta": image_meta, "vision_backend_error": str(exc), "fallback": "none"},
            )

    def classify_batch(self, segments: list[dict[str, Any]], context: dict[str, Any] | None = None) -> list[AIClassification]:
        context = context or {}
        return [self.classify_segment(segment, context) for segment in segments]


PokeVisionAnalyzer = PokeVisionProvider
