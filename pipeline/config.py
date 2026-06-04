from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path = Path("/opt/apps/coastpilot")
    version: str = "0.3.0"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    bbox: Tuple[float, float, float, float] = (13.35, 44.95, 15.05, 45.85)
    segment_length_m: float = 50.0
    match_distance_m: float = 75.0
    ai_provider: str = "noop"
    ai_endpoint: str | None = None
    ai_model: str | None = None
    ai_api_key_env: str = "BEACH_MAP_AI_API_KEY"
    ai_timeout_seconds: float = 120.0
    ai_max_retries: int = 2
    ai_retry_backoff_seconds: float = 1.5
    ai_max_output_tokens: int = 384
    ai_budget_max_tokens: int | None = None
    ai_budget_max_usd: float | None = None
    vision_image_width: int = 640
    vision_image_height: int = 640
    rock_min_confidence: float = 0.80
    prefer_rock_weight: float = 1.0

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def cache_dir(self) -> Path:
        return self.project_root / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def raw_geojson(self) -> Path:
        return self.data_dir / "raw_osm.geojson"

    @property
    def segments_geojson(self) -> Path:
        return self.data_dir / "segments.geojson"

    @property
    def classified_geojson(self) -> Path:
        return self.data_dir / "segments_classified.geojson"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / f"manifest_v{self.version}.json"

    @property
    def log_path(self) -> Path:
        return self.logs_dir / f"pipeline_v{self.version}.log"
