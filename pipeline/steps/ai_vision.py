from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone

import geopandas as gpd

from ..ai.factory import build_ai_analyzer
from ..config import PipelineConfig
from ..osm import load_geojson, save_geojson
from ..vision.imagery import resolve_imagery_request
from .base import StepResult


def _backup_output_file(output_path, backup_root, logger) -> None:
    if not output_path.exists():
        logger.warning("Backup skipped because output file is missing: %s", output_path)
        return

    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
    backup_path = backup_root / base_name
    counter = 1
    while backup_path.exists():
        backup_path = backup_root / f"{output_path.stem}_{timestamp}_{counter}{output_path.suffix}"
        counter += 1
    shutil.copy2(output_path, backup_path)
    logger.info("Backed up AI batch output to %s", backup_path)


@dataclass
class AIVisionStep:
    name: str = "ai_vision"
    batch_limit: int = 100

    def run(self, config: PipelineConfig, logger, force: bool = False) -> StepResult:
        output_path = config.classified_geojson
        segments = load_geojson(output_path) if output_path.exists() else load_geojson(config.segments_geojson)

        backup_root = config.data_dir / "backups" / "segments_classified"

        if segments.empty:
            logger.warning("No segments available for AI vision classification")
            save_geojson(segments, output_path)
            _backup_output_file(output_path, backup_root, logger)
            return StepResult(name=self.name, output_path=output_path, records=0, details={"batch_limit": self.batch_limit})

        analyzer = build_ai_analyzer(config, overrides={"ai_provider": "poke_vision"})
        candidates: list[int] = []
        skipped_cached = 0
        for idx, row in segments.iterrows():
            needs_ai = bool(row.get("needs_ai", True))
            already_done = str(row.get("ai_vision_version") or "") == config.version
            current_label = row.get("ai_surface_class")
            label_missing = current_label in (None, "", "None")
            if already_done:
                continue
            try:
                request = resolve_imagery_request(row.to_dict(), config.cache_dir)
                has_cached_image = request.image_path.exists() and request.meta_path.exists()
            except Exception as exc:
                logger.warning("Unable to resolve imagery request for %s: %s", row.get("segment_id"), exc)
                has_cached_image = False
            if has_cached_image:
                skipped_cached += 1
                continue
            if force:
                if label_missing:
                    candidates.append(idx)
            elif needs_ai or label_missing:
                candidates.append(idx)
            if len(candidates) >= self.batch_limit:
                break

        if not candidates:
            logger.info("AI vision batch already complete for the requested limit, skipping")
            _backup_output_file(output_path, backup_root, logger)
            return StepResult(name=self.name, output_path=output_path, records=int(len(segments)), skipped=True, details={"batch_limit": self.batch_limit, "processed": 0, "skipped_cached": skipped_cached})

        logger.info("Running AI vision classification for %s segments (limit=%s)", len(candidates), self.batch_limit)
        updated = segments.copy()
        results = []
        for idx in candidates:
            segment = updated.loc[idx].to_dict()
            classification = analyzer.classify_segment(segment, context={"version": config.version, "batch_limit": self.batch_limit, "logger": logger})
            updated.at[idx, "ai_surface_class"] = classification.label
            updated.at[idx, "ai_confidence"] = classification.confidence
            updated.at[idx, "ai_rationale"] = classification.rationale
            updated.at[idx, "ai_provider"] = classification.provider
            updated.at[idx, "ai_vision_version"] = config.version
            updated.at[idx, "ai_vision_source"] = classification.raw.get("image_meta", {}).get("provider")
            updated.at[idx, "vision_image_path"] = classification.raw.get("image_meta", {}).get("image_path")
            updated.at[idx, "vision_image_url"] = classification.raw.get("image_meta", {}).get("url")
            updated.at[idx, "vision_midpoint_lon"] = classification.raw.get("image_meta", {}).get("midpoint_lon")
            updated.at[idx, "vision_midpoint_lat"] = classification.raw.get("image_meta", {}).get("midpoint_lat")
            updated.at[idx, "vision_image_cached"] = classification.raw.get("image_meta", {}).get("cache_hit")
            updated.at[idx, "needs_ai"] = False
            results.append(classification)

        updated["pipeline_version"] = config.version
        save_geojson(updated, output_path)
        _backup_output_file(output_path, backup_root, logger)
        downloaded = sum(1 for r in results if r.raw.get("image_meta", {}).get("cache_hit") is False)
        cached = sum(1 for r in results if r.raw.get("image_meta", {}).get("cache_hit") is True)
        logger.info("AI vision classified %s segments; downloaded %s images and reused %s cache entries", len(results), downloaded, cached)
        return StepResult(
            name=self.name,
            output_path=output_path,
            records=int(len(updated)),
            details={
                "batch_limit": self.batch_limit,
                "processed_segments": len(results),
                "skipped_cached_candidates": skipped_cached,
                "downloaded_images": downloaded,
                "cached_images": cached,
                "vision_backend": analyzer.name,
                "vision_model": getattr(analyzer, "model", None),
                "image_cache_dir": str(analyzer.image_cache_dir),
            },
        )


AIClassificationStep = AIVisionStep
