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
        candidates: list[tuple[int, bool, str]] = []
        skipped_already_done = 0
        cache_hits = 0
        for idx, row in segments.iterrows():
            needs_ai = bool(row.get("needs_ai", True))
            already_done = str(row.get("ai_vision_version") or "") == config.version
            current_label = row.get("ai_surface_class")
            label_missing = current_label in (None, "", "None")
            if already_done and not force:
                skipped_already_done += 1
                continue
            try:
                request = resolve_imagery_request(row.to_dict(), config.cache_dir)
                has_cached_image = request.image_path.exists() and request.meta_path.exists()
            except Exception as exc:
                logger.warning("Unable to resolve imagery request for %s: %s", row.get("segment_id"), exc)
                has_cached_image = False
            if has_cached_image:
                cache_hits += 1
            if force:
                if label_missing:
                    candidates.append((idx, has_cached_image, str(row.get("segment_id") or idx)))
            elif needs_ai or label_missing:
                candidates.append((idx, has_cached_image, str(row.get("segment_id") or idx)))

        candidates.sort(key=lambda item: (0 if item[1] else 1, item[2]))
        candidates = candidates[: self.batch_limit]

        if not candidates:
            logger.info("AI vision batch already complete for the requested limit, skipping")
            _backup_output_file(output_path, backup_root, logger)
            return StepResult(name=self.name, output_path=output_path, records=int(len(segments)), skipped=True, details={"batch_limit": self.batch_limit, "processed": 0, "skipped_already_done": skipped_already_done, "cache_hit_candidates": cache_hits, "remaining_unlabeled": int(((segments.get("ai_surface_class").isna()) | (segments.get("ai_surface_class") == "")).sum()) if "ai_surface_class" in segments.columns else int(len(segments))})

        logger.info("Running AI vision classification for %s segments (limit=%s)", len(candidates), self.batch_limit)
        updated = segments.copy()
        results = []
        newly_labeled = 0
        for idx, _cache_hit, _segment_id in candidates:
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
            if classification.label not in (None, "", "unknown") and classification.confidence is not None:
                updated.at[idx, "needs_ai"] = False
                newly_labeled += 1
            results.append(classification)

        updated["pipeline_version"] = config.version
        save_geojson(updated, output_path)
        _backup_output_file(output_path, backup_root, logger)
        downloaded = sum(1 for r in results if r.raw.get("image_meta", {}).get("cache_hit") is False)
        cached = sum(1 for r in results if r.raw.get("image_meta", {}).get("cache_hit") is True)
        unknown_count = sum(1 for r in results if r.label in (None, "unknown", ""))
        remaining_unlabeled = int(((updated.get("ai_surface_class").isna()) | (updated.get("ai_surface_class") == "") | (updated.get("ai_surface_class") == "unknown")).sum()) if "ai_surface_class" in updated.columns else 0
        logger.info("AI vision classified %s segments; downloaded %s images and reused %s cache entries", len(results), downloaded, cached)
        return StepResult(
            name=self.name,
            output_path=output_path,
            records=int(len(updated)),
            details={
                "batch_limit": self.batch_limit,
                "processed_segments": len(results),
                "newly_labeled_segments": newly_labeled,
                "unknown_results": unknown_count,
                "skipped_already_done": skipped_already_done,
                "cache_hit_candidates": cache_hits,
                "downloaded_images": downloaded,
                "cached_images": cached,
                "remaining_unlabeled": remaining_unlabeled,
                "vision_backend": analyzer.name,
                "vision_model": getattr(analyzer, "model", None),
                "image_cache_dir": str(analyzer.image_cache_dir),
            },
        )


AIClassificationStep = AIVisionStep
