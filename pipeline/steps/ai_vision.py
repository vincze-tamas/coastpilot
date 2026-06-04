from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import pandas as pd

from ..ai.factory import build_ai_analyzer
from ..config import PipelineConfig
from ..osm import load_geojson, save_geojson
from ..recommendation import preference_score, recommendation_band
from ..vision.imagery import resolve_imagery_request
from .base import StepResult




def _confidence_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _strict_rock_label(label: str | None, confidence: Any, rock_min_confidence: float) -> str | None:
    if label != "rock":
        return label
    conf = _confidence_value(confidence)
    if conf is None or conf < rock_min_confidence:
        return "unknown"
    return label


def _apply_rock_objective_columns(segments, config: PipelineConfig):
    updated = segments.copy()
    if "ai_surface_class" not in updated.columns:
        return updated

    def usable(value: Any) -> bool:
        return value not in (None, "", "None") and not pd.isna(value)

    def raw_label(row) -> str | None:
        value = row.get("ai_raw_surface_class") if "ai_raw_surface_class" in updated.columns else None
        if usable(value):
            return value
        return row.get("ai_surface_class")

    def raw_confidence(row) -> Any:
        value = row.get("ai_raw_confidence") if "ai_raw_confidence" in updated.columns else None
        if usable(value):
            return value
        return row.get("ai_confidence")

    def strict_label_row(row) -> str | None:
        return _strict_rock_label(
            raw_label(row),
            raw_confidence(row),
            getattr(config, "rock_min_confidence", 0.90),
        )

    def score_row(row) -> float:
        return preference_score(
            row.get("ai_surface_class"),
            raw_confidence(row),
            prefer_rock=getattr(config, "prefer_rock_weight", 1.0),
        )

    updated["ai_surface_class"] = updated.apply(strict_label_row, axis=1)
    updated["rock_preference_score"] = updated.apply(score_row, axis=1)
    updated["rock_recommendation_band"] = updated["rock_preference_score"].apply(recommendation_band)
    return updated

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
    target_segment_ids: set[str] | None = None

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
        target_segment_ids = {str(s) for s in self.target_segment_ids} if self.target_segment_ids else None
        target_total = 0
        target_missing = 0
        for idx, row in segments.iterrows():
            segment_id = str(row.get("segment_id") or idx)
            if target_segment_ids is not None:
                if segment_id not in target_segment_ids:
                    continue
                target_total += 1
            needs_ai = bool(row.get("needs_ai", True))
            already_done = str(row.get("ai_vision_version") or "") == config.version
            current_label = row.get("ai_surface_class")
            label_missing = current_label in (None, "", "None")
            if target_segment_ids is not None and (label_missing or str(current_label).strip().lower() == "unknown"):
                target_missing += 1
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
                candidates.append((idx, has_cached_image, segment_id))
            elif needs_ai or label_missing:
                candidates.append((idx, has_cached_image, segment_id))

        candidates.sort(key=lambda item: (0 if item[1] else 1, item[2]))
        candidates = candidates[: self.batch_limit]

        if not candidates:
            logger.info("AI vision batch already complete for the requested limit, applying rock objective columns and skipping model calls")
            updated = _apply_rock_objective_columns(segments, config)
            save_geojson(updated, output_path)
            _backup_output_file(output_path, backup_root, logger)
            return StepResult(name=self.name, output_path=output_path, records=int(len(updated)), skipped=True, details={"batch_limit": self.batch_limit, "processed": 0, "skipped_already_done": skipped_already_done, "cache_hit_candidates": cache_hits, "target_segment_ids": len(target_segment_ids) if target_segment_ids is not None else None, "target_matched_segments": target_total if target_segment_ids is not None else None, "target_missing_or_unknown_before_run": target_missing if target_segment_ids is not None else None, "remaining_unlabeled": int(((updated.get("ai_surface_class").isna()) | (updated.get("ai_surface_class") == "")).sum()) if "ai_surface_class" in updated.columns else int(len(updated))})

        logger.info("Running AI vision classification for %s segments (limit=%s)", len(candidates), self.batch_limit)
        updated = segments.copy()
        results = []
        newly_labeled = 0
        for idx, _cache_hit, _segment_id in candidates:
            segment = updated.loc[idx].to_dict()
            classification = analyzer.classify_segment(segment, context={"version": config.version, "batch_limit": self.batch_limit, "logger": logger})
            rock_min_confidence = getattr(config, "rock_min_confidence", 0.90)
            label = _strict_rock_label(classification.label, classification.confidence, rock_min_confidence)
            rationale = classification.rationale
            if label != classification.label:
                rationale = f"{rationale}; demoted below rock_min_confidence={rock_min_confidence:.2f}"
            updated.at[idx, "ai_raw_surface_class"] = classification.label
            updated.at[idx, "ai_raw_confidence"] = classification.confidence
            updated.at[idx, "ai_surface_class"] = label
            updated.at[idx, "ai_confidence"] = classification.confidence
            updated.at[idx, "ai_rationale"] = rationale
            updated.at[idx, "ai_provider"] = classification.provider
            updated.at[idx, "ai_vision_version"] = config.version
            updated.at[idx, "ai_vision_source"] = classification.raw.get("image_meta", {}).get("provider")
            updated.at[idx, "vision_image_path"] = classification.raw.get("image_meta", {}).get("image_path")
            updated.at[idx, "vision_image_url"] = classification.raw.get("image_meta", {}).get("url")
            updated.at[idx, "vision_midpoint_lon"] = classification.raw.get("image_meta", {}).get("midpoint_lon")
            updated.at[idx, "vision_midpoint_lat"] = classification.raw.get("image_meta", {}).get("midpoint_lat")
            cache_hit = classification.raw.get("image_meta", {}).get("cache_hit")
            updated.at[idx, "vision_image_cached"] = float(cache_hit) if cache_hit is not None else float("nan")
            if classification.label not in (None, "", "unknown") and classification.confidence is not None:
                updated.at[idx, "needs_ai"] = False
                newly_labeled += 1
            results.append(classification)

        updated["pipeline_version"] = config.version
        updated = _apply_rock_objective_columns(updated, config)
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
                "target_segment_ids": len(target_segment_ids) if target_segment_ids is not None else None,
                "target_matched_segments": target_total if target_segment_ids is not None else None,
                "target_missing_or_unknown_before_run": target_missing if target_segment_ids is not None else None,
                "remaining_unlabeled": remaining_unlabeled,
                "vision_backend": analyzer.name,
                "vision_model": getattr(analyzer, "model", None),
                "image_cache_dir": str(analyzer.image_cache_dir),
            },
        )


AIClassificationStep = AIVisionStep
