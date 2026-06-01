from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from ..config import PipelineConfig
from ..osm import load_geojson, save_geojson
from .base import AIAnalyzer


def apply_ai_classification(
    input_path: Path,
    output_path: Path,
    analyzer: AIAnalyzer,
    config: PipelineConfig,
    logger,
    force: bool = False,
) -> gpd.GeoDataFrame:
    if output_path.exists() and not force:
        logger.info("AI-enriched segments already exist, skipping AI classification: %s", output_path)
        return load_geojson(output_path)

    segments = load_geojson(input_path)
    if segments.empty:
        logger.warning("No segments found for AI classification")
        save_geojson(segments, output_path)
        return segments

    context = {
        "version": config.version,
        "bbox": config.bbox,
        "segment_length_m": config.segment_length_m,
        "match_distance_m": config.match_distance_m,
    }
    needs_ai_mask = segments.get("needs_ai")
    if needs_ai_mask is None:
        needs_ai_mask = [True] * len(segments)

    results = []
    updated = segments.copy()
    for idx, row in updated.iterrows():
        if not bool(row.get("needs_ai", True)):
            continue
        classification = analyzer.classify_segment(row.to_dict(), context=context)
        updated.at[idx, "ai_surface_class"] = classification.label
        updated.at[idx, "ai_confidence"] = classification.confidence
        updated.at[idx, "ai_rationale"] = classification.rationale
        updated.at[idx, "ai_provider"] = classification.provider
        updated.at[idx, "needs_ai"] = classification.needs_human_review
        results.append(classification)

    updated["pipeline_version"] = config.version
    save_geojson(updated, output_path)
    logger.info("AI step processed %s segments and wrote %s", len(results), output_path)
    return updated
