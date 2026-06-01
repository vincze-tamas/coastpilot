from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from pipeline.config import PipelineConfig
from pipeline.logging_utils import setup_logging
from pipeline.steps.ai_vision import AIVisionStep
from pipeline.steps.collection import OSMDataCollectionStep
from pipeline.steps.eu_enrichment import EUEnrichmentStep
from pipeline.steps.initial_classification import InitialClassificationStep
from pipeline.steps.segmentation_step import SegmentationStep


def build_manifest(config: PipelineConfig, results: dict[str, dict], source_meta: dict, eu_meta: dict | None, vision_meta: dict | None) -> dict:
    return {
        "version": config.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bbox": config.bbox,
        "segment_length_m": config.segment_length_m,
        "match_distance_m": config.match_distance_m,
        "ai_provider": config.ai_provider,
        "raw_osm": str(config.raw_geojson),
        "segments": str(config.segments_geojson),
        "classified_segments": str(config.classified_geojson),
        "source_meta": source_meta,
        "eu_dataset": eu_meta,
        "vision": vision_meta,
        "steps": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beach map OSM pipeline")
    parser.add_argument("--step", choices=["all", "fetch", "segment", "classify", "eu", "ai"], default="all")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if files exist")
    parser.add_argument("--segment-length-m", type=float, default=50.0)
    parser.add_argument("--match-distance-m", type=float, default=75.0)
    parser.add_argument("--bbox", type=str, default=None, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--ai-provider", type=str, default="noop")
    parser.add_argument("--ai-endpoint", type=str, default=None)
    parser.add_argument("--ai-model", type=str, default=None)
    parser.add_argument("--vision-limit", type=int, default=100, help="Maximum number of segments to process in the AI vision step")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    bbox = tuple(map(float, args.bbox.split(","))) if args.bbox else PipelineConfig().bbox
    return PipelineConfig(
        segment_length_m=args.segment_length_m,
        match_distance_m=args.match_distance_m,
        bbox=bbox,
        ai_provider=args.ai_provider,
        ai_endpoint=args.ai_endpoint,
        ai_model=args.ai_model,
    )


def main() -> int:
    args = parse_args()
    config = _config_from_args(args)
    config.project_root.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(config.log_path)
    logger.info("Starting pipeline version %s", config.version)
    logger.info("Configuration: %s", config)

    results: dict[str, dict] = {}
    source_meta: dict = {}
    eu_meta: dict = {}
    vision_meta: dict = {}

    collection_step = OSMDataCollectionStep()
    segmentation_step = SegmentationStep()
    classification_step = InitialClassificationStep()
    eu_step = EUEnrichmentStep()
    vision_step = AIVisionStep(batch_limit=args.vision_limit)

    if args.step in ("all", "fetch"):
        outcome = collection_step.run(config, logger, force=args.force)
        results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        source_meta = outcome.details.get("source_meta", source_meta)

    if args.step in ("all", "segment"):
        if not config.raw_geojson.exists():
            outcome = collection_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
            source_meta = outcome.details.get("source_meta", source_meta)
        outcome = segmentation_step.run(config, logger, force=args.force)
        results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}

    if args.step in ("all", "classify"):
        if not config.raw_geojson.exists():
            outcome = collection_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
            source_meta = outcome.details.get("source_meta", source_meta)
        if not config.segments_geojson.exists():
            outcome = segmentation_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        outcome = classification_step.run(config, logger, force=args.force)
        results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}

    eu_ran = False

    if args.step in ("all", "eu"):
        if not config.raw_geojson.exists():
            outcome = collection_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
            source_meta = outcome.details.get("source_meta", source_meta)
        if not config.segments_geojson.exists():
            outcome = segmentation_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        if not config.classified_geojson.exists():
            outcome = classification_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        outcome = eu_step.run(config, logger, force=args.force)
        results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        eu_meta = outcome.details.get("dataset", eu_meta)
        eu_ran = True

    if args.step in ("all", "ai"):
        if not config.raw_geojson.exists():
            outcome = collection_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
            source_meta = outcome.details.get("source_meta", source_meta)
        if not config.segments_geojson.exists():
            outcome = segmentation_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        if not config.classified_geojson.exists():
            outcome = classification_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        if not eu_ran:
            outcome = eu_step.run(config, logger, force=args.force)
            results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
            eu_meta = outcome.details.get("dataset", eu_meta)
        outcome = vision_step.run(config, logger, force=args.force)
        results[outcome.name] = {"output_path": str(outcome.output_path), "records": outcome.records, "skipped": outcome.skipped, "details": outcome.details}
        vision_meta = outcome.details

    manifest = build_manifest(config, results, source_meta, eu_meta, vision_meta)
    config.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote manifest to %s", config.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
