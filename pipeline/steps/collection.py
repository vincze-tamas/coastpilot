from __future__ import annotations

from dataclasses import dataclass

from ..config import PipelineConfig
from ..osm import fetch_overpass_json, overpass_to_geodataframe, save_geojson
from .base import StepResult


@dataclass
class OSMDataCollectionStep:
    name: str = "osm_collection"

    def run(self, config: PipelineConfig, logger, force: bool = False) -> StepResult:
        if config.raw_geojson.exists() and not force:
            logger.info("Raw OSM already exists, skipping collection: %s", config.raw_geojson)
            return StepResult(name=self.name, output_path=config.raw_geojson, skipped=True)

        data, meta = fetch_overpass_json(config, logger, force=force)
        raw_gdf = overpass_to_geodataframe(data, config)
        save_geojson(raw_gdf, config.raw_geojson)
        logger.info("Wrote raw OSM data to %s", config.raw_geojson)
        return StepResult(name=self.name, output_path=config.raw_geojson, records=int(len(raw_gdf)), details={"source_meta": meta})
