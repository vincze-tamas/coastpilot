from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import gmtime, strftime

import geopandas as gpd
import requests

from ..config import PipelineConfig
from ..osm import load_geojson, save_geojson
from .base import StepResult

PROJECTED_CRS = "EPSG:32633"
EU_DATASET_URL = "https://gisco-services.ec.europa.eu/distribution/v2/coas/geojson/COAS_RG_10M_2016_4326.geojson"
EU_DATASET_NAME = "GISCO Coastal lines 2016 (10M, EPSG:4326)"
EU_DATASET_VERSION = "2016"


def _cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(EU_DATASET_URL.encode("utf-8")).hexdigest()[:16]
    base = f"gisco_coastline_2016_10m_4326_{digest}"
    return cache_dir / f"{base}.geojson", cache_dir / f"{base}.meta.json"


def _download_dataset(cache_path: Path, meta_path: Path, logger, version: str, force: bool = False) -> dict:
    if cache_path.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading EU coastal dataset from %s", EU_DATASET_URL)
    response = requests.get(
        EU_DATASET_URL,
        headers={"User-Agent": f"coastpilot/{version} (EUEnrichmentStep)"},
        timeout=240,
    )
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    meta = {
        "name": EU_DATASET_NAME,
        "url": EU_DATASET_URL,
        "version": EU_DATASET_VERSION,
        "downloaded_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "bytes": len(response.content),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Cached EU coastal dataset at %s", cache_path)
    return meta


def _already_enriched(segments: gpd.GeoDataFrame, version: str) -> bool:
    if "eu_enrichment_version" not in segments.columns or segments.empty:
        return False
    values = segments["eu_enrichment_version"].dropna().astype(str).unique().tolist()
    return len(values) == 1 and values[0] == version


@dataclass
class EUEnrichmentStep:
    name: str = "eu_enrichment"

    def run(self, config: PipelineConfig, logger, force: bool = False) -> StepResult:
        output_path = config.classified_geojson
        segments = load_geojson(output_path) if output_path.exists() else load_geojson(config.classified_geojson)

        cache_path, meta_path = _cache_paths(config.cache_dir)
        if _already_enriched(segments, config.version) and not force:
            logger.info("EU enrichment already present, skipping: %s", output_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            return StepResult(
                name=self.name,
                output_path=output_path,
                records=int(len(segments)),
                skipped=True,
                details={"dataset": meta or {"name": EU_DATASET_NAME, "url": EU_DATASET_URL}},
            )

        if segments.empty:
            logger.warning("No segments found for EU enrichment")
            save_geojson(segments, output_path)
            return StepResult(
                name=self.name,
                output_path=output_path,
                records=0,
                details={"dataset": {"name": EU_DATASET_NAME, "url": EU_DATASET_URL}},
            )

        dataset_meta = _download_dataset(cache_path, meta_path, logger, config.version, force=force)

        eu = gpd.read_file(cache_path)
        if eu.crs is None:
            eu = eu.set_crs("EPSG:4326")
        eu = eu.to_crs(PROJECTED_CRS)

        segments_proj = segments.to_crs(PROJECTED_CRS)
        left = segments_proj[["geometry"]].copy()
        join = gpd.sjoin_nearest(
            left,
            eu[["geometry", "COAS_ID"]],
            how="left",
            max_distance=config.match_distance_m,
            distance_col="eu_distance_m",
        )
        join = join[~join.index.duplicated(keep="first")]

        matched = join["index_right"].notna()
        matched_indices = join.index[matched]

        updated = segments.copy()
        updated["eu_enrichment_version"] = config.version
        updated["eu_dataset_name"] = EU_DATASET_NAME
        updated["eu_dataset_url"] = EU_DATASET_URL
        updated["eu_matched"] = False
        updated["eu_distance_m"] = None
        updated["eu_match_id"] = None

        if len(matched_indices) > 0:
            updated.loc[matched_indices, "type"] = "coastline"
            updated.loc[matched_indices, "confidence"] = 0.85
            updated.loc[matched_indices, "source"] = "eu"
            updated.loc[matched_indices, "eu_matched"] = True
            updated.loc[matched_indices, "eu_distance_m"] = join.loc[matched_indices, "eu_distance_m"].astype(float)
            updated.loc[matched_indices, "eu_match_id"] = join.loc[matched_indices, "COAS_ID"].astype("Int64")

        save_geojson(updated, output_path)
        matched_count = int(updated["eu_matched"].sum())
        logger.info("EU enrichment matched %s of %s segments using %s", matched_count, len(updated), EU_DATASET_NAME)
        return StepResult(
            name=self.name,
            output_path=output_path,
            records=int(len(updated)),
            details={
                "dataset": dataset_meta,
                "matched_segments": matched_count,
                "total_segments": int(len(updated)),
                "match_distance_m": config.match_distance_m,
                "cache_path": str(cache_path),
            },
        )
