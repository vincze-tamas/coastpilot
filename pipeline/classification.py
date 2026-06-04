from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from .config import PipelineConfig
from .osm import load_geojson, save_geojson

PROJECTED_CRS = "EPSG:32633"
SURFACE_ALIASES = {
    "sand": "sand",
    "pebble": "pebble",
    "pebbles": "pebble",
    "rock": "rock",
    "rocks": "rock",
    "rocky": "rock",
    "cobble": "cobble",
    "cobbles": "cobble",
    "cobblestone": "cobble",
    "cobblestones": "cobble",
    "built": "built",
    "concrete": "built",
    "paved": "built",
}


def _normalize_surface(surface: object) -> list[str]:
    if surface is None or pd.isna(surface):
        return []
    if not isinstance(surface, str):
        surface = str(surface)
    parts = [p.strip().lower() for p in surface.replace("/", ";").replace(",", ";").replace("|", ";").split(";")]
    labels: list[str] = []
    for part in parts:
        label = SURFACE_ALIASES.get(part)
        if label and label not in labels:
            labels.append(label)
    return labels


def _extract_beaches(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    beaches = raw[raw["natural"] == "beach"].copy()
    if beaches.empty:
        return gpd.GeoDataFrame(columns=["surface", "geometry"], geometry="geometry", crs="EPSG:4326")
    beaches["surface_class"] = beaches["surface"].apply(lambda s: "+".join(_normalize_surface(s)) or None)
    return beaches


def classify_segments(
    raw_path: Path,
    segments_path: Path,
    output_path: Path,
    config: PipelineConfig,
    logger,
    force: bool = False,
) -> gpd.GeoDataFrame:
    if output_path.exists() and not force:
        logger.info("Classified segments already exist, skipping classification: %s", output_path)
        return load_geojson(output_path)

    logger.info("Loading segments from %s", segments_path)
    segments = load_geojson(segments_path)
    raw = load_geojson(raw_path)

    segments_proj = segments.to_crs(PROJECTED_CRS)
    beaches = _extract_beaches(raw).to_crs(PROJECTED_CRS)

    beach_rows = list(beaches.itertuples())
    classified_labels: list[Optional[str]] = []
    needs_ai_flags: list[bool] = []
    matched_ids: list[Optional[int]] = []

    for seg in segments_proj.itertuples():
        midpoint = seg.geometry.interpolate(0.5, normalized=True)
        matched = None
        for beach in beach_rows:
            dist = midpoint.distance(beach.geometry)
            if dist <= config.match_distance_m:
                matched = beach
                break
        if matched is None:
            classified_labels.append(None)
            needs_ai_flags.append(True)
            matched_ids.append(None)
            continue
        surface_class = getattr(matched, "surface_class", None)
        classified_labels.append(surface_class)
        needs_ai_flags.append(not bool(surface_class))
        matched_ids.append(getattr(matched, "osm_id", None))

    segments["surface_class"] = classified_labels
    segments["needs_ai"] = needs_ai_flags
    segments["matched_beach_osm_id"] = matched_ids
    segments["pipeline_version"] = config.version
    save_geojson(segments, output_path)
    logger.info("Wrote classified segments to %s", output_path)
    return segments
