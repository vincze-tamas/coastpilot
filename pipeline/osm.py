from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import requests
from shapely.geometry import LineString, Polygon, mapping

from .config import PipelineConfig


def build_overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    return f"""
[out:json][timeout:180];
(
  way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["natural"="beach"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["natural"="beach"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["surface"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["surface"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out body geom;
""".strip()


def query_cache_paths(cache_dir: Path, query: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"overpass_{digest}.json", cache_dir / f"overpass_{digest}.meta.json"


def fetch_overpass_json(
    config: PipelineConfig,
    logger,
    force: bool = False,
    retries: int = 3,
    backoff_seconds: float = 3.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = build_overpass_query(config.bbox)
    cache_path, meta_path = query_cache_paths(config.cache_dir, query)

    if cache_path.exists() and not force:
        logger.info("Using cached Overpass response: %s", cache_path)
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return data, meta

    headers = {"User-Agent": f"coastpilot/{config.version} (Poke)"}
    payload = {"data": query}
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.info("Fetching Overpass data (attempt %s/%s)", attempt, retries)
            response = requests.post(config.overpass_url, data=payload, headers=headers, timeout=240)
            response.raise_for_status()
            data = response.json()
            meta = {
                "source": config.overpass_url,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "bbox": config.bbox,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version": config.version,
            }
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Cached Overpass response at %s", cache_path)
            return data, meta
        except Exception as exc:  # pragma: no cover - operational logging
            last_error = exc
            logger.warning("Overpass fetch failed on attempt %s: %s", attempt, exc)
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)

    assert last_error is not None
    raise last_error


def _element_to_geometry(element: dict[str, Any]):
    geom = element.get("geometry") or []
    if not geom:
        return None
    coords = [(pt["lon"], pt["lat"]) for pt in geom]
    tags = element.get("tags", {})
    if tags.get("natural") == "coastline":
        return LineString(coords)
    if len(coords) >= 4 and coords[0] == coords[-1]:
        try:
            return Polygon(coords)
        except Exception:
            return LineString(coords)
    return LineString(coords)


def overpass_to_geodataframe(data: dict[str, Any], config: PipelineConfig) -> gpd.GeoDataFrame:
    features: List[Dict[str, Any]] = []
    seen: set[tuple[str | None, int | None]] = set()
    for element in data.get("elements", []):
        key = (element.get("type"), element.get("id"))
        if key in seen:
            continue
        seen.add(key)
        geom = _element_to_geometry(element)
        if geom is None:
            continue
        tags = element.get("tags", {})
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                    "tags": tags,
                    "natural": tags.get("natural"),
                    "surface": tags.get("surface"),
                    "pipeline_version": config.version,
                    "source": "overpass",
                },
            }
        )
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326") if features else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    if not gdf.empty:
        gdf = gdf.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
    return gdf


def save_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(gdf.to_json())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_geojson(path: Path) -> gpd.GeoDataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    return gpd.GeoDataFrame.from_features(data.get("features", []), crs="EPSG:4326")
