from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import geopandas as gpd
import requests

PROJECTED_CRS = "EPSG:32633"
WGS84 = "EPSG:4326"
ESRI_EXPORT_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
GOOGLE_STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
MAPBOX_STATIC_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"


@dataclass(frozen=True)
class ImageryRequest:
    provider: str
    url: str
    image_path: Path
    meta_path: Path
    midpoint_lon: float
    midpoint_lat: float
    width: int
    height: int
    zoom: int | None = None
    extent_m: float | None = None


def _segment_midpoint_lonlat(geometry) -> tuple[float, float]:
    if geometry is None or geometry.is_empty:
        raise ValueError("Cannot compute midpoint for empty geometry")
    projected = gpd.GeoSeries([geometry], crs=WGS84).to_crs(PROJECTED_CRS).iloc[0]
    midpoint = projected.interpolate(0.5, normalized=True)
    midpoint_wgs = gpd.GeoSeries([midpoint], crs=PROJECTED_CRS).to_crs(WGS84).iloc[0]
    return float(midpoint_wgs.x), float(midpoint_wgs.y)


def _bbox_from_point(lon: float, lat: float, extent_m: float) -> tuple[float, float, float, float]:
    half_m = extent_m / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1e-6, 111_320.0 * math.cos(math.radians(lat)))
    delta_lat = half_m / meters_per_deg_lat
    delta_lon = half_m / meters_per_deg_lon
    return (lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat)


def _sanitize_segment_id(segment_id: str | None) -> str:
    value = (segment_id or "segment").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:64]


def _build_google_url(lat: float, lon: float, width: int, height: int, zoom: int, api_key: str) -> str:
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{width}x{height}",
        "maptype": "satellite",
        "key": api_key,
    }
    return f"{GOOGLE_STATIC_MAPS_URL}?{urlencode(params)}"


def _build_mapbox_url(lat: float, lon: float, width: int, height: int, zoom: int, token: str) -> str:
    return f"{MAPBOX_STATIC_URL}/{lon},{lat},{zoom}/{width}x{height}?{urlencode({'access_token': token})}"


def _build_esri_url(lat: float, lon: float, width: int, height: int, extent_m: float) -> str:
    min_lon, min_lat, max_lon, max_lat = _bbox_from_point(lon, lat, extent_m)
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{width},{height}",
        "format": "jpg",
        "f": "image",
        "noData": "",
        "noDataInterpretation": "esriNoDataMatchAny",
    }
    return f"{ESRI_EXPORT_URL}?{urlencode(params)}"


def resolve_imagery_request(
    segment: dict[str, Any],
    cache_dir: Path,
    *,
    width: int = 640,
    height: int = 640,
    zoom: int = 17,
    extent_m: float = 500.0,
) -> ImageryRequest:
    geometry = segment.get("geometry")
    if geometry is None:
        raise ValueError("Segment is missing geometry")
    lon, lat = _segment_midpoint_lonlat(geometry)
    segment_id = _sanitize_segment_id(str(segment.get("segment_id")))

    google_key = os.getenv("BEACH_MAP_GOOGLE_STATIC_MAPS_KEY") or os.getenv("GOOGLE_STATIC_MAPS_API_KEY")
    mapbox_token = os.getenv("BEACH_MAP_MAPBOX_ACCESS_TOKEN") or os.getenv("MAPBOX_ACCESS_TOKEN")

    if google_key:
        provider = "google_static_maps"
        url = _build_google_url(lat, lon, width, height, zoom, google_key)
    elif mapbox_token:
        provider = "mapbox_static"
        url = _build_mapbox_url(lat, lon, width, height, zoom, mapbox_token)
    else:
        provider = "esri_world_imagery"
        url = _build_esri_url(lat, lon, width, height, extent_m)

    digest = hashlib.sha256(f"{provider}|{segment_id}|{lat:.6f}|{lon:.6f}|{width}|{height}|{zoom}|{extent_m}".encode("utf-8")).hexdigest()[:20]
    request_dir = cache_dir / "satellite"
    request_dir.mkdir(parents=True, exist_ok=True)
    image_path = request_dir / f"{segment_id}_{digest}.jpg"
    meta_path = request_dir / f"{segment_id}_{digest}.meta.json"
    return ImageryRequest(
        provider=provider,
        url=url,
        image_path=image_path,
        meta_path=meta_path,
        midpoint_lon=lon,
        midpoint_lat=lat,
        width=width,
        height=height,
        zoom=zoom,
        extent_m=extent_m,
    )


def download_segment_satellite_image(
    segment: dict[str, Any],
    cache_dir: Path,
    logger,
    *,
    width: int = 640,
    height: int = 640,
    zoom: int = 17,
    extent_m: float = 500.0,
    force: bool = False,
) -> dict[str, Any]:
    request = resolve_imagery_request(segment, cache_dir, width=width, height=height, zoom=zoom, extent_m=extent_m)
    if request.image_path.exists() and request.meta_path.exists() and not force:
        meta = json.loads(request.meta_path.read_text(encoding="utf-8"))
        meta["cache_hit"] = True
        meta["image_path"] = str(request.image_path)
        return meta

    if logger is not None:
        logger.info("Downloading satellite image for %s via %s", segment.get("segment_id"), request.provider)
    response = requests.get(
        request.url,
        headers={"User-Agent": "coastpilot/0.3.0 (PokeVisionProvider)"},
        timeout=240,
    )
    response.raise_for_status()
    request.image_path.write_bytes(response.content)
    meta = {
        "segment_id": segment.get("segment_id"),
        "provider": request.provider,
        "url": request.url,
        "image_path": str(request.image_path),
        "meta_path": str(request.meta_path),
        "midpoint_lon": request.midpoint_lon,
        "midpoint_lat": request.midpoint_lat,
        "width": request.width,
        "height": request.height,
        "zoom": request.zoom,
        "extent_m": request.extent_m,
        "bytes": len(response.content),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
    }
    request.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if logger is not None:
        logger.info("Cached satellite image at %s", request.image_path)
    return meta
