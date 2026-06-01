from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from .config import PipelineConfig
from .osm import load_geojson, save_geojson

PROJECTED_CRS = "EPSG:32633"


def _iter_line_parts(geometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    try:
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return [merged]
        if isinstance(merged, MultiLineString):
            return list(merged.geoms)
    except Exception:
        pass
    return []


def _segment_line(line: LineString, segment_length_m: float) -> List[LineString]:
    length = float(line.length)
    if length == 0:
        return []
    if length <= segment_length_m:
        return [line]
    distances = [0.0]
    d = segment_length_m
    while d < length:
        distances.append(d)
        d += segment_length_m
    distances.append(length)
    points = [line.interpolate(dist) for dist in distances]
    segments: List[LineString] = []
    for idx in range(len(points) - 1):
        seg = LineString([points[idx], points[idx + 1]])
        if seg.length > 0:
            segments.append(seg)
    return segments


def build_segments(
    raw_path: Path,
    output_path: Path,
    config: PipelineConfig,
    logger,
    force: bool = False,
) -> gpd.GeoDataFrame:
    if output_path.exists() and not force:
        logger.info("Segments already exist, skipping segmentation: %s", output_path)
        return load_geojson(output_path)

    logger.info("Loading raw OSM data from %s", raw_path)
    raw = load_geojson(raw_path)
    coastline = raw[raw["natural"] == "coastline"].copy()
    if coastline.empty:
        raise RuntimeError("No coastline features found in raw data")

    coastline = coastline.to_crs(PROJECTED_CRS)
    records = []
    counter = 1
    for _, row in coastline.iterrows():
        src_id = row.get("osm_id")
        source_geom = row.geometry
        for part in _iter_line_parts(source_geom):
            for seg in _segment_line(part, config.segment_length_m):
                records.append(
                    {
                        "segment_id": f"seg_{counter:06d}",
                        "source_osm_id": src_id,
                        "segment_length_m": float(seg.length),
                        "pipeline_version": config.version,
                        "geometry": seg,
                    }
                )
                counter += 1

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=PROJECTED_CRS)
    gdf = gdf.to_crs("EPSG:4326")
    save_geojson(gdf, output_path)
    logger.info("Wrote %s segments to %s", len(gdf), output_path)
    return gdf
