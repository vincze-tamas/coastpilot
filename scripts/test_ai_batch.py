from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from statistics import mean
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd

from pipeline.ai.factory import build_ai_analyzer
from pipeline.config import PipelineConfig
from pipeline.osm import load_geojson, save_geojson
from pipeline.vision.imagery import resolve_imagery_request


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("coastpilot_test_ai")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def build_payload(row) -> dict:
    geom = row.geometry
    centroid = geom.centroid if geom is not None and not geom.is_empty else None
    bounds = geom.bounds if geom is not None and not geom.is_empty else None
    return {
        "segment_id": row.segment_id,
        "surface_class": getattr(row, "surface_class", None),
        "matched_beach_osm_id": getattr(row, "matched_beach_osm_id", None),
        "source_osm_id": getattr(row, "source_osm_id", None),
        "needs_ai": bool(getattr(row, "needs_ai", True)),
        "segment_length_m": getattr(row, "segment_length_m", None),
        "centroid_lat": centroid.y if centroid is not None else None,
        "centroid_lon": centroid.x if centroid is not None else None,
        "bounds": bounds,
        "geometry": geom,
    }


def chunks(items, size):
    for idx in range(0, len(items), size):
        yield items[idx: idx + size]


def line_points(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        xs, ys = geom.xy
        return list(zip(xs, ys))
    if geom.geom_type == "MultiLineString":
        pts = []
        for part in geom.geoms:
            xs, ys = part.xy
            pts.extend(list(zip(xs, ys)))
        return pts
    return []


def sample_contact_sheet_svg(samples: gpd.GeoDataFrame, output_path: Path) -> None:
    cols = 4
    panel_w = 260
    panel_h = 180
    rows = (len(samples) + cols - 1) // cols
    width = cols * panel_w
    height = rows * panel_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#f3f4f6"/>']

    for i, (_, row) in enumerate(samples.iterrows()):
        x0 = (i % cols) * panel_w
        y0 = (i // cols) * panel_h
        geom = row.geometry
        label = row.ai_surface_class or "unknown"
        conf = "n/a" if row.ai_confidence is None else f"{float(row.ai_confidence):.2f}"
        pts = line_points(geom)
        if geom is not None and not geom.is_empty:
            minx, miny, maxx, maxy = geom.bounds
            dx = (maxx - minx) * 0.2 or 0.001
            dy = (maxy - miny) * 0.2 or 0.001
            minx -= dx
            miny -= dy
            maxx += dx
            maxy += dy
        else:
            minx = miny = 0.0
            maxx = maxy = 1.0
        scaled = []
        for x, y in pts:
            sx = 18 + ((x - minx) / (maxx - minx) if maxx != minx else 0.5) * (panel_w - 36)
            sy = 38 + ((maxy - y) / (maxy - miny) if maxy != miny else 0.5) * (panel_h - 62)
            scaled.append(f"{sx:.1f},{sy:.1f}")
        parts.append(f'<g transform="translate({x0},{y0})">')
        parts.append(f'<rect x="6" y="6" width="{panel_w-12}" height="{panel_h-12}" rx="10" fill="#ffffff" stroke="#d1d5db" stroke-width="1.5"/>')
        parts.append(f'<rect x="18" y="28" width="{panel_w-36}" height="{panel_h-54}" rx="6" fill="#f8fafc" stroke="#e5e7eb"/>')
        if scaled:
            parts.append(f'<polyline points="{" ".join(scaled)}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        parts.append(f'<text x="18" y="20" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#111827">{escape(str(row.segment_id))}</text>')
        parts.append(f'<text x="18" y="{panel_h-18}" font-family="Arial, sans-serif" font-size="11" fill="#374151">{escape(str(label))}</text>')
        parts.append(f'<text x="{panel_w-18}" y="{panel_h-18}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">conf {escape(conf)}</text>')
        parts.append('</g>')

    parts.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(parts), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of segments to process in this run")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig(ai_provider="poke", ai_timeout_seconds=600.0)
    logger = setup_logger(config.logs_dir / "test_ai_batch.log")
    logger.info("Starting test AI batch: batch_size=%s sample_count=%s seed=%s", args.batch_size, args.sample_count, args.seed)

    classified = load_geojson(config.classified_geojson)
    needs = classified[classified["needs_ai"] == True].copy()  # noqa: E712
    selected_rows = []
    cached_skipped = 0
    for row in needs.itertuples():
        request = resolve_imagery_request(row._asdict(), config.cache_dir)
        if request.image_path.exists() and request.meta_path.exists():
            cached_skipped += 1
            continue
        selected_rows.append(row)
        if len(selected_rows) >= args.limit:
            break
    if not selected_rows:
        raise RuntimeError("No needs_ai segments available after skipping cached imagery")

    analyzer = build_ai_analyzer(config)
    logger.info("Using AI analyzer provider=%s model=%s", getattr(analyzer, "name", "unknown"), getattr(analyzer, "model", None))

    payloads = []
    rows = []
    for row in selected_rows:
        payloads.append(build_payload(row))
        rows.append(row)

    results = []
    for batch_index, (payload_batch, row_batch) in enumerate(zip(chunks(payloads, args.batch_size), chunks(rows, args.batch_size)), start=1):
        logger.info("Classifying batch %s (%s segments)", batch_index, len(payload_batch))
        batch_results = analyzer.classify_batch(payload_batch, context={"workflow": "coastpilot_test_batch", "batch_index": batch_index, "batch_size": args.batch_size})
        for row, res in zip(row_batch, batch_results):
            results.append({
                "segment_id": row.segment_id,
                "source_osm_id": row.source_osm_id,
                "segment_length_m": row.segment_length_m,
                "surface_class": getattr(row, "surface_class", None),
                "matched_beach_osm_id": getattr(row, "matched_beach_osm_id", None),
                "needs_ai_original": bool(row.needs_ai),
                "ai_surface_class": res.label,
                "ai_confidence": res.confidence,
                "ai_rationale": res.rationale,
                "ai_provider": res.provider,
                "ai_needs_review": res.needs_human_review,
                "geometry": row.geometry,
            })

    out_gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=classified.crs)
    output_path = config.data_dir / "test_ai_output.geojson"
    save_geojson(out_gdf, output_path)

    total = len(out_gdf)
    label_counts = Counter((row.ai_surface_class or "unknown") for row in out_gdf.itertuples())
    conf_values = [float(row.ai_confidence) for row in out_gdf.itertuples() if row.ai_confidence is not None]
    avg_conf = round(mean(conf_values), 4) if conf_values else None
    review_count = sum(1 for row in out_gdf.itertuples() if row.ai_needs_review)
    review_pct = round(review_count / total * 100, 2) if total else 0.0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": getattr(analyzer, "name", "unknown"),
        "model": getattr(analyzer, "model", None),
        "total_segments": total,
        "cached_skipped": cached_skipped,
        "label_distribution": dict(label_counts),
        "average_confidence": avg_conf,
        "needs_review_count": review_count,
        "needs_review_pct": review_pct,
    }
    summary_path = config.data_dir / "test_ai_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding='utf-8')

    rng = random.Random(args.seed)
    sample_indices = rng.sample(list(out_gdf.index), k=min(args.sample_count, len(out_gdf)))
    samples = out_gdf.loc[sample_indices].copy()
    samples_path = config.data_dir / "test_ai_samples.svg"
    sample_contact_sheet_svg(samples, samples_path)

    with config.log_path.open('a', encoding='utf-8') as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} | TEST_AI | {json.dumps(jsonable(summary), ensure_ascii=False)}\n")

    print(json.dumps({
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "samples_path": str(samples_path),
        "summary": summary,
        "sample_segment_ids": [row.segment_id for row in samples.itertuples()],
        "cached_skipped": cached_skipped,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
