from __future__ import annotations

from pathlib import Path
import json

from ..osm import load_geojson


def build_quality_metrics(classified_geojson: Path, output_path: Path) -> dict:
    gdf = load_geojson(classified_geojson)
    total = int(len(gdf))
    labels = gdf.get('ai_surface_class')
    conf = gdf.get('ai_confidence')

    def _count(mask):
        return int(mask.sum()) if hasattr(mask, 'sum') else 0

    if labels is None:
        metrics = {
            'total_segments': total,
            'label_coverage_pct': 0.0,
            'unknown_ratio': 1.0,
            'class_distribution': {},
            'confidence_bands': {'low': 0, 'mid': 0, 'high': 0},
        }
    else:
        valid = labels.notna()
        valid = valid.logical_and(labels != '')
        known = valid.logical_and(labels != 'unknown')
        coverage = (_count(known) / total * 100.0) if total else 0.0
        class_dist = {}
        for c in ['sand', 'pebble', 'rock', 'built', 'unknown']:
            class_dist[c] = int((labels == c).sum())
        low = mid = high = 0
        if conf is not None:
            low = int(((conf < 0.5).fillna(False)).sum())
            mid = int((((conf >= 0.5).logical_and(conf < 0.8)).fillna(False)).sum())
            high = int(((conf >= 0.8).fillna(False)).sum())
        metrics = {
            'total_segments': total,
            'label_coverage_pct': round(coverage, 3),
            'unknown_ratio': round((class_dist.get('unknown', 0) / total), 4) if total else 0.0,
            'class_distribution': class_dist,
            'confidence_bands': {'low': low, 'mid': mid, 'high': high},
        }

    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    return metrics
