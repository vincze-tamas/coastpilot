#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd


def main() -> int:
    p = argparse.ArgumentParser(description='Build annotation sample CSV from classified GeoJSON')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--size', type=int, default=400)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    gdf = gpd.read_file(args.input)
    if gdf.empty:
        raise SystemExit('No data in input geojson')

    cols = ['segment_id', 'ai_surface_class', 'ai_confidence', 'vision_image_url', 'vision_image_path', 'geometry']
    for c in cols:
        if c not in gdf.columns:
            gdf[c] = None

    sample = gdf[cols].copy()
    groups = []
    classes = ['rock', 'pebble', 'cobble', 'sand', 'built', 'unknown']
    per = max(1, args.size // max(1, len(classes)))
    for c in classes:
        subset = sample[sample['ai_surface_class'] == c]
        if len(subset) > 0:
            groups.append(subset.sample(n=min(per, len(subset)), random_state=args.seed))
    combined = pd.concat(groups, ignore_index=True) if groups else sample.sample(n=min(args.size, len(sample)), random_state=args.seed)
    if len(combined) < args.size and len(sample) > len(combined):
        remaining = sample.drop(index=combined.index, errors='ignore')
        need = min(args.size - len(combined), len(remaining))
        combined = pd.concat([combined, remaining.sample(n=need, random_state=args.seed)], ignore_index=True)

    combined['gold_label'] = ''
    combined['gold_confidence'] = ''
    combined['reviewer_note'] = ''
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f'Wrote {len(combined)} rows to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
