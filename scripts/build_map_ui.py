#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import geopandas as gpd


def _hex(label: str) -> str:
    return {
        'rock': '#2563eb',
        'pebble': '#6b7280',
        'sand': '#f59e0b',
        'built': '#ef4444',
        'unknown': '#111827',
    }.get(label, '#111827')


def main() -> int:
    p = argparse.ArgumentParser(description='Build standalone filterable map HTML from classified geojson')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    gdf = gpd.read_file(args.input).to_crs(epsg=4326)
    if gdf.empty:
        raise SystemExit('Input geojson is empty')
    if 'ai_surface_class' not in gdf.columns:
        gdf['ai_surface_class'] = 'unknown'
    if 'ai_confidence' not in gdf.columns:
        gdf['ai_confidence'] = 0.0
    if 'recommendation_score' not in gdf.columns:
        if 'rock_preference_score' in gdf.columns:
            gdf['recommendation_score'] = gdf['rock_preference_score']
        else:
            gdf['recommendation_score'] = 0.0

    features = []
    for _, r in gdf.iterrows():
        if r.geometry is None:
            continue
        label = str(r.get('ai_surface_class') or 'unknown').lower()
        conf = float(r.get('ai_confidence') or 0.0)
        score = float(r.get('recommendation_score') or 0.0)
        features.append({
            'type': 'Feature',
            'geometry': r.geometry.__geo_interface__,
            'properties': {
                'segment_id': r.get('segment_id'),
                'label': label,
                'confidence': conf,
                'recommendation_score': score,
                'color': _hex(label),
            },
        })

    fc = {'type': 'FeatureCollection', 'features': features}

    html = """<!doctype html>
<html><head><meta charset='utf-8'><title>CoastPilot Map</title>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<style>html,body,#map{height:100%;margin:0}.panel{position:absolute;top:10px;left:10px;z-index:999;background:#fff;padding:10px;border-radius:8px;font-family:Arial}</style>
</head><body>
<div class='panel'>
<label>Label <select id='label'><option value='all'>all</option><option>rock</option><option>pebble</option><option>cobble</option><option>sand</option><option>built</option><option>unknown</option></select></label><br/>
<label>Min confidence <input id='conf' type='range' min='0' max='1' step='0.05' value='0.5'/><span id='confv'>0.5</span></label><br/>
<label>Min score <input id='score' type='range' min='0' max='1' step='0.05' value='0.5'/><span id='scorev'>0.5</span></label>
</div>
<div id='map'></div>
<script>
const geojson = __GEOJSON__;
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(map);
let layer;
function redraw(){
 const label=document.getElementById('label').value;
 const conf=parseFloat(document.getElementById('conf').value);
 const score=parseFloat(document.getElementById('score').value);
 document.getElementById('confv').textContent=conf.toFixed(2);
 document.getElementById('scorev').textContent=score.toFixed(2);
 const filtered={type:'FeatureCollection',features:geojson.features.filter(function(f){
   const p=f.properties||{};
   const okLabel=(label==='all'||p.label===label);
   return okLabel && Number(p.confidence||0)>=conf && Number(p.recommendation_score||0)>=score;
 })};
 if(layer) map.removeLayer(layer);
 layer=L.geoJSON(filtered,{style:function(f){return {color:f.properties.color,weight:3};},onEachFeature:function(f,l){
   const p=f.properties||{};
   l.bindPopup('ID: '+(p.segment_id||'n/a')+'<br>label: '+(p.label||'n/a')+'<br>conf: '+Number(p.confidence||0).toFixed(2)+'<br>score: '+Number(p.recommendation_score||0).toFixed(2));
 }}).addTo(map);
 if(layer.getLayers().length){ map.fitBounds(layer.getBounds().pad(0.1)); }
}
['label','conf','score'].forEach(function(id){document.getElementById(id).addEventListener('input',redraw);});
redraw();
</script></body></html>"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace('__GEOJSON__', json.dumps(fc)), encoding='utf-8')
    print(f'Wrote map HTML to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
