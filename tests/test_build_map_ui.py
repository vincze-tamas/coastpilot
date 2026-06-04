import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point


class TestBuildMapUI(unittest.TestCase):
    def test_uses_rock_preference_score_when_recommendation_score_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / 'in.geojson'
            out = root / 'map.html'
            gdf = gpd.GeoDataFrame([
                {'segment_id': 'seg_1', 'ai_surface_class': 'rock', 'ai_confidence': 0.9, 'rock_preference_score': 0.9, 'geometry': Point(14, 45)},
            ], geometry='geometry', crs='EPSG:4326')
            gdf.to_file(inp, driver='GeoJSON')
            proc = subprocess.run([sys.executable, 'scripts/build_map_ui.py', '--input', str(inp), '--output', str(out)], text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            html = out.read_text(encoding='utf-8')
            self.assertIn('"recommendation_score": 0.9', html)
            self.assertIn('<option>cobble</option>', html)


if __name__ == '__main__':
    unittest.main()
