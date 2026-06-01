import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import Point

from pipeline.steps.ai_vision import AIVisionStep


class DummyAnalyzer:
    name = 'dummy'
    image_cache_dir = Path('/tmp')
    model = 'dummy'

    def classify_segment(self, segment, context=None):
        from pipeline.ai.base import AIClassification
        return AIClassification(label='sand', confidence=0.9, rationale='ok', provider='dummy', raw={'image_meta': {'cache_hit': True}})


class TestAIVisionStep(unittest.TestCase):
    def test_cached_candidate_not_skipped(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            tmp_path = Path(td)
            gdf = gpd.GeoDataFrame([
                {'segment_id': 'a', 'needs_ai': True, 'ai_surface_class': None, 'ai_vision_version': None, 'geometry': Point(14.0,45.0)}
            ], geometry='geometry', crs='EPSG:4326')
            in_path = tmp_path / 'segments.geojson'
            out_path = tmp_path / 'segments_classified.geojson'
            gdf.to_file(in_path, driver='GeoJSON')

            cfg = SimpleNamespace(
                classified_geojson=out_path,
                segments_geojson=in_path,
                cache_dir=tmp_path / 'cache',
                data_dir=tmp_path / 'data',
                version='test'
            )
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            cfg.data_dir.mkdir(parents=True, exist_ok=True)

            class Req: pass
            req=Req(); req.image_path=tmp_path/'x.jpg'; req.meta_path=tmp_path/'x.meta.json'
            req.image_path.write_bytes(b'x'); req.meta_path.write_text('{}')

            logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
            with patch('pipeline.steps.ai_vision.build_ai_analyzer', lambda *a, **k: DummyAnalyzer()):
                with patch('pipeline.steps.ai_vision.resolve_imagery_request', lambda *a, **k: req):
                    res = AIVisionStep(batch_limit=10).run(cfg, logger)
            self.assertEqual(res.details['processed_segments'], 1)


if __name__ == '__main__':
    unittest.main()
