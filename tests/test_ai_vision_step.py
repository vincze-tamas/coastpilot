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
    calls = []

    def classify_segment(self, segment, context=None):
        self.calls.append(segment.get('segment_id'))
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


    def test_strict_rock_threshold_demotes_low_confidence_rock(self):
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
                version='test',
                rock_min_confidence=0.90,
                prefer_rock_weight=1.0,
            )
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            cfg.data_dir.mkdir(parents=True, exist_ok=True)

            class Req: pass
            req=Req(); req.image_path=tmp_path/'x.jpg'; req.meta_path=tmp_path/'x.meta.json'
            req.image_path.write_bytes(b'x'); req.meta_path.write_text('{}')

            class RockAnalyzer(DummyAnalyzer):
                def classify_segment(self, segment, context=None):
                    from pipeline.ai.base import AIClassification
                    return AIClassification(label='rock', confidence=0.89, rationale='low confidence rock', provider='dummy', raw={'image_meta': {'cache_hit': True}})

            logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
            with patch('pipeline.steps.ai_vision.build_ai_analyzer', lambda *a, **k: RockAnalyzer()):
                with patch('pipeline.steps.ai_vision.resolve_imagery_request', lambda *a, **k: req):
                    AIVisionStep(batch_limit=10).run(cfg, logger)
            out = gpd.read_file(out_path)
            self.assertEqual(out.loc[0, 'ai_surface_class'], 'unknown')
            self.assertIn('rock_preference_score', out.columns)
            self.assertIn('rock_recommendation_band', out.columns)

    def test_target_segment_ids_restricts_ai_candidates(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            tmp_path = Path(td)
            gdf = gpd.GeoDataFrame([
                {'segment_id': 'a', 'needs_ai': True, 'ai_surface_class': None, 'ai_vision_version': None, 'geometry': Point(14.0,45.0)},
                {'segment_id': 'b', 'needs_ai': True, 'ai_surface_class': None, 'ai_vision_version': None, 'geometry': Point(14.1,45.1)},
            ], geometry='geometry', crs='EPSG:4326')
            in_path = tmp_path / 'segments.geojson'
            out_path = tmp_path / 'segments_classified.geojson'
            gdf.to_file(in_path, driver='GeoJSON')
            cfg = SimpleNamespace(
                classified_geojson=out_path,
                segments_geojson=in_path,
                cache_dir=tmp_path / 'cache',
                data_dir=tmp_path / 'data',
                version='test',
                rock_min_confidence=0.90,
                prefer_rock_weight=1.0,
            )
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            class Req: pass
            req=Req(); req.image_path=tmp_path/'x.jpg'; req.meta_path=tmp_path/'x.meta.json'
            req.image_path.write_bytes(b'x'); req.meta_path.write_text('{}')
            analyzer = DummyAnalyzer(); analyzer.calls = []
            logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
            with patch('pipeline.steps.ai_vision.build_ai_analyzer', lambda *a, **k: analyzer):
                with patch('pipeline.steps.ai_vision.resolve_imagery_request', lambda *a, **k: req):
                    res = AIVisionStep(batch_limit=10, target_segment_ids={'b'}).run(cfg, logger)
            self.assertEqual(analyzer.calls, ['b'])
            self.assertEqual(res.details['target_segment_ids'], 1)
            self.assertEqual(res.details['target_matched_segments'], 1)


    def test_stores_raw_label_before_strict_rock_demote(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            tmp_path = Path(td)
            gdf = gpd.GeoDataFrame([
                {'segment_id': 'a', 'needs_ai': True, 'ai_surface_class': None, 'ai_vision_version': None, 'geometry': Point(14.0,45.0)}
            ], geometry='geometry', crs='EPSG:4326')
            in_path = tmp_path / 'segments.geojson'
            out_path = tmp_path / 'segments_classified.geojson'
            gdf.to_file(in_path, driver='GeoJSON')
            cfg = SimpleNamespace(classified_geojson=out_path, segments_geojson=in_path, cache_dir=tmp_path / 'cache', data_dir=tmp_path / 'data', version='test', rock_min_confidence=0.90, prefer_rock_weight=1.0)
            cfg.cache_dir.mkdir(parents=True, exist_ok=True); cfg.data_dir.mkdir(parents=True, exist_ok=True)
            class Req: pass
            req=Req(); req.image_path=tmp_path/'x.jpg'; req.meta_path=tmp_path/'x.meta.json'
            req.image_path.write_bytes(b'x'); req.meta_path.write_text('{}')
            class RockAnalyzer(DummyAnalyzer):
                def classify_segment(self, segment, context=None):
                    from pipeline.ai.base import AIClassification
                    return AIClassification(label='rock', confidence=0.85, rationale='rocky but not certain', provider='dummy', raw={'image_meta': {'cache_hit': True}})
            logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
            with patch('pipeline.steps.ai_vision.build_ai_analyzer', lambda *a, **k: RockAnalyzer()):
                with patch('pipeline.steps.ai_vision.resolve_imagery_request', lambda *a, **k: req):
                    AIVisionStep(batch_limit=10).run(cfg, logger)
            out = gpd.read_file(out_path)
            self.assertEqual(out.loc[0, 'ai_surface_class'], 'unknown')
            self.assertEqual(out.loc[0, 'ai_raw_surface_class'], 'rock')
            self.assertAlmostEqual(float(out.loc[0, 'ai_raw_confidence']), 0.85)

    def test_reapplies_threshold_from_raw_label_when_no_model_calls(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            tmp_path = Path(td)
            gdf = gpd.GeoDataFrame([
                {'segment_id': 'a', 'needs_ai': False, 'ai_surface_class': 'unknown', 'ai_raw_surface_class': 'rock', 'ai_confidence': 0.85, 'ai_raw_confidence': 0.85, 'ai_vision_version': 'test', 'geometry': Point(14.0,45.0)}
            ], geometry='geometry', crs='EPSG:4326')
            out_path = tmp_path / 'segments_classified.geojson'
            gdf.to_file(out_path, driver='GeoJSON')
            cfg = SimpleNamespace(classified_geojson=out_path, segments_geojson=tmp_path / 'segments.geojson', cache_dir=tmp_path / 'cache', data_dir=tmp_path / 'data', version='test', rock_min_confidence=0.80, prefer_rock_weight=1.0)
            cfg.cache_dir.mkdir(parents=True, exist_ok=True); cfg.data_dir.mkdir(parents=True, exist_ok=True)
            logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
            with patch('pipeline.steps.ai_vision.build_ai_analyzer', lambda *a, **k: DummyAnalyzer()):
                res = AIVisionStep(batch_limit=10).run(cfg, logger, force=False)
            out = gpd.read_file(out_path)
            self.assertTrue(res.skipped)
            self.assertEqual(out.loc[0, 'ai_surface_class'], 'rock')


    def test_rethreshold_ignores_nan_raw_columns(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            tmp_path = Path(td)
            gdf = gpd.GeoDataFrame([
                {'segment_id': 'a', 'needs_ai': False, 'ai_surface_class': 'sand', 'ai_raw_surface_class': None, 'ai_confidence': 0.85, 'ai_raw_confidence': None, 'ai_vision_version': 'test', 'geometry': Point(14.0,45.0)},
                {'segment_id': 'b', 'needs_ai': False, 'ai_surface_class': 'unknown', 'ai_raw_surface_class': 'rock', 'ai_confidence': 0.85, 'ai_raw_confidence': 0.85, 'ai_vision_version': 'test', 'geometry': Point(14.1,45.1)},
            ], geometry='geometry', crs='EPSG:4326')
            out_path = tmp_path / 'segments_classified.geojson'
            gdf.to_file(out_path, driver='GeoJSON')
            cfg = SimpleNamespace(classified_geojson=out_path, segments_geojson=tmp_path / 'segments.geojson', cache_dir=tmp_path / 'cache', data_dir=tmp_path / 'data', version='test', rock_min_confidence=0.80, prefer_rock_weight=1.0)
            cfg.cache_dir.mkdir(parents=True, exist_ok=True); cfg.data_dir.mkdir(parents=True, exist_ok=True)
            logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
            with patch('pipeline.steps.ai_vision.build_ai_analyzer', lambda *a, **k: DummyAnalyzer()):
                AIVisionStep(batch_limit=10).run(cfg, logger, force=False)
            out = gpd.read_file(out_path)
            self.assertEqual(out.loc[0, 'ai_surface_class'], 'sand')


if __name__ == '__main__':
    unittest.main()
