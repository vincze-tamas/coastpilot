import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.build_error_review_queue import build_queue


class TestBuildErrorReviewQueue(unittest.TestCase):
    def test_builds_only_strict_rock_errors_and_missing_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gold = root / "gold.csv"
            pred = root / "pred.geojson"
            out = root / "queue.csv"
            pd.DataFrame([
                {"segment_id": "tp", "gold_label": "rock", "vision_image_path": "tp.jpg"},
                {"segment_id": "fp", "gold_label": "sand", "vision_image_path": "fp.jpg"},
                {"segment_id": "fn", "gold_label": "rock", "vision_image_path": "fn.jpg"},
                {"segment_id": "missing", "gold_label": "rock", "vision_image_path": "missing.jpg"},
                {"segment_id": "tn", "gold_label": "built", "vision_image_path": "tn.jpg"},
            ]).to_csv(gold, index=False)
            gpd.GeoDataFrame([
                {"segment_id": "tp", "ai_surface_class": "rock", "ai_confidence": 0.9, "geometry": Point(0, 0)},
                {"segment_id": "fp", "ai_surface_class": "rock", "ai_confidence": 0.8, "geometry": Point(1, 1)},
                {"segment_id": "fn", "ai_surface_class": "sand", "ai_confidence": 0.7, "geometry": Point(2, 2)},
                {"segment_id": "tn", "ai_surface_class": "built", "ai_confidence": 0.6, "geometry": Point(3, 3)},
            ], geometry="geometry", crs="EPSG:4326").to_file(pred, driver="GeoJSON")

            summary = build_queue(gold, pred, out)

            self.assertEqual(summary["rows"], 3)
            queue = pd.read_csv(out)
            self.assertEqual(queue["segment_id"].tolist(), ["fp", "fn", "missing"])
            self.assertEqual(queue["review_error_type"].tolist(), ["false_positive_rock", "false_negative_rock", "missing_prediction"])
            self.assertIn("review_label", queue.columns)
            self.assertTrue(queue["review_label"].fillna("").eq("").all())
            self.assertEqual(queue.loc[0, "ai_surface_class"], "rock")
            self.assertEqual(queue.loc[1, "gold_label"], "rock")


if __name__ == "__main__":
    unittest.main()
