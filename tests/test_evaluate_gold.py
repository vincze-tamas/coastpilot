import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point


class TestEvaluateGold(unittest.TestCase):
    def run_eval(self, csv_text: str):
        with tempfile.TemporaryDirectory() as td:
            gold = Path(td) / "gold.csv"
            out = Path(td) / "out.json"
            gold.write_text(csv_text, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "scripts/evaluate_gold.py", "--gold-csv", str(gold), "--output-json", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )
            out_text = out.read_text(encoding="utf-8") if out.exists() else ""
            return proc, out_text

    def test_pebble_and_cobble_are_strict_rock_negatives(self):
        proc, out_text = self.run_eval(
            "segment_id,ai_surface_class,gold_label\n"
            "1,rock,rock\n"
            "2,rock,pebble\n"
            "3,pebble,cobble\n"
            "4,cobble,rock\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(out_text)
        self.assertEqual(data["strict_rock"]["tp"], 1)
        self.assertEqual(data["strict_rock"]["fp"], 1)
        self.assertEqual(data["strict_rock"]["fn"], 1)
        self.assertEqual(data["strict_rock"]["tn"], 1)
        self.assertIn("pebble/cobble are NOT rocky", data["strict_rock"]["definition"])

    def test_invalid_gold_label_fails(self):
        proc, _ = self.run_eval("segment_id,ai_surface_class,gold_label\n1,rock,gravel\n")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid gold labels", proc.stderr + proc.stdout)

    def test_predictions_geojson_joins_by_exact_segment_id_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gold = root / "gold.csv"
            pred = root / "pred.geojson"
            out = root / "out.json"
            gold.write_text(
                "segment_id,ai_surface_class,gold_label\n"
                "seg_a,unknown,rock\n"
                "seg_b,unknown,sand\n"
                "seg_c,unknown,rock\n",
                encoding="utf-8",
            )
            gdf = gpd.GeoDataFrame(
                [
                    {"segment_id": "seg_a", "ai_surface_class": "rock", "ai_confidence": 0.95, "geometry": Point(0, 0)},
                    {"segment_id": "seg_b", "ai_surface_class": "sand", "ai_confidence": 0.95, "geometry": Point(1, 1)},
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )
            gdf.to_file(pred, driver="GeoJSON")
            proc = subprocess.run(
                [sys.executable, "scripts/evaluate_gold.py", "--gold-csv", str(gold), "--predictions-geojson", str(pred), "--output-json", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["strict_rock"]["tp"], 1)
            self.assertEqual(data["strict_rock"]["fn"], 1)
            self.assertEqual(data["prediction_coverage"]["usable_predictions"], 2)
            self.assertEqual(data["prediction_coverage"]["missing_or_unknown_predictions"], 1)

            proc_fail = subprocess.run(
                [sys.executable, "scripts/evaluate_gold.py", "--gold-csv", str(gold), "--predictions-geojson", str(pred), "--fail-on-missing-predictions", "--output-json", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(proc_fail.returncode, 0)
            self.assertIn("Missing usable predictions", proc_fail.stderr + proc_fail.stdout)


if __name__ == "__main__":
    unittest.main()
