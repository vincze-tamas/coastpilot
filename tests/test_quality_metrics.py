import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.qa.metrics import build_quality_metrics


class TestQualityMetrics(unittest.TestCase):
    def assert_metrics(self, frame, expected):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "bemenet.geojson"
            output = Path(directory) / "minőség.json"
            with patch("pipeline.qa.metrics.load_geojson", return_value=frame) as load:
                metrics = build_quality_metrics(source, output)
            load.assert_called_once_with(source)
            self.assertEqual(metrics, expected)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), metrics)

    def expected(self, total, coverage=0.0, unknown=0.0, counts=None, bands=None):
        return {
            "total_segments": total,
            "label_coverage_pct": coverage,
            "unknown_ratio": unknown,
            "class_distribution": counts if counts is not None else {
                "sand": 0, "pebble": 0, "cobble": 0,
                "rock": 0, "built": 0, "unknown": 0,
            },
            "confidence_bands": bands or {"low": 0, "mid": 0, "high": 0},
        }

    def test_all_six_classes_blank_and_null_labels(self):
        frame = pd.DataFrame({
            "ai_surface_class": [
                "sand", "pebble", "cobble", "rock", "built", "unknown", "", None, pd.NA,
            ],
        })
        self.assert_metrics(frame, self.expected(
            9, coverage=55.556, unknown=0.1111,
            counts={"sand": 1, "pebble": 1, "cobble": 1,
                    "rock": 1, "built": 1, "unknown": 1},
        ))

    def test_confidence_boundaries_and_missing_values(self):
        values = [0.0, 0.4999, 0.5, 0.7999, 0.8, 1.0, None, float("nan")]
        for dtype in ("float64", "Float64"):
            with self.subTest(dtype=dtype):
                frame = pd.DataFrame({
                    "ai_surface_class": ["unknown"] * len(values),
                    "ai_confidence": pd.Series(values, dtype=dtype),
                })
                self.assert_metrics(frame, self.expected(
                    8, unknown=1.0,
                    counts={"sand": 0, "pebble": 0, "cobble": 0,
                            "rock": 0, "built": 0, "unknown": 8},
                    bands={"low": 2, "mid": 2, "high": 2},
                ))

    def test_confidence_counts_include_blank_and_null_labels(self):
        frame = pd.DataFrame({
            "ai_surface_class": ["", None, pd.NA],
            "ai_confidence": [0.4, 0.5, 0.8],
        })
        self.assert_metrics(frame, self.expected(
            3, bands={"low": 1, "mid": 1, "high": 1},
        ))

    def test_missing_confidence_column(self):
        frame = pd.DataFrame({"ai_surface_class": ["cobble", "cobble"]})
        self.assert_metrics(frame, self.expected(
            2, coverage=100.0,
            counts={"sand": 0, "pebble": 0, "cobble": 2,
                    "rock": 0, "built": 0, "unknown": 0},
        ))

    def test_missing_label_column_ignores_confidence(self):
        frame = pd.DataFrame({"ai_confidence": [0.4, 0.5, 0.8]})
        self.assert_metrics(frame, self.expected(3, unknown=1.0, counts={}))

    def test_missing_both_columns(self):
        frame = pd.DataFrame(index=range(2))
        self.assert_metrics(frame, self.expected(2, unknown=1.0, counts={}))

    def test_empty_dataframe_without_columns(self):
        self.assert_metrics(pd.DataFrame(), self.expected(0, unknown=1.0, counts={}))

    def test_empty_dataframe_with_columns(self):
        frame = pd.DataFrame({
            "ai_surface_class": pd.Series(dtype="object"),
            "ai_confidence": pd.Series(dtype="float64"),
        })
        self.assert_metrics(frame, self.expected(0))


if __name__ == "__main__":
    unittest.main()
