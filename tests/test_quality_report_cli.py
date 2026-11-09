import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_pipeline
from pipeline.config import PipelineConfig


class TestQualityReportCLI(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = PipelineConfig(project_root=self.root)
        self.report = self.root / "quality.json"
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(patch("run_pipeline._config_from_args", return_value=self.config))
        self.stack.enter_context(patch("run_pipeline.setup_logging", return_value=Mock()))
        self.metrics = self.stack.enter_context(patch(
            "run_pipeline.build_quality_metrics", wraps=run_pipeline.build_quality_metrics,
        ))
        for name in ("OSMDataCollectionStep", "SegmentationStep", "InitialClassificationStep",
                     "EUEnrichmentStep", "AIVisionStep"):
            step = self.stack.enter_context(patch(f"run_pipeline.{name}"))
            step.return_value.run.return_value = SimpleNamespace(
                name=name, output_path=self.config.classified_geojson,
                records=2, skipped=False, details={},
            )
            if name == "InitialClassificationStep":
                step.return_value.run.side_effect = self.write_classified
                self.classification = step.return_value.run

    def write_classified(self, *args, **kwargs):
        self.config.classified_geojson.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": None, "properties": {
                    "ai_surface_class": label, "ai_confidence": confidence,
                }} for label, confidence in (("rock", 0.9), ("unknown", 0.4))
            ],
        }), encoding="utf-8")
        return SimpleNamespace(
            name="initial_classification", output_path=self.config.classified_geojson,
            records=2, skipped=False, details={},
        )

    def run_cli(self, *extra):
        with patch("sys.argv", ["run_pipeline.py", "--step", "classify", *extra]):
            return run_pipeline.main()

    def test_quality_report_writes_metrics_from_classified_data(self):
        self.assertEqual(self.run_cli("--quality-report", str(self.report)), 0)
        self.metrics.assert_called_once_with(self.config.classified_geojson, self.report)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report, {
            "total_segments": 2,
            "label_coverage_pct": 50.0,
            "unknown_ratio": 0.5,
            "class_distribution": {"sand": 0, "pebble": 0, "cobble": 0,
                                   "rock": 1, "built": 0, "unknown": 1},
            "confidence_bands": {"low": 1, "mid": 0, "high": 1},
        })
        self.assertTrue(self.config.manifest_path.exists())

    def test_without_flag_does_not_generate_quality_report(self):
        self.assertEqual(self.run_cli(), 0)
        self.metrics.assert_not_called()
        self.assertFalse(self.report.exists())
        self.assertEqual(set(self.config.data_dir.iterdir()), {
            self.config.classified_geojson, self.config.manifest_path,
        })

    def test_failed_pipeline_does_not_generate_quality_report(self):
        self.classification.side_effect = RuntimeError("classification failed")
        with self.assertRaisesRegex(RuntimeError, "classification failed"):
            self.run_cli("--quality-report", str(self.report))
        self.metrics.assert_not_called()
        self.assertFalse(self.report.exists())


if __name__ == "__main__":
    unittest.main()
