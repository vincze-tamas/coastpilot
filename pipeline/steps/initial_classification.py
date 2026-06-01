from __future__ import annotations

from dataclasses import dataclass

from ..classification import classify_segments
from ..config import PipelineConfig
from .base import StepResult


@dataclass
class InitialClassificationStep:
    name: str = "initial_classification"

    def run(self, config: PipelineConfig, logger, force: bool = False) -> StepResult:
        classified = classify_segments(config.raw_geojson, config.segments_geojson, config.classified_geojson, config, logger, force=force)
        return StepResult(name=self.name, output_path=config.classified_geojson, records=int(len(classified)))
