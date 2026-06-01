from __future__ import annotations

from dataclasses import dataclass

from ..config import PipelineConfig
from ..segmentation import build_segments
from .base import StepResult


@dataclass
class SegmentationStep:
    name: str = "segmentation"

    def run(self, config: PipelineConfig, logger, force: bool = False) -> StepResult:
        segments = build_segments(config.raw_geojson, config.segments_geojson, config, logger, force=force)
        return StepResult(name=self.name, output_path=config.segments_geojson, records=int(len(segments)))
