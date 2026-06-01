from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config import PipelineConfig


@dataclass(frozen=True)
class StepResult:
    name: str
    output_path: Path | None = None
    records: int | None = None
    skipped: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PipelineStep(Protocol):
    name: str

    def run(self, config: PipelineConfig, logger, force: bool = False) -> StepResult:
        ...
