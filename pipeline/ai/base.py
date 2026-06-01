from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AIClassification:
    label: str | None
    confidence: float | None = None
    rationale: str | None = None
    provider: str = "noop"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_human_review(self) -> bool:
        return self.label is None


@runtime_checkable
class AIAnalyzer(Protocol):
    name: str

    def classify_segment(self, segment: dict[str, Any], context: dict[str, Any] | None = None) -> AIClassification:
        ...

    def classify_batch(
        self, segments: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> list[AIClassification]:
        ...
