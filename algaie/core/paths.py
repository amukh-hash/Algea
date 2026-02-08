from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def canonical_daily(self) -> Path:
        return self.root / "canonical" / "daily"

    @property
    def eligibility(self) -> Path:
        return self.root / "eligibility"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def priors(self) -> Path:
        return self.root / "priors"

    @property
    def models_foundation(self) -> Path:
        return self.root / "models" / "foundation"

    @property
    def models_ranker(self) -> Path:
        return self.root / "models" / "ranker"

    @property
    def signals(self) -> Path:
        return self.root / "signals"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def backtests(self) -> Path:
        return self.root / "backtests"

    @property
    def paper(self) -> Path:
        return self.root / "paper"

    @property
    def live(self) -> Path:
        return self.root / "live"


def ensure_artifact_dirs(paths: ArtifactPaths) -> None:
    for path in [
        paths.canonical_daily,
        paths.eligibility,
        paths.features,
        paths.priors,
        paths.models_foundation,
        paths.models_ranker,
        paths.signals,
        paths.reports,
        paths.runs,
        paths.backtests,
        paths.paper,
        paths.live,
    ]:
        path.mkdir(parents=True, exist_ok=True)
