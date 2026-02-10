from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


# Maps property name -> relative path segments from root
_ARTIFACT_DIRS: Dict[str, tuple] = {
    "canonical_daily": ("canonical", "daily"),
    "eligibility": ("eligibility",),
    "features": ("features",),
    "priors": ("priors",),
    "models_foundation": ("models", "foundation"),
    "models_ranker": ("models", "ranker"),
    "signals": ("signals",),
    "reports": ("reports",),
    "runs": ("runs",),
    "backtests": ("backtests",),
    "paper": ("paper",),
    "live": ("live",),
}


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    def _resolve(self, name: str) -> Path:
        return self.root.joinpath(*_ARTIFACT_DIRS[name])

    @property
    def canonical_daily(self) -> Path:
        return self._resolve("canonical_daily")

    @property
    def eligibility(self) -> Path:
        return self._resolve("eligibility")

    @property
    def features(self) -> Path:
        return self._resolve("features")

    @property
    def priors(self) -> Path:
        return self._resolve("priors")

    @property
    def models_foundation(self) -> Path:
        return self._resolve("models_foundation")

    @property
    def models_ranker(self) -> Path:
        return self._resolve("models_ranker")

    @property
    def signals(self) -> Path:
        return self._resolve("signals")

    @property
    def reports(self) -> Path:
        return self._resolve("reports")

    @property
    def runs(self) -> Path:
        return self._resolve("runs")

    @property
    def backtests(self) -> Path:
        return self._resolve("backtests")

    @property
    def paper(self) -> Path:
        return self._resolve("paper")

    @property
    def live(self) -> Path:
        return self._resolve("live")


def ensure_artifact_dirs(paths: ArtifactPaths) -> None:
    for name in _ARTIFACT_DIRS:
        paths._resolve(name).mkdir(parents=True, exist_ok=True)
