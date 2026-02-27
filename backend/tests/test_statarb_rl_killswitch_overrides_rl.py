from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.models.itransformer.types import ITransformerSignalResponse
from backend.app.strategies.statarb.sleeve import StatArbSleeve


class _Client:
    def itransformer_signal(self, req, critical=True):
        return ITransformerSignalResponse(model_version="i1", scores={"A": 1.0, "B": -1.0}, uncertainty=0.2, correlation_regime=9.0, latency_ms=1.0)


def test_statarb_rl_killswitch_overrides_rl():
    out = StatArbSleeve(_Client(), cfg=MLPlatformConfig(enable_rl_overlay_statarb=True)).generate_targets("2026-01-01", ["A", "B"], [[1], [2]], "t")
    assert out["status"] == "halted"
