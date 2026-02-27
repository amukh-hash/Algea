from __future__ import annotations

from ...ml_platform.models.chronos2.types import TSFMRequest
from ...ml_platform.inference_gateway.client import InferenceGatewayClient
from .decision import should_trade
from .features import build_series_features
from .risk import sleeve_risk_payload


class FuturesOvernightSleeve:
    def __init__(self, client: InferenceGatewayClient, enabled: bool = True):
        self.client = client
        self.enabled = enabled

    def generate_targets(self, instrument_id: str, prices: list[float], trace_id: str, asof: str) -> dict:
        if not self.enabled:
            return {"status": "disabled", "targets": []}
        series = build_series_features(prices, context_length=min(len(prices), 32))
        req = TSFMRequest(
            asof=asof,
            series=series,
            freq="1d",
            prediction_length=3,
            instrument_id=instrument_id,
            trace_id=trace_id,
        )
        resp = self.client.chronos2_forecast(req, critical=True)
        assert resp is not None
        median = resp.forecast.get("0.50", [])
        if not should_trade(median, resp.uncertainty.get("iqr_mean", 999.0)):
            return {
                "status": "halted",
                "reason": "uncertainty_or_edge",
                "targets": [],
                "ml_risk": sleeve_risk_payload(resp.model_version, resp.uncertainty.get("iqr_mean", 0.0), resp.latency_ms, resp.ood_score or 0.0),
            }
        direction = 1.0 if (median[-1] - median[0]) > 0 else -1.0
        return {
            "status": "ok",
            "targets": [{"symbol": instrument_id, "target_weight": 0.02 * direction}],
            "ml_risk": sleeve_risk_payload(resp.model_version, resp.uncertainty.get("iqr_mean", 0.0), resp.latency_ms, resp.ood_score or 0.0),
        }
