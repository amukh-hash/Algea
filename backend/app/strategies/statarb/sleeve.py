from __future__ import annotations

from ...ml_platform.inference_gateway.client import InferenceGatewayClient
from ...ml_platform.models.itransformer.types import ITransformerSignalRequest
from .correlation_killswitch import correlation_break
from .portfolio_constructor import build_statarb_targets
from .risk import statarb_risk_payload


class StatArbSleeve:
    def __init__(self, client: InferenceGatewayClient, model_alias: str = "prod"):
        self.client = client
        self.model_alias = model_alias

    def generate_targets(self, asof: str, symbols: list[str], feature_matrix: list[list[float]], trace_id: str) -> dict:
        req = ITransformerSignalRequest(
            asof=asof,
            symbols=symbols,
            feature_matrix=feature_matrix,
            model_alias=self.model_alias,
            trace_id=trace_id,
        )
        resp = self.client.itransformer_signal(req, critical=True)
        if resp is None:
            return {"status": "halted", "reason": "inputs_missing", "targets": []}
        if correlation_break(resp.correlation_regime, threshold=2.0):
            return {
                "status": "halted",
                "reason": "correlation_break",
                "targets": [],
                "ml_risk": statarb_risk_payload(resp.model_version, self.model_alias, resp.uncertainty, resp.correlation_regime, resp.latency_ms),
            }
        targets = build_statarb_targets(resp.scores)
        return {
            "status": "ok",
            "targets": targets,
            "ml_risk": statarb_risk_payload(resp.model_version, self.model_alias, resp.uncertainty, resp.correlation_regime, resp.latency_ms),
        }
