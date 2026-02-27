from __future__ import annotations

from ...ml_platform.inference_gateway.client import InferenceGatewayClient
from ...ml_platform.models.selector_smoe.types import SMoERankRequest
from .portfolio_constructor import SelectorPortfolioConstructor


class SelectorSleeve:
    def __init__(self, client: InferenceGatewayClient, model_alias: str = "prod"):
        self.client = client
        self.model_alias = model_alias
        self.constructor = SelectorPortfolioConstructor()

    def generate_targets(self, asof: str, symbols: list[str], feature_matrix: list[list[float]], trace_id: str, market_context: dict[str, float]) -> dict:
        req = SMoERankRequest(
            asof=asof,
            symbols=symbols,
            feature_matrix=feature_matrix,
            market_context=market_context,
            model_alias=self.model_alias,
            trace_id=trace_id,
        )
        resp = self.client.smoe_rank(req, critical=True)
        if resp is None:
            return {"status": "halted", "reason": "inference_unavailable", "targets": []}
        if resp.latency_ms > 500:
            return {"status": "halted", "reason": "latency", "targets": []}
        targets = self.constructor.construct(resp.scores)
        return {
            "status": "ok",
            "targets": targets,
            "ml_risk": {
                "model_name": "selector_smoe",
                "model_version": resp.model_version,
                "model_alias": self.model_alias,
                "latency_ms_p95": resp.latency_ms,
                "router_entropy_mean": resp.router_entropy_mean,
                "expert_utilization": resp.expert_utilization,
                "load_balance_score": resp.load_balance_score,
                "ood_score": 0.0,
                "drift_score": 0.0,
                "fallback_used": False,
            },
        }
