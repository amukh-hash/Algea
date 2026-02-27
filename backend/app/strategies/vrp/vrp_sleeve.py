from __future__ import annotations

from ...ml_platform.inference_gateway.client import InferenceGatewayClient
from ...ml_platform.models.vol_surface.types import VolSurfaceRequest
from .vrp_decision import choose_tenor
from .vrp_features import build_vrp_history
from .vrp_risk import build_vrp_ml_risk


class VRPSleeve:
    def __init__(self, client: InferenceGatewayClient, model_alias: str = "prod"):
        self.client = client
        self.model_alias = model_alias

    def generate_targets(self, asof: str, underlying_symbol: str, iv_atm_by_tenor: dict[int, float], features_by_tenor: dict[int, dict], trace_id: str, edge_threshold: float = 0.01, uncertainty_threshold: float = 0.2, drift_threshold: float = 3.0) -> dict:
        history = build_vrp_history(features_by_tenor)
        req = VolSurfaceRequest(
            asof=asof,
            underlying_symbol=underlying_symbol,
            tenors=sorted(iv_atm_by_tenor.keys()),
            history=history,
            model_alias=self.model_alias,
            trace_id=trace_id,
        )
        resp = self.client.vol_surface_forecast(req, critical=True)
        if resp is None:
            return {"status": "halted", "reason": "inputs_missing", "targets": []}
        if resp.drift_score > drift_threshold:
            return {"status": "halted", "reason": "drift", "targets": [], "ml_risk": build_vrp_ml_risk(resp.model_version, self.model_alias, resp.predicted_rv, {}, resp.uncertainty, resp.drift_score, resp.ood_score, resp.latency_ms)}

        edge = {int(t): float(iv_atm_by_tenor[int(t)] - float(resp.predicted_rv[int(t)]["0.50"])) for t in iv_atm_by_tenor.keys()}
        tenor = choose_tenor(edge, resp.uncertainty, edge_threshold, uncertainty_threshold)
        if tenor is None:
            return {"status": "halted", "reason": "edge_or_uncertainty", "targets": [], "ml_risk": build_vrp_ml_risk(resp.model_version, self.model_alias, resp.predicted_rv, edge, resp.uncertainty, resp.drift_score, resp.ood_score, resp.latency_ms)}

        size = min(0.03, max(0.01, edge[tenor]))
        targets = [{"symbol": underlying_symbol, "target_weight": round(size, 6), "tenor": tenor}]
        return {
            "status": "ok",
            "targets": targets,
            "ml_risk": build_vrp_ml_risk(resp.model_version, self.model_alias, resp.predicted_rv, edge, resp.uncertainty, resp.drift_score, resp.ood_score, resp.latency_ms),
        }
