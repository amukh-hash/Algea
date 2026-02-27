from __future__ import annotations

from ...ml_platform.config import MLPlatformConfig
from ...ml_platform.inference_gateway.client import InferenceGatewayClient
from ...ml_platform.models.rl_policy.types import RLPolicyRequest
from ...ml_platform.models.vol_surface.types import VolSurfaceRequest
from ...ml_platform.rl.action_projection import project_action
from .vrp_decision import choose_tenor
from .vrp_features import build_vrp_history
from .vrp_risk import build_vrp_ml_risk


class VRPSleeve:
    def __init__(self, client: InferenceGatewayClient, model_alias: str = "prod", cfg: MLPlatformConfig | None = None):
        self.client = client
        self.model_alias = model_alias
        self.cfg = cfg or MLPlatformConfig()

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

        base_size = min(0.03, max(0.01, edge[tenor]))
        rl_fields = {}
        final_size = base_size
        if self.cfg.enable_rl_overlay_vrp:
            rl_req = RLPolicyRequest(
                asof=asof,
                sleeve="vrp",
                state={"edge": edge[tenor], "uncertainty": resp.uncertainty.get(tenor, 0.0), "drift": resp.drift_score},
                proposal={"symbol": underlying_symbol, "tenor": tenor, "base_size": base_size},
                constraints={"max_multiplier": 1.0},
                model_alias=self.cfg.rl_policy_alias_vrp,
                trace_id=f"{trace_id}-rl-vrp",
            )
            try:
                rl_resp = self.client.rl_policy_act(rl_req, critical=True)
            except Exception:
                if self.cfg.rl_fail_mode == "fallback_to_1.0":
                    rl_resp = None
                else:
                    return {"status": "halted", "reason": "rl_unavailable", "targets": []}
            if rl_resp is not None:
                projected, reason = project_action(
                    {"size_multiplier": rl_resp.size_multiplier, "veto": rl_resp.veto},
                    rl_req.constraints,
                    rl_req.proposal,
                    rl_req.state,
                )
                final_size = base_size * float(projected["size_multiplier"])
                rl_fields = {
                    "rl_model_version": rl_resp.model_version,
                    "rl_model_alias": self.cfg.rl_policy_alias_vrp,
                    "raw_action": {"size_multiplier": rl_resp.size_multiplier, "veto": rl_resp.veto},
                    "projected_action": projected,
                    "projection_reason": reason,
                    "veto": bool(projected["veto"]),
                    "latency_ms": rl_resp.latency_ms,
                    "drift_score": rl_resp.drift_score,
                    "ood_score": rl_resp.ood_score,
                }
                if projected["veto"] or projected["size_multiplier"] <= 0.0:
                    return {"status": "ok", "reason": "noop_rl_veto", "targets": [], "ml_risk": build_vrp_ml_risk(resp.model_version, self.model_alias, resp.predicted_rv, edge, resp.uncertainty, resp.drift_score, resp.ood_score, resp.latency_ms, rl_fields=rl_fields)}

        targets = [{"symbol": underlying_symbol, "target_weight": round(final_size, 6), "tenor": tenor}]
        return {
            "status": "ok",
            "targets": targets,
            "ml_risk": build_vrp_ml_risk(resp.model_version, self.model_alias, resp.predicted_rv, edge, resp.uncertainty, resp.drift_score, resp.ood_score, resp.latency_ms, rl_fields=rl_fields),
        }
