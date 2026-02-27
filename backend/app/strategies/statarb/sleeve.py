from __future__ import annotations

from ...ml_platform.config import MLPlatformConfig
from ...ml_platform.inference_gateway.client import InferenceGatewayClient
from ...ml_platform.models.itransformer.types import ITransformerSignalRequest
from ...ml_platform.models.rl_policy.types import RLPolicyRequest
from ...ml_platform.rl.action_projection import project_action
from .correlation_killswitch import correlation_break
from .portfolio_constructor import build_statarb_targets
from .risk import statarb_risk_payload


class StatArbSleeve:
    def __init__(self, client: InferenceGatewayClient, model_alias: str = "prod", cfg: MLPlatformConfig | None = None):
        self.client = client
        self.model_alias = model_alias
        self.cfg = cfg or MLPlatformConfig()

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
                "ml_risk": statarb_risk_payload(resp.model_version, self.model_alias, resp.uncertainty, resp.correlation_regime, resp.latency_ms, rl_fields={"projection_reason": "correlation_killswitch"}),
            }
        targets = build_statarb_targets(resp.scores)
        rl_fields = {}
        if self.cfg.enable_rl_overlay_statarb and targets:
            rl_req = RLPolicyRequest(
                asof=asof,
                sleeve="statarb",
                state={"uncertainty": resp.uncertainty, "correlation_regime": resp.correlation_regime},
                proposal={"gross_scale": 1.0, "n_names": len(targets)},
                constraints={"max_multiplier": 1.0, "max_gross_scale": 1.0, "correlation_break_threshold": 2.0},
                model_alias=self.cfg.rl_policy_alias_statarb,
                trace_id=f"{trace_id}-rl-statarb",
            )
            try:
                rl_resp = self.client.rl_policy_act(rl_req, critical=True)
            except Exception:
                if self.cfg.rl_fail_mode == "fallback_to_1.0":
                    rl_resp = None
                else:
                    return {"status": "halted", "reason": "rl_unavailable", "targets": []}
            if rl_resp is not None:
                projected, reason = project_action({"size_multiplier": rl_resp.size_multiplier, "veto": rl_resp.veto}, rl_req.constraints, rl_req.proposal, rl_req.state)
                for t in targets:
                    t["target_weight"] = round(float(t["target_weight"]) * float(projected["size_multiplier"]), 6)
                rl_fields = {
                    "rl_model_version": rl_resp.model_version,
                    "rl_model_alias": self.cfg.rl_policy_alias_statarb,
                    "raw_action": {"size_multiplier": rl_resp.size_multiplier, "veto": rl_resp.veto},
                    "projected_action": projected,
                    "projection_reason": reason,
                    "veto": bool(projected["veto"]),
                    "latency_ms": rl_resp.latency_ms,
                    "drift_score": rl_resp.drift_score,
                    "ood_score": rl_resp.ood_score,
                }
                if projected["veto"]:
                    targets = []
        return {
            "status": "ok",
            "targets": targets,
            "ml_risk": statarb_risk_payload(resp.model_version, self.model_alias, resp.uncertainty, resp.correlation_regime, resp.latency_ms, rl_fields=rl_fields),
        }
