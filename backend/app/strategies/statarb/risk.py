from __future__ import annotations


def statarb_risk_payload(model_version: str, model_alias: str, uncertainty: float, correlation_regime: float, latency_ms: float, rl_fields: dict | None = None) -> dict:
    return {
        "model_name": "itransformer",
        "model_version": model_version,
        "model_alias": model_alias,
        "uncertainty": uncertainty,
        "latency_ms_p95": latency_ms,
        "corr_break_score": correlation_regime,
        "beta_residual": 0.0,
        "pair_concentration": 0.0,
        "drift_score": 0.0,
        "ood_score": 0.0,
        "fallback_used": False,
        "rl_policy": rl_fields or {},
    }
