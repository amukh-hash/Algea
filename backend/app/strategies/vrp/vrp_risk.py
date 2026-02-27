from __future__ import annotations


def build_vrp_ml_risk(model_version: str, model_alias: str, predicted_rv: dict, edge_by_tenor: dict[int, float], uncertainty: dict[int, float], drift: float, ood: float, latency_ms: float) -> dict:
    return {
        "model_name": "vol_surface",
        "model_version": model_version,
        "model_alias": model_alias,
        "predicted_rv": predicted_rv,
        "edge_by_tenor": edge_by_tenor,
        "uncertainty": uncertainty,
        "drift_score": drift,
        "ood_score": ood,
        "latency_ms_p95": latency_ms,
        "router_entropy_mean": 0.0,
        "expert_utilization": {},
        "load_balance_score": 0.0,
        "fallback_used": False,
    }
