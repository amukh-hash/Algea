from __future__ import annotations


def sleeve_risk_payload(model_version: str, uncertainty: float, latency_ms: float, ood_score: float) -> dict:
    return {
        "model_version": model_version,
        "uncertainty": uncertainty,
        "latency_ms": latency_ms,
        "ood_score": ood_score,
        "drift_score": ood_score,
        "fallback_used": False,
    }
