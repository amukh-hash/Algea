from __future__ import annotations


def health_payload(
    endpoints: dict[str, bool], latency_p95_ms: dict[str, float], model_status: dict[str, dict[str, str]] | None = None
) -> dict:
    model_status = model_status or {}
    healthy = all(endpoints.values()) and all(v < 200 for v in latency_p95_ms.values())
    ready_models = True
    for v in model_status.values():
        status = v.get("status", "")
        if status.startswith("error") and "optional_missing" not in status:
            ready_models = False
    ready = healthy and ready_models
    return {
        "healthy": healthy,
        "ready": ready,
        "endpoints": endpoints,
        "latency_p95_ms": latency_p95_ms,
        "models": model_status,
    }
