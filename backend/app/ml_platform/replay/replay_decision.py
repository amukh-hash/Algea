from __future__ import annotations

import json
from pathlib import Path

from ..inference_gateway.client import InferenceGatewayClient
from ..models.rl_policy.types import RLPolicyRequest


def replay_rl_policy_decision(trace_path: Path, client: InferenceGatewayClient, pinned_alias: str = "prod") -> dict:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    req_payload = payload.get("request", payload.get("request_payload", {}))
    if not req_payload:
        raise ValueError("trace missing request payload for RL replay")
    req_payload["model_alias"] = pinned_alias
    req = RLPolicyRequest(**req_payload)
    out = client.rl_policy_act(req, critical=True)
    if out is None:
        raise RuntimeError("replay failed: no response")
    return out.model_dump()
