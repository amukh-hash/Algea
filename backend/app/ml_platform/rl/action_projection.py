from __future__ import annotations

import math


def project_action(action: dict, constraints: dict, proposal: dict, sleeve_state: dict) -> tuple[dict, str]:
    raw_multiplier = float(action.get("size_multiplier", 0.0))
    veto = bool(action.get("veto", False))
    if math.isnan(raw_multiplier) or math.isinf(raw_multiplier):
        return {"size_multiplier": 0.0, "veto": True}, "nan_or_inf"
    if veto:
        return {"size_multiplier": 0.0, "veto": True}, "veto"
    m = max(0.0, min(1.0, raw_multiplier))
    cap = float(constraints.get("max_multiplier", 1.0))
    if proposal.get("gross_scale") is not None:
        cap = min(cap, float(constraints.get("max_gross_scale", cap)))
    m = min(m, cap)
    if float(sleeve_state.get("correlation_regime", 0.0)) >= float(constraints.get("correlation_break_threshold", 9e9)):
        return {"size_multiplier": 0.0, "veto": True}, "correlation_killswitch"
    return {"size_multiplier": m, "veto": False}, "clamped"
