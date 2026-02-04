from typing import Dict, Any
import numpy as np
from backend.app.options.gate.context import OptionsContext

def compute_gate_features(ctx: OptionsContext) -> Dict[str, float]:
    """
    Extract flat feature vector for gate logic/tuning.
    """
    feats = {}
    
    # Student Signal
    # Assuming quantiles 3D horizon
    q_3d = ctx.student_signal.quantiles.get("3D", {})
    feats["student_p50_3d"] = q_3d.get("0.50", 0.0)
    feats["student_p05_3d"] = q_3d.get("0.05", 0.0)
    
    # Breadth
    feats["ad_line"] = ctx.breadth.get("ad_line", 0.0)
    feats["bpi"] = ctx.breadth.get("bpi", 50.0)
    
    # IV
    if ctx.iv_snapshot:
        feats["iv_rank"] = ctx.iv_snapshot.iv_rank if ctx.iv_snapshot.iv_rank is not None else 0.5
        feats["atm_iv"] = ctx.iv_snapshot.atm_iv
        
    return feats
