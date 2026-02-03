from typing import Tuple
from backend.app.options.types import GateReasonCode
from backend.app.options.data.types import IVSnapshot

def vibe_check(
    ai_uncertainty: float,
    iv_snapshot: IVSnapshot,
    posture: str,
    liquidity_ok: bool
) -> Tuple[bool, GateReasonCode]:

    # 1. Liquidity
    if not liquidity_ok:
        return False, GateReasonCode.REJECT_LIQUIDITY

    # 2. IV Extremes
    if iv_snapshot.iv_rank is not None:
        if iv_snapshot.iv_rank < 0.05: # Too cheap to sell
            return False, GateReasonCode.REJECT_IV
        if iv_snapshot.iv_rank > 0.95: # Too expensive/panic? Maybe good for selling but risky
            if posture != "DEFENSIVE":
                return False, GateReasonCode.REJECT_IV

    # 3. Model Uncertainty vs Market
    # If model is super uncertain, don't trade
    # Threshold depends on normalization. Assuming raw 0-1 from Mock.
    if ai_uncertainty > 0.8:
        return False, GateReasonCode.REJECT_MODEL

    return True, GateReasonCode.PASS
