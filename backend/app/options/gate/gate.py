from backend.app.options.gate.context import OptionsContext
from backend.app.options.types import GateDecision, GateReasonCode
from backend.app.options.gate.features import compute_gate_features
from backend.app.options.strategy.vibe_check import vibe_check

class OptionsGate:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or {
            "min_bpi": 30.0,
            "min_student_p50": -0.01,
            "min_prior_trend_conf": 0.5,
            "max_prior_downside_q10": -0.10 # e.g. -10% drop
        }
        
    def evaluate(self, ctx: OptionsContext) -> GateDecision:
        # 1. Compute Features
        feats = compute_gate_features(ctx)
        
        # 2. Stage 1: Regime / Student
        if feats["bpi"] < self.thresholds["min_bpi"]:
            return GateDecision(False, GateReasonCode.REJECT_REGIME, f"BPI {feats['bpi']} < {self.thresholds['min_bpi']}")
            
        if feats["student_p50_3d"] < self.thresholds["min_student_p50"]:
             return GateDecision(False, GateReasonCode.REJECT_REGIME, f"Student P50 {feats['student_p50_3d']:.4f} too low")
        # 0. Stage 0: Equity Selection & Priors
        if not ctx.in_equity_selection:
            return GateDecision(False, GateReasonCode.REJECT_REGIME, "Not in equity selection")
            
        if not ctx.teacher_priors:
            return GateDecision(False, GateReasonCode.UNCERTAINTY_HIGH, "Missing teacher priors")
            
        priors = ctx.teacher_priors
        if priors.trend_conf_20d < self.thresholds["min_prior_trend_conf"]:
             return GateDecision(False, GateReasonCode.TREND_WEAK, f"Prior Trend Conf {priors.trend_conf_20d:.2f} < {self.thresholds['min_prior_trend_conf']}")

        # 1. Compute Features(Legacy + New)
        # Assuming Mock AI uncertainty 0.5 for now
        if ctx.iv_snapshot:
            passed, code = vibe_check(
                ai_uncertainty=0.5, # Placeholder or from ctx
                iv_snapshot=ctx.iv_snapshot,
                posture=ctx.posture,
                liquidity_ok=True # Placeholder
            )
            if not passed:
                return GateDecision(False, code, "Vibe check failed")
        
        return GateDecision(True, GateReasonCode.PASS, "All checks passed", metadata=feats)
