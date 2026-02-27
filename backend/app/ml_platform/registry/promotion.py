from __future__ import annotations

from dataclasses import dataclass

from .store import ModelRegistryStore


@dataclass(frozen=True)
class PromotionGate:
    min_sharpe: float = 1.0
    max_drawdown: float = 0.2
    min_calibration: float = 0.5
    max_pinball_loss: float = 0.2
    min_rank_ic: float = 0.5
    min_top_bottom_spread: float = 0.0
    max_expert_collapse: float = 0.8
    min_router_entropy: float = 0.01
    min_context_sensitivity_score: float = 0.01
    max_vol_pinball_loss: float = 0.3
    min_edge_hit_rate: float = 0.5
    min_itr_rank_ic: float = 0.5
    min_pair_stability: float = 0.5
    max_rl_constraint_violation_rate: float = 0.0
    min_rl_seed_stability_score: float = 0.2
    max_rl_drawdown: float = 0.2


def _base_gate(metrics: dict, gate: PromotionGate) -> bool:
    if metrics.get("sharpe", gate.min_sharpe) < gate.min_sharpe:
        return False
    if metrics.get("max_drawdown", 0.0) > gate.max_drawdown:
        return False
    if metrics.get("calibration_score", 0.0) < gate.min_calibration:
        return False
    return True


def _chronos2_gate(metrics: dict, gate: PromotionGate) -> bool:
    if "pinball_loss" in metrics and float(metrics.get("pinball_loss", 9e9)) > gate.max_pinball_loss:
        return False
    return True


def _selector_smoe_gate(metrics: dict, gate: PromotionGate) -> bool:
    if float(metrics.get("rank_ic", -9e9)) < gate.min_rank_ic:
        return False
    if float(metrics.get("top_bottom_spread", -9e9)) < gate.min_top_bottom_spread:
        return False
    collapse = float(metrics.get("expert_collapse_score", metrics.get("load_balance_score", 9e9)))
    if collapse > gate.max_expert_collapse:
        return False
    if float(metrics.get("router_entropy_mean", 0.0)) < gate.min_router_entropy:
        return False
    if float(metrics.get("context_sensitivity_score", 0.0)) < gate.min_context_sensitivity_score:
        return False
    return True


def promote_if_eligible(
    store: ModelRegistryStore,
    model_name: str,
    version: str,
    metrics: dict,
    gate: PromotionGate | None = None,
    to_alias: str = "prod",
) -> bool:
    gate = gate or PromotionGate()
    if not _base_gate(metrics, gate):
        return False
    if model_name == "chronos2" and not _chronos2_gate(metrics, gate):
        return False
    if model_name == "selector_smoe" and not _selector_smoe_gate(metrics, gate):
        return False
    if model_name == "vol_surface" and not _vol_surface_gate(metrics, gate):
        return False
    if model_name == "itransformer" and not _itransformer_gate(metrics, gate):
        return False
    if model_name == "rl_policy" and not _rl_policy_gate(metrics, gate):
        return False
    store.set_alias(model_name, to_alias, version)
    return True


def _vol_surface_gate(metrics: dict, gate: PromotionGate) -> bool:
    if float(metrics.get("pinball_loss", 9e9)) > gate.max_vol_pinball_loss:
        return False
    if float(metrics.get("edge_hit_rate", -9e9)) < gate.min_edge_hit_rate:
        return False
    if float(metrics.get("calibration_score", 0.0)) < gate.min_calibration:
        return False
    return True


def _itransformer_gate(metrics: dict, gate: PromotionGate) -> bool:
    if float(metrics.get("rank_ic", -9e9)) < gate.min_itr_rank_ic:
        return False
    if float(metrics.get("pair_stability", -9e9)) < gate.min_pair_stability:
        return False
    return True



def _rl_policy_gate(metrics: dict, gate: PromotionGate) -> bool:
    if float(metrics.get("constraint_violation_rate", 9e9)) > gate.max_rl_constraint_violation_rate:
        return False
    if float(metrics.get("seed_stability_score", -9e9)) < gate.min_rl_seed_stability_score:
        return False
    if float(metrics.get("max_drawdown", 9e9)) > gate.max_rl_drawdown:
        return False
    return True
