from __future__ import annotations


def rank_mse_loss(pred: list[float], target: list[float]) -> float:
    if not pred:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(pred, target)) / len(pred)


def load_balance_loss(utilization: dict[int, int], n_experts: int) -> float:
    total = sum(utilization.values()) or 1
    target = 1.0 / max(n_experts, 1)
    return sum(abs((utilization.get(i, 0) / total) - target) for i in range(n_experts))
