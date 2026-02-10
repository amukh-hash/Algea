"""
Promotion gate — compares candidate vs baseline model metrics.

Ported from deprecated/backend_app_snapshot/eval/promotion_gate.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


class PromotionGate:
    """
    Gate-check before promoting a candidate model to production.

    Compares candidate metrics against a baseline and rejects if any
    threshold is violated.

    Parameters
    ----------
    accuracy_tol : acceptable accuracy drop relative to baseline
    coverage_min / coverage_max : 90% coverage bounds
    width_tol_pct : max fractional increase in interval width
    """

    def __init__(
        self,
        accuracy_tol: float = 0.02,
        coverage_min: float = 0.80,
        coverage_max: float = 0.98,
        width_tol_pct: float = 0.10,
    ) -> None:
        self.accuracy_tol = accuracy_tol
        self.coverage_min = coverage_min
        self.coverage_max = coverage_max
        self.width_tol_pct = width_tol_pct

    def check(
        self,
        candidate: Dict[str, float],
        baseline: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """
        Returns ``(passed, reasons)`` — ``reasons`` lists failing criteria.
        """
        reasons: List[str] = []

        gates = self._build_gates(candidate, baseline)
        for failed, msg in gates:
            if failed:
                reasons.append(msg)

        return len(reasons) == 0, reasons

    def _build_gates(
        self,
        candidate: Dict[str, float],
        baseline: Dict[str, float],
    ) -> List[Tuple[bool, str]]:
        """Return a list of ``(is_failing, message)`` tuples for each gate."""
        gates: List[Tuple[bool, str]] = []

        # 1. Accuracy regression
        if "accuracy" in baseline and "accuracy" in candidate:
            diff = candidate["accuracy"] - baseline["accuracy"]
            gates.append((
                diff < -self.accuracy_tol,
                f"Accuracy dropped by {abs(diff):.4f} (tol={self.accuracy_tol})",
            ))

        # 2. Coverage bounds
        if "coverage_90" in candidate:
            cov = candidate["coverage_90"]
            gates.append((
                cov < self.coverage_min or cov > self.coverage_max,
                f"Coverage {cov:.4f} out of bounds [{self.coverage_min}, {self.coverage_max}]",
            ))

        # 3. Width inflation
        if "width_90" in baseline and "width_90" in candidate:
            base_w = baseline["width_90"]
            cand_w = candidate["width_90"]
            pct_change = (cand_w - base_w) / (base_w + 1e-9)
            gates.append((
                pct_change > self.width_tol_pct,
                f"Interval width increased by {pct_change:.2%}",
            ))

        return gates
