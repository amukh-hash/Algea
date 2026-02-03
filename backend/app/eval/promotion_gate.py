from typing import Dict, List, Tuple

class PromotionGate:
    def __init__(self, accuracy_tol: float = 0.02, coverage_min: float = 0.80, coverage_max: float = 0.98, width_tol_pct: float = 0.10):
        self.accuracy_tol = accuracy_tol
        self.coverage_min = coverage_min
        self.coverage_max = coverage_max
        self.width_tol_pct = width_tol_pct

    def check(self, candidate: Dict[str, float], baseline: Dict[str, float]) -> Tuple[bool, List[str]]:
        reasons = []
        passed = True

        # 1. Accuracy
        # If baseline has accuracy, compare.
        if "accuracy" in baseline and "accuracy" in candidate:
            diff = candidate["accuracy"] - baseline["accuracy"]
            if diff < -self.accuracy_tol:
                passed = False
                reasons.append(f"Accuracy dropped by {abs(diff):.4f} (Tol: {self.accuracy_tol})")

        # 2. Coverage (Calibration)
        # Check absolute bounds on candidate
        if "coverage_90" in candidate:
            cov = candidate["coverage_90"]
            if cov < self.coverage_min or cov > self.coverage_max:
                passed = False
                reasons.append(f"Coverage {cov:.4f} out of bounds [{self.coverage_min}, {self.coverage_max}]")

        # 3. Width (Uncertainty)
        if "width_90" in baseline and "width_90" in candidate:
            base_width = baseline["width_90"]
            cand_width = candidate["width_90"]
            pct_change = (cand_width - base_width) / (base_width + 1e-9)
            if pct_change > self.width_tol_pct:
                passed = False
                reasons.append(f"Interval width increased by {pct_change:.2%}")

        return passed, reasons
