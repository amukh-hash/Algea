from __future__ import annotations


class ITransformerModel:
    def __init__(self, hidden_size: int = 32):
        self.hidden_size = hidden_size

    def signal(self, feature_matrix: list[list[float]]) -> tuple[list[float], float]:
        if not feature_matrix:
            return [], 0.0
        scores = []
        for row in feature_matrix:
            s = sum((i + 1) * float(v) for i, v in enumerate(row)) / max(len(row), 1)
            scores.append(-s)  # mean-reversion polarity
        corr_regime = sum(abs(v) for row in feature_matrix for v in row) / max(
            sum(len(r) for r in feature_matrix), 1
        )
        return scores, corr_regime
