from __future__ import annotations


def build_panel_feature_row(close: list[float], idx: int) -> dict[str, float]:
    def _ret(lb: int) -> float:
        j = idx - lb
        if j < 0 or close[j] == 0:
            return 0.0
        return (close[idx] / close[j]) - 1.0

    def _rv(lb: int) -> float:
        rs = []
        for i in range(max(1, idx - lb + 1), idx + 1):
            if close[i - 1] != 0:
                rs.append((close[i] / close[i - 1]) - 1.0)
        if not rs:
            return 0.0
        return (sum(r * r for r in rs) / len(rs)) ** 0.5

    return {
        "ret_1": _ret(1),
        "ret_5": _ret(5),
        "ret_20": _ret(20),
        "rv_20": _rv(20),
    }
