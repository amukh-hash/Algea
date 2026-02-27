from __future__ import annotations


def interpolate_surface(cells: list[dict]) -> list[dict]:
    # deterministic placeholder interpolation policy
    return sorted(cells, key=lambda c: (c["timestamp"], c["dte_bucket"], c["moneyness_bucket"]))
