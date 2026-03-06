from __future__ import annotations

import math
from typing import Any

from .beta_map import get_spy_beta


def allocate_sleeve_gross(
    sleeve_metrics: dict[str, dict[str, float]],
    total_gross_cap: float = 1.0,
    sleeve_min: float = 0.0,
    sleeve_max: float = 0.7,
    max_turnover: float = 0.25,
    prev_allocations: dict[str, float] | None = None,
) -> dict[str, float]:
    prev_allocations = prev_allocations or {}
    raw = {}
    for sleeve, m in sorted(sleeve_metrics.items()):
        score = float(m.get("expected_return_proxy", 0.0)) - float(m.get("uncertainty", 0.0)) - float(m.get("drift", 0.0)) - float(m.get("drawdown", 0.0))
        raw[sleeve] = max(0.0, score)
    s = sum(raw.values()) or 1.0
    alloc = {k: min(sleeve_max, max(sleeve_min, total_gross_cap * v / s)) for k, v in raw.items()}
    for k, v in list(alloc.items()):
        prev = prev_allocations.get(k, v)
        delta = v - prev
        if abs(delta) > max_turnover:
            alloc[k] = prev + (max_turnover if delta > 0 else -max_turnover)
    return alloc


def enforce_portfolio_beta_limits(
    intents: list[Any],
    prices: dict[str, float],
    nav: float,
    *,
    max_beta: float = 1.2,
) -> list[Any]:
    """Enforce max portfolio-level SPY beta exposure.

    Computes SPY-equivalent notional for each intent using::

        spy_delta = qty × multiplier × price × beta × direction

    If the aggregate net SPY delta exceeds ``max_beta × nav``, scales
    down **only** the intents whose direction matches the breach side.
    Applies ``math.floor()`` truncation to prevent fractional quantities.

    Parameters
    ----------
    intents
        List of intent-like objects with ``symbol``, ``qty``, ``multiplier``,
        ``direction`` (+1 / -1), and ``asset_class`` attributes.
    prices
        Current prices keyed by symbol.
    nav
        Total portfolio net asset value.
    max_beta
        Maximum allowed net SPY beta exposure as a fraction of NAV.
        Default 1.2 (i.e. 120% net long SPY equivalent).
    """
    if nav <= 0 or not intents:
        return intents

    max_exposure = max_beta * nav

    # ── Compute per-intent SPY-equivalent notional ────────────────────
    exposures: list[float] = []
    for intent in intents:
        sym = getattr(intent, "symbol", "")
        qty = getattr(intent, "qty", 0)
        mult = getattr(intent, "multiplier", 1.0)
        direction = getattr(intent, "direction", 1)
        price = prices.get(sym, 0.0)
        beta = get_spy_beta(sym)

        # Use abs(qty) to prevent double-negative inversion when intents
        # carry both signed qty AND signed direction.  Directionality
        # must derive solely from the `direction` attribute.
        spy_notional = abs(qty) * mult * price * beta * direction
        exposures.append(spy_notional)

    net_spy_delta = sum(exposures)

    # ── Check if breach ──────────────────────────────────────────────
    if abs(net_spy_delta) <= max_exposure:
        return intents  # Within limits, no scaling needed

    # Determine breach direction: positive = too long, negative = too short
    breach_direction = 1 if net_spy_delta > 0 else -1

    # Sum only the exposures contributing to the breach
    contributing_total = sum(
        e for e in exposures
        if (e > 0 and breach_direction > 0) or (e < 0 and breach_direction < 0)
    )

    if abs(contributing_total) < 1e-10:
        return intents  # No contributing exposure to scale

    # How much exposure must we remove?
    excess = abs(net_spy_delta) - max_exposure
    scale_factor = 1.0 - (excess / abs(contributing_total))
    scale_factor = max(0.0, scale_factor)

    # ── Apply directional scaling with integer truncation ────────────
    for i, intent in enumerate(intents):
        if (exposures[i] > 0 and breach_direction > 0) or \
           (exposures[i] < 0 and breach_direction < 0):
            # This intent contributes to the breach — scale it down
            current_qty = getattr(intent, "qty", 0)
            new_qty = math.floor(abs(current_qty) * scale_factor)
            intent.qty = new_qty * (1 if current_qty >= 0 else -1)

    # NOTE: Do NOT filter out qty=0 intents here. qty=0 represents a
    # valid FLATTEN command that the execution engine must receive.
    # The IB Error 104 guard (qty != 0) must be applied downstream,
    # after computing differential orders (target_qty - current_qty).
    return intents

