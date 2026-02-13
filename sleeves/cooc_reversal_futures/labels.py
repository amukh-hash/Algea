from __future__ import annotations

import math


def r_co(p_open_t: float, p_close_t_minus_1: float) -> float:
    return math.log(p_open_t / p_close_t_minus_1)


def r_oc(p_close_t: float, p_open_t: float) -> float:
    return math.log(p_close_t / p_open_t)


def build_labels(p_close_t: float, p_open_t: float, cost_ret: float) -> dict[str, float | int]:
    raw = r_oc(p_close_t, p_open_t)
    net = raw - cost_ret
    return {"y_raw": raw, "y_net": net, "meta_label": int(net > 0)}
