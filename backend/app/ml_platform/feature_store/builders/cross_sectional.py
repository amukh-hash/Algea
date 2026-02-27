from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..labels import fwd_return
from ..schemas_cross_sectional import CROSS_SECTIONAL_FIELDS
from ..transforms import winsorize, zscore
from ..universe import canonical_universe, universe_hash


def _ret(series: list[float], i: int, lookback: int) -> float:
    j = i - lookback
    if j < 0 or series[j] == 0:
        return 0.0
    return (series[i] / series[j]) - 1.0


def _vol(series: list[float], i: int, lookback: int) -> float:
    start = max(1, i - lookback + 1)
    rs = []
    for k in range(start, i + 1):
        if series[k - 1] != 0:
            rs.append((series[k] / series[k - 1]) - 1.0)
    if not rs:
        return 0.0
    return (sum(r * r for r in rs) / len(rs)) ** 0.5


def build_cross_sectional_dataset(
    dataset_id: str,
    asof: str,
    symbols: list[str],
    close_by_symbol: dict[str, list[float]],
    high_by_symbol: dict[str, list[float]] | None = None,
    low_by_symbol: dict[str, list[float]] | None = None,
    volume_by_symbol: dict[str, list[float]] | None = None,
    sector_by_symbol: dict[str, str] | None = None,
    industry_by_symbol: dict[str, str] | None = None,
    idx: int | None = None,
    out_root: Path | None = None,
) -> dict:
    high_by_symbol = high_by_symbol or {}
    low_by_symbol = low_by_symbol or {}
    volume_by_symbol = volume_by_symbol or {}
    sector_by_symbol = sector_by_symbol or {}
    industry_by_symbol = industry_by_symbol or {}
    stable_symbols = canonical_universe("selector", symbols)
    if not stable_symbols:
        raise ValueError("empty universe")

    rows = []
    for s in stable_symbols:
        c = close_by_symbol[s]
        i = idx if idx is not None else len(c) - 6
        i = min(i, len(c) - 6)
        i = max(i, 20)

        h = high_by_symbol.get(s, c)
        l = low_by_symbol.get(s, c)
        v = volume_by_symbol.get(s, [1.0] * len(c))

        row = {
            "symbol": s,
            "asof": asof,
            "ret_1d": _ret(c, i, 1),
            "ret_5d": _ret(c, i, 5),
            "ret_20d": _ret(c, i, 20),
            "vol_20d": _vol(c, i, 20),
            "range_20d": (max(h[max(0, i - 19): i + 1]) - min(l[max(0, i - 19): i + 1])) / max(c[i], 1e-9),
            "dollar_vol_20d": sum(c[max(0, i - 19): i + 1][k] * v[max(0, i - 19): i + 1][k] for k in range(len(c[max(0, i - 19): i + 1]))) / 20.0,
            "spread_proxy": (h[i] - l[i]) / max(c[i], 1e-9),
            "ema_fast_slope": _ret(c, i, 3),
            "ema_slow_slope": _ret(c, i, 10),
            "sector_id": sector_by_symbol.get(s, "UNK"),
            "industry_id": industry_by_symbol.get(s, "UNK"),
            "beta_60d_to_spy": 1.0,
            "label_fwd_ret_5d": fwd_return(c, i, 5) or 0.0,
        }
        rows.append(row)

    # cross-sectional normalization for selected numeric fields
    num_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "range_20d", "dollar_vol_20d", "spread_proxy", "ema_fast_slope", "ema_slow_slope", "beta_60d_to_spy"]
    for col in num_cols:
        vals = [float(r[col]) for r in rows]
        vals = zscore(winsorize(vals))
        for r, v in zip(rows, vals):
            r[col] = float(v)

    rows = sorted(rows, key=lambda r: r["symbol"])
    matrix = [[r[f] for f in CROSS_SECTIONAL_FIELDS if f not in {"symbol", "asof"}] for r in rows]

    manifest = {
        "dataset_id": dataset_id,
        "asof": asof,
        "universe_hash": universe_hash("selector", stable_symbols),
        "feature_code_hash": hashlib.sha256("cross_sectional_v1".encode()).hexdigest(),
        "source_bars_hash": hashlib.sha256(json.dumps(close_by_symbol, sort_keys=True).encode()).hexdigest(),
        "symbols": stable_symbols,
        "feature_order": [f for f in CROSS_SECTIONAL_FIELDS if f not in {"symbol", "asof"}],
    }
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    if out_root is not None:
        ds_root = out_root / "cross_sectional" / dataset_id
        ds_root.mkdir(parents=True, exist_ok=True)
        (ds_root / "rows.json").write_text(json.dumps(rows, sort_keys=True, indent=2), encoding="utf-8")
        (ds_root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    return {"rows": rows, "matrix": matrix, "manifest": manifest}


def build_cross_sectional(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r.get("asof"), r.get("universe_id", ""), r.get("symbol")))
