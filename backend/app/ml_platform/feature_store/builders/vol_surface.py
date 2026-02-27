from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..labels_vol import forward_realized_vol
from ..realized_vol import realized_vol

TENORS = [7, 14, 30, 60]


def _atm_iv(rows: list[dict], tenor: int) -> float | None:
    cands = [r for r in rows if abs(int(r.get("dte", 0)) - tenor) <= 2]
    if not cands:
        return None
    best = sorted(cands, key=lambda r: abs(float(r.get("strike", 0)) - float(r.get("spot", 0))))[0]
    return float(best.get("implied_vol", 0.0))


def _skew_25d(rows: list[dict], tenor: int) -> float:
    cands = [r for r in rows if abs(int(r.get("dte", 0)) - tenor) <= 2]
    p = [r for r in cands if r.get("option_type") == "P"]
    c = [r for r in cands if r.get("option_type") == "C"]
    if not p or not c:
        return 0.0
    p25 = sorted(p, key=lambda r: abs(abs(float(r.get("delta", -1))) - 0.25))[0]
    c25 = sorted(c, key=lambda r: abs(abs(float(r.get("delta", 1))) - 0.25))[0]
    return float(p25.get("implied_vol", 0.0)) - float(c25.get("implied_vol", 0.0))


def build_vol_surface_dataset(dataset_id: str, asof: str, underlying_symbol: str, option_rows: list[dict], underlying_close: list[float], idx: int, out_root: Path | None = None) -> dict:
    features: dict[int, dict] = {}
    labels: dict[int, float] = {}
    for t in TENORS:
        atm = _atm_iv(option_rows, t)
        if atm is None:
            atm = 0.0
            missing = 1
        else:
            missing = 0
        features[t] = {
            "iv_atm": atm,
            "iv_skew_25d": _skew_25d(option_rows, t),
            "rv_hist_5": realized_vol(underlying_close[: idx + 1], 5),
            "rv_hist_20": realized_vol(underlying_close[: idx + 1], 20),
            "missing_flag": missing,
        }
        labels[t] = forward_realized_vol(underlying_close, idx, t) or 0.0

    manifest = {
        "dataset_id": dataset_id,
        "asof": asof,
        "underlying_symbol": underlying_symbol,
        "tenors": TENORS,
        "source_hash": hashlib.sha256(json.dumps(option_rows, sort_keys=True).encode()).hexdigest(),
        "feature_hash": hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest(),
    }
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    payload = {"features": features, "labels": labels, "manifest": manifest}
    if out_root is not None:
        root = out_root / "vol_surface" / dataset_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "dataset.json").write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload
