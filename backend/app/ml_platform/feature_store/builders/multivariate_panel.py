from __future__ import annotations

import json
from pathlib import Path

from ..panel_features import build_panel_feature_row
from ..panel_labels import panel_label_fwd_ret
from ..panel_manifest import build_panel_manifest
from ..universe_panel import build_panel_universe, panel_universe_hash


FEATURE_ORDER = ["ret_1", "ret_5", "ret_20", "rv_20"]


def build_multivariate_panel_dataset(
    dataset_id: str,
    asof: str,
    symbols: list[str],
    close_by_symbol: dict[str, list[float]],
    idx: int,
    out_root: Path | None = None,
) -> dict:
    universe = build_panel_universe(symbols)
    rows = []
    matrix = []
    labels = []
    for asset_idx, symbol in enumerate(universe):
        close = close_by_symbol[symbol]
        feat = build_panel_feature_row(close, idx)
        label = panel_label_fwd_ret(close, idx, horizon=5)
        rows.append({"asset_index": asset_idx, "symbol": symbol, **feat, "label_fwd_ret_5": label or 0.0})
        matrix.append([feat[f] for f in FEATURE_ORDER])
        labels.append(label or 0.0)

    manifest = build_panel_manifest(
        {
            "dataset_id": dataset_id,
            "asof": asof,
            "universe_hash": panel_universe_hash(universe),
            "symbols": universe,
            "feature_order": FEATURE_ORDER,
        }
    )

    payload = {"rows": rows, "matrix": matrix, "labels": labels, "manifest": manifest}
    if out_root:
        root = out_root / "multivariate_panel" / dataset_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "dataset.json").write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload


def build_multivariate_panel(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["timestamp"], r["asset_index"]))
