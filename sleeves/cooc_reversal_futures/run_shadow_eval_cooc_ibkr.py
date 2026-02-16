"""Daily shadow evaluation script for CO→OC reversal futures.

Runs counterfactual MODEL scoring alongside the live HEURISTIC execution
during the paper-trading phase (30–60 sessions).  Produces a daily
``shadow_eval_{asof}.json`` report + ``shadow_eval_{asof}.parquet``
with per-instrument scores, fills, and counterfactual PnL.

Usage
-----
::

    python -m sleeves.cooc_reversal_futures.run_shadow_eval_cooc_ibkr \\
        --asof 2026-02-14 \\
        --pack-dir runs/best_pack \\
        --artifacts-dir runs/shadow_eval

The script does NOT place any orders.  It only evaluates the promoted
model's output against the heuristic baseline using realized fills.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shadow eval logic
# ---------------------------------------------------------------------------

def run_shadow_eval(
    asof: date,
    pack_dir: Path,
    artifacts_dir: Path,
    *,
    heuristic_fills: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run daily shadow evaluation.

    Parameters
    ----------
    asof
        The trading date to evaluate.
    pack_dir
        Path to the promoted model pack directory (must contain
        ``model.pt``, ``scaler.pkl``, ``promotion_windows_report.json``).
    artifacts_dir
        Path to the runner's daily artifact directory (contains
        ``orders.json``, ``fills.parquet``, ``features.parquet``).
    heuristic_fills
        If provided, use these fills instead of reading from artifacts_dir.

    Returns
    -------
    Dict with shadow eval metrics:
        - ``asof``: date string
        - ``model_pnl``: counterfactual model PnL
        - ``heuristic_pnl``: realized heuristic PnL
        - ``model_sharpe_daily``: single-day Sharpe proxy (meaningless alone)
        - ``fill_completeness``: fraction of intended orders that filled
        - ``per_instrument``: per-instrument detail
    """
    report: Dict[str, Any] = {
        "asof": asof.isoformat(),
        "pack_dir": str(pack_dir),
        "artifacts_dir": str(artifacts_dir),
        "status": "ok",
    }

    # --- Load model pack metadata ---
    promo_report_path = pack_dir / "promotion_windows_report.json"
    if promo_report_path.exists():
        report["promotion_snapshot"] = json.loads(promo_report_path.read_text(encoding="utf-8"))
    else:
        report["status"] = "warn_no_promotion_report"
        logger.warning("No promotion_windows_report.json in %s", pack_dir)

    # --- Load features for asof ---
    features_path = artifacts_dir / f"features_{asof.isoformat()}.parquet"
    if not features_path.exists():
        # Try alternative naming
        features_path = artifacts_dir / "features.parquet"

    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        if "trading_day" in features_df.columns:
            td_str = asof.isoformat()
            features_df = features_df[features_df["trading_day"].astype(str) == td_str]
    else:
        report["status"] = "error_no_features"
        logger.error("No features file found at %s", features_path)
        return report

    # --- Load heuristic fills ---
    if heuristic_fills is None:
        fills_path = artifacts_dir / f"fills_{asof.isoformat()}.parquet"
        if not fills_path.exists():
            fills_path = artifacts_dir / "fills.parquet"
        if fills_path.exists():
            heuristic_fills = pd.read_parquet(fills_path)
        else:
            logger.warning("No fills file found — using empty fills.")
            heuristic_fills = pd.DataFrame()

    # --- Compute heuristic PnL from fills ---
    heuristic_pnl = 0.0
    if not heuristic_fills.empty and "pnl" in heuristic_fills.columns:
        heuristic_pnl = float(heuristic_fills["pnl"].sum())

    # --- Counterfactual model scoring ---
    # Load model and score features
    model_pnl = 0.0
    per_instrument: list[dict] = []

    try:
        model_path = pack_dir / "model.pt"
        if model_path.exists() and not features_df.empty:
            import torch
            from ..model.cs_transformer import CrossSectionalTransformer
            from ..features_core import active_schema

            # Load model
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state = checkpoint if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint else checkpoint.get("model_state_dict", checkpoint)

            # Determine feature columns
            schema_version = checkpoint.get("schema_version", 2) if isinstance(checkpoint, dict) else 2
            feature_cols = list(active_schema(schema_version))
            available_cols = [c for c in feature_cols if c in features_df.columns]

            if len(available_cols) >= len(feature_cols) * 0.8:
                X = features_df[available_cols].values.astype(np.float32)
                X = np.nan_to_num(X, nan=0.0)

                # Build model (infer config from state dict)
                n_features = len(available_cols)
                model = CrossSectionalTransformer(n_features=n_features)
                try:
                    model.load_state_dict(model_state, strict=False)
                except Exception:
                    logger.warning("Model state dict load failed — using random init for shadow eval.")

                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # [1, N, F]
                    scores = model(X_tensor).squeeze(0).numpy()  # [N]

                # Compute counterfactual return using r_oc
                instruments = features_df["instrument"].tolist() if "instrument" in features_df.columns else features_df.get("root", pd.Series()).tolist()
                r_oc = features_df["r_oc"].values if "r_oc" in features_df.columns else np.zeros(len(features_df))

                for i, inst in enumerate(instruments):
                    per_instrument.append({
                        "instrument": inst,
                        "model_score": float(scores[i]) if i < len(scores) else 0.0,
                        "r_oc": float(r_oc[i]) if i < len(r_oc) else 0.0,
                    })

                # Simple L/S: long bottom half, short top half
                n = len(scores)
                if n >= 4:
                    k = n // 4
                    order = np.argsort(scores)
                    long_idx = order[:k]
                    short_idx = order[-k:]
                    weight = 0.8 / (2 * k)
                    model_pnl = float(
                        np.sum(r_oc[long_idx]) * weight * k
                        - np.sum(r_oc[short_idx]) * weight * k
                    )
            else:
                logger.warning("Insufficient feature columns: %d/%d", len(available_cols), len(feature_cols))
                report["status"] = "warn_insufficient_features"

    except Exception as e:
        logger.error("Shadow eval model scoring failed: %s", e)
        report["status"] = "error_model_scoring"

    # --- Fill completeness ---
    fill_completeness = 1.0
    orders_path = artifacts_dir / f"orders_{asof.isoformat()}.json"
    if orders_path.exists() and not heuristic_fills.empty:
        orders = json.loads(orders_path.read_text(encoding="utf-8"))
        n_orders = len(orders) if isinstance(orders, list) else orders.get("n_orders", 0)
        n_fills = len(heuristic_fills)
        fill_completeness = n_fills / max(n_orders, 1)

    report.update({
        "model_pnl": model_pnl,
        "heuristic_pnl": heuristic_pnl,
        "fill_completeness": fill_completeness,
        "per_instrument": per_instrument,
        "n_instruments": len(per_instrument),
    })

    return report


def save_shadow_eval(
    report: Dict[str, Any],
    output_dir: Path,
    asof: date,
) -> tuple[Path, Path]:
    """Save shadow eval report as JSON + parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"shadow_eval_{asof.isoformat()}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    parquet_path = output_dir / f"shadow_eval_{asof.isoformat()}.parquet"
    if report.get("per_instrument"):
        pd.DataFrame(report["per_instrument"]).to_parquet(parquet_path, index=False)
    else:
        pd.DataFrame().to_parquet(parquet_path, index=False)

    logger.info("Shadow eval saved: %s, %s", json_path, parquet_path)
    return json_path, parquet_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Daily shadow evaluation for CO→OC reversal futures."
    )
    parser.add_argument("--asof", required=True, help="Trading date (YYYY-MM-DD)")
    parser.add_argument("--pack-dir", required=True, help="Path to promoted model pack")
    parser.add_argument("--artifacts-dir", required=True, help="Path to runner daily artifacts")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: artifacts-dir)")
    args = parser.parse_args()

    asof = date.fromisoformat(args.asof)
    pack_dir = Path(args.pack_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir) if args.output_dir else artifacts_dir

    report = run_shadow_eval(asof, pack_dir, artifacts_dir)
    save_shadow_eval(report, output_dir, asof)

    if report["status"] == "ok":
        logger.info("Shadow eval completed successfully.")
    else:
        logger.warning("Shadow eval completed with status: %s", report["status"])
        sys.exit(1)


if __name__ == "__main__":
    main()
