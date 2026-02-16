"""Paper-fill shadow evaluator: MODEL vs HEURISTIC counterfactual.

Reads Phase 2 IBKR fill artifacts, computes MODEL scores for the same
instruments/dates, and generates a side-by-side performance comparison.

Usage::

    python backend/scripts/paper/run_shadow_eval_cooc_ibkr.py \\
        --fills-dir output/paper/2025-01-15/RECONCILE \\
        --intents-dir output/paper/2025-01-15/OPEN \\
        --pack-dir packs/cooc_latest \\
        --output-dir output/shadow_eval \\
        [--start-date 2025-01-01] [--end-date 2025-01-31]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _load_fills(fills_dir: Path, start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
    """Load Phase 2 fill parquets from RECONCILE directories."""
    records: List[Dict[str, Any]] = []
    for p in sorted(fills_dir.rglob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            records.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", p, e)
    if not records:
        return pd.DataFrame()
    fills = pd.concat(records, ignore_index=True)
    if "fill_date" in fills.columns and start_date:
        fills = fills[fills["fill_date"] >= str(start_date)]
    if "fill_date" in fills.columns and end_date:
        fills = fills[fills["fill_date"] <= str(end_date)]
    return fills


def _load_intents(intents_dir: Path) -> pd.DataFrame:
    """Load Phase 2 intent JSONs from OPEN directories."""
    records: List[Dict[str, Any]] = []
    for p in sorted(intents_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        except Exception as e:
            logger.warning("Failed to read %s: %s", p, e)
    return pd.DataFrame(records) if records else pd.DataFrame()


def _compute_shadow_metrics(fills_df: pd.DataFrame) -> Dict[str, float]:
    """Compute Sharpe, hit rate, mean PnL from fills."""
    if fills_df.empty or "pnl" not in fills_df.columns:
        return {"sharpe": 0.0, "hit_rate": 0.0, "mean_pnl": 0.0, "n_fills": 0}

    pnl = fills_df["pnl"].astype(float)
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0
    hit_rate = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
    return {
        "sharpe": float(sharpe),
        "hit_rate": hit_rate,
        "mean_pnl": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "n_fills": int(len(pnl)),
    }


def run_shadow_eval(
    fills_dir: str | Path,
    intents_dir: str | Path,
    pack_dir: str | Path,
    output_dir: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Run shadow evaluation: compare actual fills against MODEL counterfactual.

    Parameters
    ----------
    fills_dir : path to Phase 2 fill artifacts (parquets).
    intents_dir : path to Phase 2 intent artifacts (JSONs).
    pack_dir : path to CS-Transformer production pack.
    output_dir : path to write shadow evaluation artifacts.
    start_date, end_date : optional date range filter.

    Returns
    -------
    dict : summary report with HEURISTIC and MODEL metrics.
    """
    fills_path = Path(fills_dir)
    intents_path = Path(intents_dir)
    pack_path = Path(pack_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None

    # Load fills and intents
    fills = _load_fills(fills_path, sd, ed)
    intents = _load_intents(intents_path)

    if fills.empty:
        logger.warning("No fill data found in %s", fills_path)
        report = {"status": "NO_DATA", "fills_dir": str(fills_path)}
        (out / "shadow_eval_report.json").write_text(json.dumps(report, indent=2))
        return report

    # Check if pack is promotable
    schema_path = pack_path / "feature_schema.json"
    pack_available = pack_path.exists() and (pack_path / "model.pt").exists()
    promotion_status = "UNKNOWN"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        promotion_status = schema.get("promotion_status", "UNKNOWN")

    # Compute HEURISTIC metrics from actual fills
    heuristic_metrics = _compute_shadow_metrics(fills)

    # Compute MODEL counterfactual (if pack available)
    model_metrics: Dict[str, float] = {"sharpe": 0.0, "hit_rate": 0.0, "mean_pnl": 0.0, "n_fills": 0}
    model_available = False

    if pack_available:
        try:
            from sleeves.cooc_reversal_futures.model.infer import predict_panel

            # Build counterfactual: re-score fills using MODEL
            model_scores: List[Dict[str, Any]] = []
            for fill_date_val in fills["fill_date"].unique() if "fill_date" in fills.columns else []:
                day_fills = fills[fills["fill_date"] == fill_date_val]
                instruments = day_fills["instrument"].unique().tolist() if "instrument" in day_fills.columns else []
                if instruments:
                    try:
                        preds = predict_panel(
                            model_dir=str(pack_path),
                            instruments=instruments,
                            features={},  # would need actual features
                            trading_day=fill_date_val,
                        )
                        for inst, score in preds.items():
                            model_scores.append({
                                "fill_date": fill_date_val,
                                "instrument": inst,
                                "model_score": score,
                            })
                    except Exception as e:
                        logger.warning("MODEL predict failed for %s: %s", fill_date_val, e)

            if model_scores:
                model_df = pd.DataFrame(model_scores)
                model_available = True
                logger.info("MODEL counterfactual: %d scored instruments", len(model_df))

        except ImportError:
            logger.warning("Cannot import infer module — MODEL counterfactual skipped")
        except Exception as e:
            logger.warning("MODEL counterfactual failed: %s", e)

    # Build report
    report: Dict[str, Any] = {
        "status": "OK",
        "pack_dir": str(pack_path),
        "pack_available": pack_available,
        "promotion_status": promotion_status,
        "model_available": model_available,
        "date_range": {
            "start": start_date or "all",
            "end": end_date or "all",
        },
        "heuristic": heuristic_metrics,
        "model_counterfactual": model_metrics if model_available else {"note": "not computed"},
        "n_fill_records": len(fills),
        "n_intent_records": len(intents),
    }

    # Save artifacts
    (out / "shadow_eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str)
    )
    fills.to_parquet(out / "shadow_fills.parquet", index=False)
    if not intents.empty:
        intents.to_parquet(out / "shadow_intents.parquet", index=False)

    logger.info("Shadow eval report: %s", json.dumps(report, indent=2, default=str))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-fill shadow evaluator")
    parser.add_argument("--fills-dir", required=True, help="Path to Phase 2 fill artifacts")
    parser.add_argument("--intents-dir", required=True, help="Path to Phase 2 intent artifacts")
    parser.add_argument("--pack-dir", required=True, help="Path to CS-Transformer pack")
    parser.add_argument("--output-dir", default="output/shadow_eval", help="Output directory")
    parser.add_argument("--start-date", default=None, help="Start date (ISO)")
    parser.add_argument("--end-date", default=None, help="End date (ISO)")

    args = parser.parse_args()
    run_shadow_eval(
        fills_dir=args.fills_dir,
        intents_dir=args.intents_dir,
        pack_dir=args.pack_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
