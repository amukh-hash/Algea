#!/usr/bin/env python
"""CLI: Run CO→OC validation pipeline.

Usage
-----
  python -m backend.scripts.run_cooc_validation --config path/to/config.yaml

Options
-------
  --config       Path to YAML config (default: COOC sleeve default config)
  --tiers        Realism tiers to evaluate (0, 1, 2 — default: all)
  --seeds        Override training seeds (comma-separated)
  --report-only  Skip training, load existing artifacts, and report
  --oracle       Run oracle sanity check
  --output-dir   Directory for reports (default: ./validation_output)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CO→OC Cross-Sectional Validation")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--tiers", type=str, default="0,1,2", help="Realism tiers (comma-separated)")
    p.add_argument("--seeds", type=str, default=None, help="Training seeds (comma-separated)")
    p.add_argument("--report-only", action="store_true", help="Report mode only")
    p.add_argument("--oracle", action="store_true", help="Oracle sanity check")
    p.add_argument("--output-dir", type=str, default="./validation_output", help="Output dir")
    return p.parse_args(argv)


def _load_config(config_path: Optional[str]):
    """Load COOCReversalConfig from YAML or defaults."""
    from sleeves.cooc_reversal_futures.config import COOCReversalConfig, load_config_from_yaml

    if config_path is not None:
        logger.info("Loading config from %s", config_path)
        return load_config_from_yaml(config_path)
    return COOCReversalConfig()


def _oracle_sanity_check(output_dir: Path) -> dict:
    """Run oracle and anti-oracle to verify polarity."""
    from backend.app.portfolio.alpha_conventions import label_y
    from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

    # Build minimal panel
    rng = np.random.RandomState(42)
    n_days, n_inst = 100, 6
    base = pd.Timestamp("2024-01-02")
    rows = []
    for d in range(n_days):
        td = (base + pd.offsets.BDay(d)).date()
        r_co = rng.randn(n_inst) * 0.01
        r_oc = -0.7 * r_co + rng.randn(n_inst) * 0.003
        for i in range(n_inst):
            rows.append({
                "trading_day": td, "root": f"R{i}", "instrument": f"R{i}",
                "r_co": r_co[i], "r_oc": r_oc[i],
            })
    panel = pd.DataFrame(rows)
    panel["y"] = label_y(panel)

    zero_cost = {"cost_per_contract": 0.0, "slippage_bps_open": 0.0, "slippage_bps_close": 0.0}

    oracle_report = evaluate_trade_proxy(panel, panel["y"].values, config=zero_cost)
    anti_report = evaluate_trade_proxy(panel, -panel["y"].values, config=zero_cost)

    result = {
        "oracle_sharpe": float(oracle_report.sharpe_model),
        "anti_oracle_sharpe": float(anti_report.sharpe_model),
        "oracle_positive": oracle_report.sharpe_model > 0,
        "anti_oracle_negative": anti_report.sharpe_model < 0,
        "polarity_correct": oracle_report.sharpe_model > 0 and anti_report.sharpe_model < 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oracle_sanity.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)

    # Oracle sanity check
    if args.oracle:
        logger.info("Running oracle sanity check...")
        result = _oracle_sanity_check(output_dir)
        status = "PASS" if result["polarity_correct"] else "FAIL"
        logger.info(
            "Oracle sanity: %s (oracle=%.3f, anti=%.3f)",
            status, result["oracle_sharpe"], result["anti_oracle_sharpe"],
        )
        if not result["polarity_correct"]:
            logger.error("POLARITY BUG: oracle should be positive, anti-oracle negative")
            return 1

    # Parse tiers
    tiers = [int(t.strip()) for t in args.tiers.split(",")]
    logger.info("Tiers: %s", tiers)

    # Parse seeds
    if args.seeds is not None:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    else:
        seeds = config.training.seeds
    logger.info("Seeds: %s", seeds)

    if args.report_only:
        logger.info("Report-only mode — skipping training")
        # In report-only mode, we'd load existing artifacts
        # For now, just verify polarity and exit
        return 0

    logger.info("Full validation pipeline not yet automated — use run_validation() API")
    logger.info("Config: universe=%s, model=%s", list(config.universe), config.model.estimator_type)
    logger.info("Output dir: %s", output_dir)

    # Save config snapshot
    (output_dir / "config_snapshot.json").write_text(
        json.dumps({
            "universe": list(config.universe),
            "model_type": config.model.estimator_type,
            "seeds": list(seeds),
            "tiers": tiers,
        }, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
