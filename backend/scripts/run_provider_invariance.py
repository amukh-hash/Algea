"""CLI: Provider Invariance Report (R1).

Usage
-----
python -m backend.scripts.run_provider_invariance \
    --bars-a data/yfinance_bars.parquet \
    --bars-b data/ibkr_bars.parquet \
    --output-dir runs/provider_invariance

Exit code 0 = consistent, 1 = flags raised.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run provider invariance check (R1 deliverable).",
    )
    parser.add_argument(
        "--bars-a", required=True,
        help="Path to provider-A bars (parquet or CSV).",
    )
    parser.add_argument(
        "--bars-b", required=True,
        help="Path to provider-B bars (parquet or CSV).",
    )
    parser.add_argument(
        "--roots", nargs="*", default=None,
        help="Roots to compare (default: intersect both files).",
    )
    parser.add_argument(
        "--sample-days", type=int, default=60,
        help="Max overlapping days to sample per root.",
    )
    parser.add_argument(
        "--correlation-threshold", type=float, default=0.90,
        help="Min baseline proxy correlation threshold.",
    )
    parser.add_argument(
        "--output-dir", default="runs/provider_invariance",
        help="Output directory for reports.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # Load bars
    # ------------------------------------------------------------------
    def _load(path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)

    df_a = _load(args.bars_a)
    df_b = _load(args.bars_b)

    # Resolve roots
    roots = args.roots
    if roots is None:
        roots_a = set(df_a["root"].unique()) if "root" in df_a.columns else set()
        roots_b = set(df_b["root"].unique()) if "root" in df_b.columns else set()
        roots = sorted(roots_a & roots_b)
        if not roots:
            logger.error("No overlapping roots found between the two files.")
            return 1
    logger.info("Comparing %d roots: %s", len(roots), roots)

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    from sleeves.cooc_reversal_futures.pipeline.session_semantics import (
        build_provider_invariance_report,
    )

    report = build_provider_invariance_report(
        df_a, df_b, roots,
        sample_days=args.sample_days,
        correlation_threshold=args.correlation_threshold,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = report.to_dict()
    (out / "provider_invariance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    if report.overall_consistent:
        logger.info("PASS — providers are consistent across %d roots.", len(roots))
        return 0
    else:
        logger.warning("FAIL — flags raised:")
        for f in report.flags:
            logger.warning("  • %s", f)
        return 1


if __name__ == "__main__":
    sys.exit(main())
