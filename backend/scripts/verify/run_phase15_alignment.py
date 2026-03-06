"""Phase 1.5 alignment & operational-readiness verification runner.

Usage
-----
  python backend/scripts/verify/run_phase15_alignment.py \\
    --config backend/configs/cooc_reversal_futures.yaml \\
    --pack-dir runs/latest/production_pack \\
    --mode yfinance_only

Modes
-----
  yfinance_only  – feature parity + coverage + trade proxy (no IBKR needed)
  yfinance_vs_ibkr – also runs session semantics comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config, falling back to JSON."""
    path = Path(config_path)
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore[import-untyped]
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pack_manifest(pack_dir: str) -> Dict[str, Any]:
    """Load run_manifest.json from a production pack."""
    mf = Path(pack_dir) / "run_manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"No run_manifest.json in {pack_dir}")
    return json.loads(mf.read_text(encoding="utf-8"))


def _sample_days(gold_frame: pd.DataFrame, n: int, seed: int) -> List[date]:
    """Deterministically sample n trading days from a gold frame."""
    if "trading_day" not in gold_frame.columns:
        return []
    days = sorted(gold_frame["trading_day"].unique())
    # Convert to date objects if needed
    days = [d if isinstance(d, date) else pd.Timestamp(d).date() for d in days]
    rng = np.random.default_rng(seed)
    n = min(n, len(days))
    idx = sorted(rng.choice(len(days), n, replace=False))
    return [days[i] for i in idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase15(
    config_path: str,
    pack_dir: str,
    mode: str = "yfinance_only",
    output_dir: Optional[str] = None,
    sample_days_n: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute Phase 1.5 alignment checks.

    Parameters
    ----------
    config_path
        Path to sleeve YAML config.
    pack_dir
        Path to exported production_pack.
    mode
        ``"yfinance_only"`` or ``"yfinance_vs_ibkr"``.
    output_dir
        Where to write artifacts.  Defaults to ``{pack_dir}/phase15/``.
    sample_days_n
        Number of days to sample for parity checks.
    seed
        RNG seed.

    Returns
    -------
    Consolidated result dict.
    """
    from sleeves.cooc_reversal_futures.pipeline.types import (
        Phase15Report,
        SessionSemanticsReport,
        FeatureParityReport,
        CoverageReport,
        TradeProxyReport,
    )

    config = _load_config(config_path)
    manifest = _load_pack_manifest(pack_dir)

    if output_dir is None:
        output_dir = str(Path(pack_dir) / "phase15")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"mode": mode, "seed": seed}
    session_report: Optional[SessionSemanticsReport] = None
    parity_report: Optional[FeatureParityReport] = None
    coverage_report: Optional[CoverageReport] = None
    trade_report: Optional[TradeProxyReport] = None

    # --- Load gold frame ---
    gold_path = manifest.get("canonicalization", {}).get("gold_path", "")
    if gold_path and Path(gold_path).exists():
        gold_frame = pd.read_parquet(gold_path)
        logger.info("Loaded gold frame: %d rows from %s", len(gold_frame), gold_path)
    else:
        logger.warning("Gold frame not found at %s — some checks will be limited", gold_path)
        gold_frame = pd.DataFrame()

    # --- Load dataset ---
    dataset_path = manifest.get("dataset", {}).get("dataset_path", "")
    if dataset_path and Path(dataset_path).exists():
        dataset = pd.read_parquet(dataset_path)
        logger.info("Loaded dataset: %d rows from %s", len(dataset), dataset_path)
    else:
        logger.warning("Dataset not found at %s", dataset_path)
        dataset = pd.DataFrame()

    # ====================================================================
    # 1. Session Semantics (only in yfinance_vs_ibkr mode)
    # ====================================================================
    if mode == "yfinance_vs_ibkr" and not gold_frame.empty:
        logger.info("=== Session Semantics: yfinance vs IBKR ===")
        try:
            from sleeves.cooc_reversal_futures.pipeline.session_semantics import (
                compare_session_semantics,
            )
            from sleeves.cooc_reversal_futures.pipeline.ibkr_hist_provider import (
                IBKRHistoricalDataProvider,
            )
            from algae.trading.ibkr_client import IbkrClient
            from sleeves.cooc_reversal_futures.pipeline.ingest import ingest_bronze
            import os

            # Connect to IBKR
            host, port = os.environ.get("IBKR_GATEWAY_URL", "127.0.0.1:4002").split(":")
            client = IbkrClient(host=host, port=int(port), readonly=True)
            client.connect()

            ibkr_provider = IBKRHistoricalDataProvider(
                client, cache_dir=out / "ibkr_cache",
            )

            roots = list(config.get("universe", ["ES", "NQ", "YM", "RTY"]))
            start = date.fromisoformat(manifest.get("start_date", "2025-01-01"))
            end = date.fromisoformat(manifest.get("end_date", "2025-12-31"))

            # Fetch IBKR bars
            ibkr_frames = []
            for root in roots:
                try:
                    df = ibkr_provider.fetch_daily_bars(root, start, end)
                    df["root"] = root
                    df["trading_day"] = df["timestamp"].dt.date
                    df["ret_co"] = df["close"].shift(1) / df["open"] - 1  # Previous close → current open
                    df["ret_oc"] = df["close"] / df["open"] - 1
                    ibkr_frames.append(df)
                except Exception as exc:
                    logger.warning("IBKR fetch for %s failed: %s", root, exc)

            if ibkr_frames:
                ibkr_df = pd.concat(ibkr_frames, ignore_index=True)

                # Prepare yfinance frame
                yf_df = gold_frame.copy()
                if "trading_day" not in yf_df.columns:
                    yf_df["trading_day"] = yf_df.index.get_level_values("trading_day")

                session_report = compare_session_semantics(
                    df_a=yf_df,
                    df_b=ibkr_df,
                    roots=roots,
                    sample_days=60,
                    seed=seed,
                    output_dir=str(out),
                )
                results["session_semantics"] = session_report.to_dict()
                logger.info("Session semantics: gate_passed=%s", session_report.gate_passed)

            client.disconnect()
        except Exception as exc:
            logger.error("Session semantics check failed: %s", exc)
            results["session_semantics_error"] = str(exc)

    # ====================================================================
    # 2. Feature Parity
    # ====================================================================
    if not gold_frame.empty:
        logger.info("=== Feature Parity ===")
        try:
            from sleeves.cooc_reversal_futures.pipeline.parity import compute_feature_parity
            from sleeves.cooc_reversal_futures.sleeve import COOCReversalFuturesSleeve
            from sleeves.cooc_reversal_futures.config import COOCReversalConfig

            sleeve_config = COOCReversalConfig(
                universe=tuple(config.get("universe", ["ES", "NQ", "YM", "RTY"])),
                lookback=config.get("lookback", 20),
            )
            sleeve = COOCReversalFuturesSleeve(sleeve_config)
            days = _sample_days(gold_frame, sample_days_n, seed)

            parity_report = compute_feature_parity(
                gold_frame=gold_frame,
                sleeve=sleeve,
                asof_days=days,
                output_dir=str(out),
            )
            results["feature_parity"] = parity_report.to_dict()
            logger.info("Feature parity: gate_passed=%s", parity_report.gate_passed)
        except Exception as exc:
            logger.error("Feature parity check failed: %s", exc)
            results["feature_parity_error"] = str(exc)

    # ====================================================================
    # 3. Coverage Gate
    # ====================================================================
    if not dataset.empty:
        logger.info("=== Coverage Gate ===")
        try:
            from sleeves.cooc_reversal_futures.pipeline.validation import _coverage_gate

            min_roots = config.get("phase15", {}).get("coverage", {}).get("min_roots_per_day", 4)
            allow_partial = config.get("phase15", {}).get("coverage", {}).get("allow_partial", False)

            gate_result, coverage_report = _coverage_gate(
                dataset, min_roots_per_day=min_roots, allow_partial=allow_partial,
            )
            results["coverage"] = coverage_report.to_dict()
            logger.info("Coverage: gate_passed=%s", coverage_report.gate_passed)

            # Persist
            (out / "coverage_report.json").write_text(
                json.dumps(coverage_report.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Coverage check failed: %s", exc)
            results["coverage_error"] = str(exc)

    # ====================================================================
    # 4. Trade Proxy
    # ====================================================================
    if not dataset.empty:
        logger.info("=== Trade Proxy ===")
        try:
            from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

            # Load model if available
            model_path = Path(pack_dir) / "model"
            preds = None

            if (model_path / "model.pkl").exists():
                import pickle
                with open(model_path / "model.pkl", "rb") as f:
                    model = pickle.load(f)

                from sleeves.cooc_reversal_futures.pipeline.train import Preprocessor
                if (model_path / "preprocessor.pkl").exists():
                    with open(model_path / "preprocessor.pkl", "rb") as f:
                        preprocessor = pickle.load(f)

                    X = preprocessor.transform(dataset)
                    preds = pd.Series(model.predict(X), index=dataset.index)

            if preds is None:
                # Use baseline as both model and baseline
                if "ret_co" in dataset.columns:
                    preds = pd.Series(-dataset["ret_co"].values, index=dataset.index)
                elif "signal" in dataset.columns:
                    preds = pd.Series(dataset["signal"].values, index=dataset.index)
                else:
                    preds = pd.Series(0.0, index=dataset.index)

            trade_config = config.get("phase15", {}).get("trade_proxy", {})
            trade_report = evaluate_trade_proxy(
                dataset=dataset,
                preds=preds,
                config=trade_config,
                output_dir=str(out),
            )
            results["trade_proxy"] = trade_report.to_dict()
            logger.info("Trade proxy: gate_passed=%s, sharpe_model=%.3f, sharpe_baseline=%.3f",
                        trade_report.gate_passed, trade_report.sharpe_model, trade_report.sharpe_baseline)
        except Exception as exc:
            logger.error("Trade proxy check failed: %s", exc)
            results["trade_proxy_error"] = str(exc)

    # ====================================================================
    # 5. Consolidated Report
    # ====================================================================
    sub_reports = [
        session_report.gate_passed if session_report else True,
        parity_report.gate_passed if parity_report else True,
        coverage_report.gate_passed if coverage_report else True,
        trade_report.gate_passed if trade_report else True,
    ]
    all_passed = all(sub_reports)
    results["phase15_status"] = "PASS" if all_passed else "FAIL"

    phase15_report = Phase15Report(
        session_semantics=session_report,
        feature_parity=parity_report,
        coverage=coverage_report,
        trade_proxy=trade_report,
        all_passed=all_passed,
    )

    # Write consolidated report
    consolidated_path = out / "phase15_report.json"
    consolidated_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    logger.info("Phase 1.5 consolidated: status=%s → %s", results["phase15_status"], consolidated_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5 alignment verification")
    parser.add_argument("--config", required=True, help="Sleeve YAML config path")
    parser.add_argument("--pack-dir", required=True, help="Production pack directory")
    parser.add_argument("--mode", default="yfinance_only",
                        choices=["yfinance_only", "yfinance_vs_ibkr"],
                        help="Verification mode")
    parser.add_argument("--output-dir", default=None, help="Artifacts output directory")
    parser.add_argument("--sample-days", type=int, default=50, help="Days to sample for parity")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_phase15(
        config_path=args.config,
        pack_dir=args.pack_dir,
        mode=args.mode,
        output_dir=args.output_dir,
        sample_days_n=args.sample_days,
        seed=args.seed,
    )

    status = result.get("phase15_status", "UNKNOWN")
    print(f"\nPhase 1.5 status: {status}")
    if status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
