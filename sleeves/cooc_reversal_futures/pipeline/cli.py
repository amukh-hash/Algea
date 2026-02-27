"""CLI entrypoint: ingest → validate → canonicalize → dataset → split → train → validate → export."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from ..config import COOCReversalConfig, load_config_from_yaml

from .bronze_validate import persist_validation_report as persist_bronze_report
from .bronze_validate import validate_bronze_bars
from .canonicalize import (
    build_contract_map,
    build_gold_frame,
    build_silver_bars,
    normalize_bars,
    persist_canonicalized,
)
from .dataset import assemble_dataset
from .export import export_production_pack
from .ingest import CsvDataProvider, ingest_bronze
from .splits import persist_splits, time_based_split, walk_forward_cv
from .train import save_model_bundle, train_model
from .types import RunManifest, SplitSpec
from .validation import persist_validation_report, run_validation


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cooc_pipeline",
        description="CO→OC reversal futures end-to-end pipeline",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path (optional)")
    parser.add_argument("--bronze-dir", type=str, default="data_lake/bronze", help="Bronze data directory")
    parser.add_argument("--output-dir", type=str, default="data_lake/runs", help="Output runs directory")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory (overrides output-dir + auto ID)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data_lake/raw", help="Raw data source directory")
    parser.add_argument("--data-provider", type=str, default="csv",
                        choices=["csv", "ibkr_hist", "yfinance", "hybrid"], help="Data provider")
    parser.add_argument("--ibkr-port", type=int, default=4001,
                        help="IBKR Gateway/TWS port (4001=Gateway live, 4002=Gateway paper, 7497=TWS)")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> RunManifest:
    """Run the full pipeline: ingest → validate → canonicalize → dataset → split → train → validate → export."""
    args = _parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    seed = args.seed

    # Load config from YAML if provided, otherwise use defaults
    if args.config:
        config = load_config_from_yaml(args.config)
        print(f"  Config loaded from: {args.config}")
    else:
        config = COOCReversalConfig()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_hash = ""
    roots = sorted(config.universe)
    data_provider_name = args.data_provider

    print(f"=== CO→OC Pipeline Run: {run_id} ===")
    print(f"  Roots: {roots}")
    print(f"  Date range: {start} → {end}")
    print(f"  Seed: {seed}")
    print(f"  Data provider: {data_provider_name}")
    print(f"  Schema version: {config.schema_version}")
    print(f"  Output: {run_dir}")
    print()

    # ------------------------------------------------------------------
    # 1. INGEST
    # ------------------------------------------------------------------
    print("[1/10] Ingesting bronze data...")
    if data_provider_name in ("ibkr_hist", "hybrid"):
        try:
            from algea.trading.ibkr_client import IbkrClient
            from .ibkr_hist_provider import IBKRHistoricalDataProvider
            client = IbkrClient(port=args.ibkr_port)
            client.connect()
            ibkr_provider = IBKRHistoricalDataProvider(
                client,
                cache_dir=Path(args.data_dir) / "ibkr_cache",
            )
            print(f"  ✓ Connected to IBKR Gateway")

            if data_provider_name == "hybrid":
                from .yfinance_provider import YFinanceDataProvider
                from .hybrid_provider import HybridDataProvider
                yf_provider = YFinanceDataProvider(
                    cache_dir=Path(args.data_dir) / "yfinance_cache",
                )
                provider = HybridDataProvider(ibkr_provider, yf_provider)
                print("  ✓ Hybrid mode: IBKR primary + yfinance fallback")
            else:
                provider = ibkr_provider
        except Exception as e:
            print(f"  ✗ IBKR connection failed: {e}")
            print(f"  Falling back to CSV provider")
            data_provider_name = "csv"
            provider = CsvDataProvider(args.data_dir)
    elif data_provider_name == "yfinance":
        from .yfinance_provider import YFinanceDataProvider
        provider = YFinanceDataProvider(
            cache_dir=Path(args.data_dir) / "yfinance_cache",
        )
        print("  ✓ Using Yahoo Finance data provider")
    else:
        provider = CsvDataProvider(args.data_dir)

    bronze_dir = run_dir / "bronze"
    try:
        bronze_manifest = ingest_bronze(provider, roots, start, end, bronze_dir)
        skipped = set(roots) - set(bronze_manifest.roots)
        print(f"  ✓ Ingested {len(bronze_manifest.roots)} roots")
        if skipped:
            print(f"  ⚠ Skipped {len(skipped)} roots (no data): {sorted(skipped)}")
    except FileNotFoundError as e:
        print(f"  ⚠ Data source not found: {e}")
        if data_provider_name in ("ibkr_hist", "hybrid"):
            raise  # IBKR provider failures should not fall back to synthetic
        print("  Generating synthetic data for demonstration...")
        _generate_synthetic_data(args.data_dir, roots, start, end, seed)
        provider = CsvDataProvider(args.data_dir)  # re-create after generation
        bronze_manifest = ingest_bronze(provider, roots, start, end, bronze_dir)
        print(f"  ✓ Ingested {len(bronze_manifest.roots)} roots (synthetic)")

    # Print hybrid source breakdown
    if data_provider_name == "hybrid" and hasattr(provider, "source_map"):
        ibkr_roots = sorted(r for r, s in provider.source_map.items() if s == "ibkr_hist")
        yf_roots = sorted(r for r, s in provider.source_map.items() if s == "yfinance")
        print(f"  📊 IBKR: {len(ibkr_roots)} roots ({', '.join(ibkr_roots)})")
        print(f"  📊 yfinance: {len(yf_roots)} roots ({', '.join(yf_roots)})")

    # ------------------------------------------------------------------
    # 2. BRONZE VALIDATE
    # ------------------------------------------------------------------
    print("[2/10] Validating bronze bars...")
    all_bronze_ok = True
    for root in bronze_manifest.roots:
        root_bars = pd.read_parquet(bronze_manifest.paths[root])
        report = validate_bronze_bars(root_bars, root)
        persist_bronze_report(report, run_dir / "bronze_validation")
        if not report.ok:
            print(f"  ✗ {root}: {report.violations}")
            all_bronze_ok = False
        else:
            print(f"  ✓ {root}: {report.row_count} rows, {len(report.gap_report)} gaps")
    if not all_bronze_ok:
        print("  ⚠ Bronze validation failures detected")

    # ------------------------------------------------------------------
    # 3. CANONICALIZE
    # ------------------------------------------------------------------
    print("[3/10] Canonicalizing bars...")
    # Collect all bronze bars with root column
    all_bars: List[pd.DataFrame] = []
    for root in bronze_manifest.roots:
        df = pd.read_parquet(bronze_manifest.paths[root])
        df["root"] = root
        df = normalize_bars(df)
        all_bars.append(df)
    combined_bars = pd.concat(all_bars, ignore_index=True)

    contract_map = build_contract_map(roots, start, end)
    silver = build_silver_bars(combined_bars, contract_map)
    gold = build_gold_frame(silver)

    canon_dir = run_dir / "canonical"
    canon_manifest = persist_canonicalized(silver, gold, contract_map, canon_dir)
    print(f"  ✓ {canon_manifest.row_count} gold rows, {canon_manifest.trading_days} trading days")

    # ------------------------------------------------------------------
    # 4. DATASET
    # ------------------------------------------------------------------
    print("[4/10] Assembling dataset...")
    dataset_dir = run_dir / "dataset"
    dataset, dataset_manifest = assemble_dataset(
        gold, config, lookback=config.lookback, output_dir=dataset_dir,
    )
    print(f"  ✓ {dataset_manifest.row_count} rows, {len(dataset_manifest.feature_columns)} features")
    config_hash = dataset_manifest.config_hash

    # ------------------------------------------------------------------
    # 5. SPLITS
    # ------------------------------------------------------------------
    print("[5/10] Computing CV splits...")
    cv_splits = walk_forward_cv(
        dataset,
        fold_size_days=config.cv.fold_size_days,
        embargo_days=config.cv.embargo_days,
    )
    persist_splits(cv_splits, run_dir / "splits")
    print(f"  ✓ {len(cv_splits)} walk-forward folds")

    # ------------------------------------------------------------------
    # 6. TRAIN
    # ------------------------------------------------------------------
    print("[6/10] Training model...")
    if cv_splits:
        bundle_info, model, preprocessor = train_model(config, dataset, cv_splits, seed=seed)  # type: ignore[misc]
        model_dir = run_dir / "model"
        bundle = save_model_bundle(bundle_info, model, preprocessor, model_dir)
        print(f"  ✓ Best params: {bundle.chosen_params}")
        print(f"  ✓ {bundle.primary_metric}={bundle.primary_metric_value:.4f}")
    else:
        print("  ⚠ No CV splits available, skipping training")
        bundle = None
        model = None
        preprocessor = None

    # ------------------------------------------------------------------
    # 7. VALIDATE
    # ------------------------------------------------------------------
    print("[7/10] Running validation...")
    require_ibkr = data_provider_name in ("ibkr_hist", "hybrid")
    if model is not None and preprocessor is not None and bundle is not None:
        # Extract promotion config
        promo = config.promotion
        promo_windows = list(promo.windows) if promo.windows else None

        val_report = run_validation(
            bundle, model, preprocessor, dataset, cv_splits, config,
            contract_map=contract_map,
            require_ibkr=require_ibkr or promo.require_ibkr,
            min_sharpe_delta=promo.min_sharpe_delta,
            max_drawdown_tolerance=promo.max_drawdown_tolerance,
            worst_1pct_tolerance_bps=promo.worst_1pct_tolerance_bps,
            min_hit_rate=promo.min_hit_rate,
            promotion_windows=promo_windows,
            stress_required=promo.stress_required,
            provider_name=data_provider_name,
        )
        persist_validation_report(val_report, run_dir / "validation")
        for g in val_report.gates:
            status = "✓" if g.passed else "✗"
            print(f"  {status} {g.name}: {g.detail}")
        if val_report.all_passed:
            print("  ✓ All gates PASSED")
        else:
            print("  ✗ Some gates FAILED")
    else:
        val_report = None
        print("  ⚠ Skipped (no model)")

    # ------------------------------------------------------------------
    # 8. RISK CALIBRATION (informational)
    # ------------------------------------------------------------------
    print("[8/10] Risk calibration check...")
    if model is not None and preprocessor is not None:
        try:
            from .risk_calibration import run_risk_calibration
            from .train import FEATURE_COLUMNS
            import numpy as np
            features = list(FEATURE_COLUMNS)
            last_split = cv_splits[-1] if cv_splits else None
            if last_split:
                X_all = preprocessor.transform(dataset)
                preds_all = model.predict(X_all)
                rc_report = run_risk_calibration(dataset, preds_all)
                rc_path = run_dir / "risk_calibration_report.json"
                rc_path.write_text(json.dumps(rc_report, indent=2, default=str), encoding="utf-8")
                corr = rc_report.get("spearman_correlation", 0)
                mono = rc_report.get("monotonicity_ok", False)
                print(f"  ✓ Spearman corr={corr:.4f}, monotonicity={'✓' if mono else '✗'}")
            else:
                print("  ⚠ Skipped (no splits)")
        except Exception as e:
            print(f"  ⚠ Risk calibration failed: {e}")
    else:
        print("  ⚠ Skipped (no model)")

    # ------------------------------------------------------------------
    # 9. SCORE MODE COMPARISON (informational)
    # ------------------------------------------------------------------
    print("[9/10] Score mode comparison...")
    if model is not None and preprocessor is not None:
        try:
            from .score_mode_comparison import run_score_mode_comparison
            smc_report = run_score_mode_comparison(dataset, model, preprocessor, cv_splits, config)
            smc_path = run_dir / "score_mode_comparison.json"
            smc_path.write_text(json.dumps(smc_report, indent=2, default=str), encoding="utf-8")
            best = smc_report.get("best_mode", "unknown")
            print(f"  ✓ Best mode: {best}")
        except Exception as e:
            print(f"  ⚠ Score mode comparison failed: {e}")
    else:
        print("  ⚠ Skipped (no model)")

    # ------------------------------------------------------------------
    # 8. EXPORT
    # ------------------------------------------------------------------
    print("[10/10] Exporting production pack...")
    run_manifest = RunManifest(
        run_id=run_id,
        run_dir=str(run_dir),
        seed=seed,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        config_hash=config_hash,
        bronze=bronze_manifest,
        canonicalization=canon_manifest,
        dataset=dataset_manifest,
        splits=tuple(cv_splits),
        model=bundle,
        validation=val_report,
    )
    pack_dir = export_production_pack(run_manifest, run_dir)
    print(f"  ✓ Production pack: {pack_dir}")

    # --- Summary ---
    print()
    print("=== Run Summary ===")
    print(f"  Run ID:     {run_id}")
    print(f"  Run dir:    {run_dir}")
    print(f"  Pack dir:   {pack_dir}")
    if val_report is not None:
        print(f"  Baseline IC: {val_report.baseline_ic:.4f}")
        print(f"  Model IC:    {val_report.model_ic:.4f}")
        print(f"  All gates:   {'PASSED' if val_report.all_passed else 'FAILED'}")

    return run_manifest


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing / demo)
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    data_dir: str,
    roots: List[str],
    start: date,
    end: date,
    seed: int,
) -> None:
    """Generate synthetic OHLCV CSV files for testing."""
    import numpy as np

    np.random.seed(seed)
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    trading_days = pd.bdate_range(start, end)

    for root in sorted(roots):
        n = len(trading_days)
        base_price = {
            "ES": 4500.0, "NQ": 15000.0, "YM": 35000.0, "RTY": 2000.0,
            "CL": 75.0, "GC": 2000.0, "SI": 25.0, "HG": 4.0,
            "ZN": 110.0, "ZB": 120.0,
            "6E": 1.10, "6J": 0.0072, "6B": 1.27, "6A": 0.67,
        }.get(root, 5000.0)
        returns = np.random.normal(0, 0.01, n)
        prices = base_price * np.cumprod(1 + returns)

        opens = prices * (1 + np.random.normal(0, 0.002, n))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.003, n)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.003, n)))
        volumes = np.random.poisson(100000, n)

        df = pd.DataFrame({
            "timestamp": trading_days.tz_localize("UTC"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        })
        df.to_csv(out / f"{root}.csv", index=False)


if __name__ == "__main__":
    main()
