"""
Phase 6.1: Contract / Regression Tests for Algaie Artifacts.

Tests fail on schema/key/dtype drift or partial builds.
Uses pytest. Skips gracefully if artifacts not present on CI.
Resolves roots via pathmap (no hardcoded paths).
Uses Polars scan_parquet with recursive globs for Hive datasets.
"""

import pytest
import polars as pl
import os
import sys
import datetime
from pathlib import Path

from backend.app.ops import pathmap
from backend.app.data import schema_contracts


# ---------------------------------------------------------------------------
# 1) UniverseFrame Contract
# ---------------------------------------------------------------------------

def test_universe_frame_contract():
    root = pathmap.get_universe_frame_root(version="v2")
    if not root.exists():
        pytest.skip(f"UniverseFrame root not found: {root}")

    print(f"Scanning UniverseFrame from {root}")
    try:
        lf = pl.scan_parquet(str(root / "**/*.parquet"))
    except Exception as e:
        pytest.fail(f"Failed to scan UniverseFrame: {e}")

    # Collect schema to check column names + dtypes
    schema = lf.collect_schema()

    # --- symbol column ---
    if "ticker" in schema and "symbol" not in schema:
        lf = lf.rename({"ticker": "symbol"})
        schema = lf.collect_schema()

    assert "symbol" in schema, "UniverseFrame missing 'symbol' column"

    # --- date dtype ---
    assert schema["date"] == pl.Date, (
        f"UniverseFrame 'date' column must be pl.Date, got {schema['date']}"
    )

    # --- duplicate key check (memory-safe via group_by + limit) ---
    dupes = (
        lf.group_by("date", "symbol")
        .len()
        .filter(pl.col("len") > 1)
        .limit(1)
        .collect()
    )
    assert len(dupes) == 0, f"Found duplicate (date,symbol) keys in UniverseFrame: {dupes}"

    # --- weight invariant: (weight > 0) iff is_tradable ---
    violations = (
        lf.filter((pl.col("weight") > 0) != pl.col("is_tradable"))
        .limit(5)
        .collect()
    )
    assert len(violations) == 0, (
        f"Weight/tradable invariant violated: {violations}"
    )

    # --- breadth sanity ---
    breadth_df = (
        lf.filter(pl.col("is_tradable"))
        .group_by("date")
        .len()
        .collect()
    )

    n_unique_dates = len(breadth_df)
    median_breadth = breadth_df["len"].median()

    print(
        f"UniverseFrame: {n_unique_dates} tradable dates, "
        f"median breadth: {median_breadth}"
    )

    assert median_breadth > 50, (
        f"UniverseFrame median tradable breadth too low: {median_breadth} (expected > 50)"
    )
    assert n_unique_dates > 4000, (
        f"UniverseFrame has too few unique dates: {n_unique_dates} (expected > 4000 for 2006-2025)"
    )


# ---------------------------------------------------------------------------
# 2) SelectorFeatureFrame Contract
# ---------------------------------------------------------------------------

def test_selector_features_contract():
    # Try horizon-specific dir first, then base
    root = pathmap.get_selector_features_root(version="v2", horizon="5")
    if not root.exists():
        root = pathmap.get_selector_features_root(version="v2")

    if not root.exists():
        pytest.skip(f"SelectorFeatures root not found: {root}")

    print(f"Scanning SelectorFeatures from {root}")
    lf = pl.scan_parquet(str(root / "**/*.parquet"))
    schema = lf.collect_schema()

    # Normalize ticker -> symbol
    if "ticker" in schema and "symbol" not in schema:
        lf = lf.rename({"ticker": "symbol"})
        schema = lf.collect_schema()

    # --- required columns ---
    required = schema_contracts.SELECTOR_FEATURES_V2_REQUIRED_COLS
    missing = [c for c in required if c not in schema]
    assert not missing, f"SelectorFeatures missing columns: {missing}"

    # --- date dtype ---
    assert schema["date"] == pl.Date, (
        f"SelectorFeatures 'date' must be pl.Date, got {schema['date']}"
    )

    # --- x_* finite and within [-1.001, 1.001] ---
    feature_cols = ["x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol"]
    for col in feature_cols:
        if col not in schema:
            continue
        bad_rows = (
            lf.select(pl.col(col))
            .filter((pl.col(col).abs() > 1.001) | pl.col(col).is_infinite())
            .limit(1)
            .collect()
        )
        assert len(bad_rows) == 0, (
            f"Feature {col} has values outside [-1.001, 1.001] or infinite"
        )

    # --- n_unique_dates > 3000 ---
    dates_df = lf.group_by("date").len().collect()
    n_unique_dates = len(dates_df)
    median_breadth = dates_df["len"].median()

    print(
        f"SelectorFeatures: {n_unique_dates} unique dates, "
        f"median breadth: {median_breadth}"
    )
    assert n_unique_dates > 2000, (
        f"SelectorFeatures too few dates: {n_unique_dates} (expected > 2000)"
    )
    assert median_breadth > 50, (
        f"SelectorFeatures median breadth too low: {median_breadth}"
    )

    # --- weight > 0 for all rows ---
    min_weight = lf.select(pl.col("weight").min()).collect().item()
    assert min_weight > 0, (
        f"SelectorFeatures found non-positive weights (min={min_weight})"
    )


# ---------------------------------------------------------------------------
# 3) Priors Contract (Versioned)
# ---------------------------------------------------------------------------

def test_priors_contract():
    as_of_date_str = os.getenv("AS_OF_DATE")
    priors_version = os.getenv("PRIORS_VERSION")

    # Resolve path
    resolved_path = None
    try:
        if as_of_date_str:
            resolved_path = pathmap.resolve(
                "priors_date",
                date=as_of_date_str,
                version=priors_version or "latest",
            )
        else:
            priors_root = pathmap.get_priors_root()
            if not priors_root.exists():
                pytest.skip("Priors root does not exist")
            date_dirs = sorted(
                [d.name.split("=")[1] for d in priors_root.glob("date=*")]
            )
            if not date_dirs:
                pytest.skip("No date partitions found in priors")
            latest_date = date_dirs[-1]
            resolved_path = pathmap.resolve(
                "priors_date", date=latest_date, version=priors_version or "latest"
            )
            as_of_date_str = latest_date
            print(f"Auto-selected latest priors date: {latest_date}")
    except Exception as e:
        pytest.skip(f"Could not resolve priors path: {e}")

    priors_path = Path(resolved_path)
    if not priors_path.exists():
        pytest.skip(f"Priors artifact not found at {priors_path}")

    print(f"Scanning Priors from {priors_path}")
    df = pl.read_parquet(str(priors_path))
    df = schema_contracts.normalize_keys(df)

    # --- required schema columns ---
    required = schema_contracts.PRIORS_REQUIRED_COLS
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Priors missing columns: {missing}"

    # --- version check ---
    if priors_version and priors_version != "latest":
        mismatches = df.filter(pl.col("prior_version") != priors_version).height
        assert mismatches == 0, (
            f"prior_version mismatch: expected {priors_version}, "
            f"found other values in {mismatches} rows"
        )

    # --- coverage check ---
    dates_in_priors = df.select("date").unique().get_column("date").to_list()
    assert len(dates_in_priors) >= 1, "Priors file is empty (no dates)"

    priors_date = dates_in_priors[0]

    univ_root = pathmap.get_universe_frame_root(version="v2")
    if not univ_root.exists():
        print("UniverseFrame not found, skipping coverage check")
        return

    univ_lf = pl.scan_parquet(str(univ_root / "**/*.parquet"))
    # Normalize
    if "ticker" in univ_lf.collect_schema() and "symbol" not in univ_lf.collect_schema():
        univ_lf = univ_lf.rename({"ticker": "symbol"})

    tradable_syms = (
        univ_lf.filter(
            (pl.col("date") == priors_date) & pl.col("is_tradable")
        )
        .select("symbol")
        .collect()
        .get_column("symbol")
        .to_list()
    )

    if not tradable_syms:
        print(f"No tradable universe found for {priors_date}, skipping coverage")
        return

    priors_syms = set(df.get_column("symbol").to_list())
    common = set(tradable_syms).intersection(priors_syms)
    coverage = len(common) / len(tradable_syms)

    threshold = float(os.getenv("MIN_COVERAGE", "0.95"))
    print(
        f"Priors coverage: {coverage:.2%} ({len(common)}/{len(tradable_syms)})"
    )
    assert coverage >= threshold, (
        f"Priors coverage {coverage:.2%} below threshold {threshold}"
    )


# ---------------------------------------------------------------------------
# 4) ChronosDataset Contract
# ---------------------------------------------------------------------------

def test_chronos_dataset_contract():
    from backend.app.training.chronos_dataset import ChronosDataset

    gold_dir = pathmap.get_gold_daily_root()
    univ_root = pathmap.get_universe_frame_root(version="v2")

    if not gold_dir.exists():
        pytest.skip(f"Gold root not found: {gold_dir}")
    if not univ_root.exists():
        pytest.skip(f"Universe root not found: {univ_root}")

    # Gather gold files (limit for speed)
    max_files = int(os.getenv("CHRONOS_TEST_MAX_FILES", "50"))
    gold_files = sorted(gold_dir.glob("*.parquet"))[:max_files]
    if not gold_files:
        pytest.skip("No gold parquet files found")

    context_len = int(os.getenv("CHRONOS_CONTEXT_LEN", "180"))
    prediction_len = int(os.getenv("CHRONOS_PRED_LEN", "20"))
    stride = int(os.getenv("CHRONOS_STRIDE", "5"))
    max_spf_str = os.getenv("CHRONOS_MAX_SAMPLES_PER_FILE", "10")
    max_spf = int(max_spf_str) if max_spf_str else None

    ds = ChronosDataset(
        files=gold_files,
        context_len=context_len,
        prediction_len=prediction_len,
        stride=stride,
        universe_path=str(univ_root / "**/*.parquet"),
        target_col=os.getenv("CHRONOS_TARGET_COL", "close"),
        max_samples_per_file=max_spf,
        seed=42,
    )

    # --- len > 0 ---
    assert len(ds) > 0, "ChronosDataset is empty (no valid windows)"

    # --- stats exist ---
    assert hasattr(ds, "stats"), "ChronosDataset missing 'stats' attribute"
    assert ds.stats["n_final_samples"] > 0, (
        f"n_final_samples is {ds.stats['n_final_samples']}"
    )

    # --- sample up to 25 items ---
    n_sample = min(25, len(ds))
    negative_seen = False

    for i in range(n_sample):
        item = ds[i]
        assert isinstance(item, dict), "ChronosDataset.__getitem__ must return dict"

        for key in ("past_target", "future_target", "scale"):
            assert key in item, f"Missing key '{key}' in item"

        past = item["past_target"]
        fut = item["future_target"]

        # Shape [L, 1] and [H, 1]
        assert past.shape == (context_len, 1), (
            f"past_target shape mismatch: {past.shape}"
        )
        assert fut.shape == (prediction_len, 1), (
            f"future_target shape mismatch: {fut.shape}"
        )

        if (past < 0).any().item():
            negative_seen = True

    assert negative_seen, (
        "No negative values in past_target across sampled windows; "
        "expected for relative-log modeling"
    )
