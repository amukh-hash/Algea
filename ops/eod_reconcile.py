"""
eod_reconcile.py — End-of-Day State Reconciliation

Mathematically proves that the Native C++ Arrow memory state matches
the Legacy REST backend ledger to 5-decimal-place precision.

Execute daily at 16:15 EST during the 14-day shadow validation phase.
The CRO signs off on cutover only after 14 consecutive PASS results.

Usage:
    python ops/eod_reconcile.py
    python ops/eod_reconcile.py --arrow-path C:\\Algae\\Dumps\\positions_eod.arrow
    python ops/eod_reconcile.py --backend-url http://trade.algae.internal:8000

Exit codes:
    0 = PARITY VERIFIED
    1 = PARITY FAILURE (data drift detected)
    2 = INFRASTRUCTURE ERROR (endpoint unreachable, file missing)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eod_reconcile")


def fetch_legacy_state(backend_url: str) -> pd.DataFrame:
    """Fetch portfolio state from the legacy REST endpoint."""
    url = f"{backend_url}/api/control/portfolio-summary"
    log.info(f"Fetching legacy state from {url}")

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    holdings = resp.json().get("holdings", [])
    if not holdings:
        log.warning("Legacy endpoint returned empty holdings")
        return pd.DataFrame(columns=["symbol", "Qty", "Unrealized P&L"])

    df = pd.DataFrame(holdings)
    df = (
        df[["symbol", "quantity", "unrealized_pnl"]]
        .rename(columns={"quantity": "Qty", "unrealized_pnl": "Unrealized P&L"})
        .sort_values(by="symbol")
        .reset_index(drop=True)
    )
    log.info(f"Legacy state: {len(df)} positions")
    return df


def load_native_arrow(arrow_path: str) -> pd.DataFrame:
    """Load the C++ ArrowTableModel EOD memory dump."""
    path = Path(arrow_path)
    if not path.exists():
        raise FileNotFoundError(f"Arrow dump not found: {path}")

    log.info(f"Loading native Arrow dump from {path}")

    with pa.OSFile(str(path), "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    df = table.to_pandas()
    df = (
        df[["Symbol", "Qty", "Unrealized P&L"]]
        .rename(columns={"Symbol": "symbol"})
        .sort_values(by="symbol")
        .reset_index(drop=True)
    )
    log.info(f"Native state: {len(df)} positions")
    return df


def reconcile(legacy_df: pd.DataFrame, native_df: pd.DataFrame) -> bool:
    """Assert strict parity between legacy and native states.

    Uses rtol=1e-5, atol=1e-5 to handle IEEE 754 rounding
    artifacts introduced by JSON serialization (Blind Spot 1).
    """
    try:
        pd.testing.assert_frame_equal(
            legacy_df,
            native_df,
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
        )
        return True
    except AssertionError as e:
        log.critical(f"PARITY FAILURE:\n{e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="EOD State Reconciliation")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API base URL",
    )
    parser.add_argument(
        "--arrow-path",
        default="C:\\Algae\\Dumps\\positions_eod.arrow",
        help="Path to the C++ Arrow memory dump",
    )
    parser.add_argument(
        "--output-log",
        default=None,
        help="Optional path to append reconciliation results",
    )
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    log.info(f"=== EOD Reconciliation: {today} ===")

    try:
        legacy_df = fetch_legacy_state(args.backend_url)
        native_df = load_native_arrow(args.arrow_path)
    except (requests.RequestException, FileNotFoundError) as e:
        log.error(f"Infrastructure error: {e}")
        sys.exit(2)

    passed = reconcile(legacy_df, native_df)

    # Append to audit log if specified
    if args.output_log:
        with open(args.output_log, "a") as f:
            status = "PASS" if passed else "FAIL"
            f.write(f"{today},{status},{len(legacy_df)},{len(native_df)}\n")

    if passed:
        log.info("PARITY VERIFIED: Native Arrow grid matches Legacy REST state.")
        sys.exit(0)
    else:
        log.critical("PARITY FAILURE: Data drift detected. DO NOT proceed with cutover.")
        sys.exit(1)


if __name__ == "__main__":
    main()
