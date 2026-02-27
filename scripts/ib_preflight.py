#!/usr/bin/env python
"""IB Gateway / TWS preflight connectivity test.

Usage::

    python scripts/ib_preflight.py

Reads IBKR_* environment variables from ``.env`` (via dotenv) or the shell.
Connects, prints account summary, positions, and disconnects.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env if python-dotenv available
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from algea.trading.broker_ibkr import IBKRLiveBroker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    print("=" * 60)
    print("  IBKR Preflight Check")
    print("=" * 60)

    # --- Show config ---
    gw = os.environ.get("IBKR_GATEWAY_URL", "127.0.0.1:7497")
    acct = os.environ.get("IBKR_ACCOUNT_ID", "")
    cid = os.environ.get("IBKR_CLIENT_ID", "17")
    paper = os.environ.get("IBKR_PAPER_ONLY", "1")
    readonly = os.environ.get("IBKR_READONLY", "0")

    print(f"\n  Gateway URL : {gw}")
    print(f"  Account ID  : {acct[:3]}***{acct[-3:] if len(acct) > 6 else '***'}")
    print(f"  Client ID   : {cid}")
    print(f"  Paper Only  : {paper}")
    print(f"  Read-Only   : {readonly}")
    print()

    # --- Connect ---
    try:
        broker = IBKRLiveBroker.from_env()
    except Exception as exc:
        print(f"\n[FAIL] FAILED to create broker: {exc}")
        print("\nCheck:")
        print("  1. TWS / IB Gateway is running")
        print("  2. API -> Settings -> Enable ActiveX and Socket Clients [OK]")
        print("  3. Socket port matches IBKR_GATEWAY_URL")
        print(f"  4. Current IBKR_GATEWAY_URL = {gw}")
        sys.exit(1)

    # --- Account summary ---
    try:
        account = broker.get_account()
        print("[OK] Connection successful!\n")
        print(f"  Net Liquidation Value : ${account.equity:,.2f}")
        print(f"  Cash                  : ${account.cash:,.2f}")
        print(f"  Buying Power          : ${account.buying_power:,.2f}")

        # Margin cushion
        if account.equity > 0:
            margin_cushion = account.buying_power / account.equity
            status = "[OK]" if margin_cushion > 0.30 else "[WARN]"
            print(f"  Margin Cushion        : {margin_cushion:.1%} {status}")
        print()

        # Sleeve capital breakdown
        alloc = {"core": 0.50, "vrp": 0.30, "selector": 0.20}
        print("  Sleeve Capital Split (hardcoded):")
        for name, pct in alloc.items():
            print(f"    {name:12s} : ${account.equity * pct:>12,.2f}  ({pct:.0%})")
        print()

    except Exception as exc:
        print(f"\n[FAIL] Connected but failed to get account: {exc}")
        broker._disconnect()
        sys.exit(1)

    # --- Positions ---
    try:
        positions = broker.get_positions()
        if positions:
            print(f"  Current Positions ({len(positions)}):")
            for pos in positions:
                print(f"    {pos.ticker:12s}  qty={pos.quantity:>6.0f}  avg_cost=${pos.avg_cost:>10,.2f}")
        else:
            print("  Current Positions: (none)")
        print()
    except Exception as exc:
        print(f"  [WARN] Could not fetch positions: {exc}")

    # --- Paper account check ---
    acct_id = os.environ.get("IBKR_ACCOUNT_ID", "")
    if acct_id.startswith("DU") or acct_id.startswith("DF"):
        print("  [OK] Paper account confirmed (DU/DF prefix)")
    else:
        print("  [WARN] Account does not look like paper (expected DU*/DF* prefix)")
    print()

    # --- Disconnect ---
    broker._disconnect()
    print("=" * 60)
    print("  Preflight complete. You are ready to trade.")
    print("=" * 60)


if __name__ == "__main__":
    main()
