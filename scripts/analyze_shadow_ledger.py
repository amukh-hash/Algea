"""
Offline Shadow Ledger Analytics

Parses the Orchestrator's JSONL output to audit theoretical predictions
from all 5 sleeves. Safe to run during code freeze — reads only, no mutations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def analyze_ledger(date_str: str):
    """Parse and display shadow ledger for the given date."""
    ledger_path = Path(
        f"backend/artifacts/orchestrator/shadow_ledger/shadow_{date_str}.jsonl"
    )

    if not ledger_path.exists():
        print(f"\n[!] Ledger for {date_str} NOT FOUND at {ledger_path}")
        print("    Possible causes:")
        print("    1. Orchestrator DAG did not run today")
        print("    2. Shadow ledger path differs from expected")
        print("    3. Sleeves ran but JSONL writer is not wired")

        # Search for any JSONL files in artifacts
        arts = Path("backend/artifacts")
        if arts.exists():
            jsonl_files = list(arts.rglob("*.jsonl"))
            if jsonl_files:
                print(f"\n    Found {len(jsonl_files)} JSONL files in artifacts:")
                for f in jsonl_files[:10]:
                    print(f"      {f} ({f.stat().st_size / 1024:.1f} KB)")
            else:
                print("\n    No JSONL files found anywhere in backend/artifacts/")
        return

    records = []
    with open(ledger_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("Ledger is empty.")
        return

    df = pd.DataFrame(records)
    print(
        f"\n{'=' * 60}\n"
        f"Shadow Ledger Audit: {date_str} | Total Cycles: {len(df)}\n"
        f"{'=' * 60}"
    )

    latest = df.iloc[-1]

    print("\n[🛡️ TD3 Risk Agent]")
    print(f"  Veto Flag:       {latest.get('rl_veto', 'N/A')}")
    print(f"  Size Multiplier: {latest.get('rl_multiplier', 'N/A')}")

    print("\n[🕒 Chronos2 Overnight Gap]")
    print(f"  P10: {latest.get('chronos2_p10', 'N/A')}")
    print(f"  P90: {latest.get('chronos2_p90', 'N/A')}")

    print("\n[📈 SMoE Selector (Hysteresis Active)]")
    longs = latest.get("smoe_longs", [])
    shorts = latest.get("smoe_shorts", [])
    print(f"  Top 3 Longs:  {longs[:3] if isinstance(longs, list) else 'N/A'}")
    print(f"  Top 3 Shorts: {shorts[:3] if isinstance(shorts, list) else 'N/A'}")

    print("\n[⚖️ StatArb V3 (Beta-Neutral)]")
    statarb = latest.get("statarb_weights", {})
    if isinstance(statarb, dict):
        active = {k: v for k, v in statarb.items() if abs(v) > 0.01}
        print(f"  Active Pairs ({len(active)}/10): {list(active.keys())[:5]}...")

    print("\n[📉 VRP Options]")
    print(f"  IV/RV Forecast: {latest.get('vrp_forecast', 'N/A')}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "20260304"
    analyze_ledger(date)
