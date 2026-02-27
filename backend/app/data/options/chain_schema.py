from __future__ import annotations

REQUIRED_COLUMNS = {
    "asof",
    "underlying_symbol",
    "expiry",
    "dte",
    "option_type",
    "strike",
    "bid",
    "ask",
    "mid",
    "implied_vol",
    "delta",
    "gamma",
    "vega",
    "theta",
    "spot",
}


def validate_chain_rows(rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        missing = REQUIRED_COLUMNS - set(row.keys())
        if missing:
            raise ValueError(f"row {i} missing required columns: {sorted(missing)}")
