from __future__ import annotations

from datetime import datetime, timezone


class OptionChainFetcher:
    def fetch(self, underlying_symbol: str, expiries: list[str]) -> list[dict]:
        asof = datetime.now(timezone.utc).isoformat()
        rows: list[dict] = []
        for exp in expiries:
            for strike_mult in [0.95, 1.0, 1.05]:
                spot = 100.0
                strike = spot * strike_mult
                iv = 0.2 + (strike_mult - 1.0) * 0.1
                for opt_type, delta_sign in [("C", 1), ("P", -1)]:
                    rows.append(
                        {
                            "asof": asof,
                            "underlying_symbol": underlying_symbol,
                            "expiry": exp,
                            "dte": max(1, int(exp.split("-")[-1]) % 30),
                            "option_type": opt_type,
                            "strike": strike,
                            "bid": 1.0,
                            "ask": 1.2,
                            "mid": 1.1,
                            "implied_vol": iv,
                            "delta": 0.25 * delta_sign,
                            "gamma": 0.02,
                            "vega": 0.1,
                            "theta": -0.01,
                            "open_interest": 100,
                            "volume": 10,
                            "spot": spot,
                            "rate": 0.02,
                            "div_yield": 0.01,
                        }
                    )
        return rows
