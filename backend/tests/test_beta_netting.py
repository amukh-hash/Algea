"""Tests for multiplier-aware cross-sleeve beta netting.

Verifies:
  - SPY-equivalent notional is correctly computed (qty × multiplier × price × beta)
  - GC, 6E, and other zero-beta instruments are excluded from equity netting
  - Integer truncation prevents fractional quantities
  - Directional scaling only affects breach-side intents
  - Scale-down preserves uncorrelated instruments
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.app.allocator.sleeve_allocator import enforce_portfolio_beta_limits
from backend.app.allocator.beta_map import get_spy_beta


@dataclass
class MockIntent:
    symbol: str
    qty: int
    multiplier: float
    direction: int  # +1 long, -1 short
    asset_class: str = "FUTURE"


class TestBetaMap:
    def test_es_beta_is_one(self):
        assert get_spy_beta("ES") == 1.0

    def test_nq_beta_is_higher(self):
        assert get_spy_beta("NQ") == 1.2

    def test_gold_zero_beta(self):
        assert get_spy_beta("GC") == 0.0

    def test_fx_zero_beta(self):
        assert get_spy_beta("6E") == 0.0
        assert get_spy_beta("6J") == 0.0

    def test_crude_zero_beta(self):
        assert get_spy_beta("CL") == 0.0

    def test_bonds_zero_beta(self):
        assert get_spy_beta("ZN") == 0.0
        assert get_spy_beta("TLT") == 0.0

    def test_unknown_symbol_defaults_to_one(self):
        assert get_spy_beta("UNKNOWN_TICKER") == 1.0

    def test_gold_miners_have_partial_beta(self):
        assert 0 < get_spy_beta("GDXJ") < 1.0


class TestEnforcePortfolioBetaLimits:
    """Core beta netting correctness tests."""

    def test_within_limits_no_change(self):
        """Intents within beta cap should not be scaled."""
        intents = [MockIntent("ES", 1, 50.0, 1)]
        prices = {"ES": 5000.0}
        nav = 500_000.0  # 1 ES = $250K notional = 50% NAV beta, well under 120%
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        assert result[0].qty == 1

    def test_breach_scales_down_long(self):
        """Excessive long beta exposure should be scaled down."""
        intents = [
            MockIntent("ES", 5, 50.0, 1),  # 5 × 50 × 5000 × 1.0 = $1,250,000
        ]
        prices = {"ES": 5000.0}
        nav = 500_000.0  # max exposure = 120% × 500K = $600K, intent = $1.25M
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        assert result[0].qty < 5
        assert result[0].qty >= 0

    def test_gold_excluded_from_netting(self):
        """GC (zero beta) should not be scaled, even alongside breach."""
        intents = [
            MockIntent("ES", 5, 50.0, 1),   # $1.25M beta exposure
            MockIntent("GC", 3, 100.0, 1),   # beta=0.0, should be untouched
        ]
        prices = {"ES": 5000.0, "GC": 2000.0}
        nav = 500_000.0
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        assert result[1].qty == 3, "GC should not be affected by equity beta cap"

    def test_fx_excluded_from_netting(self):
        """6E (zero beta) should not be scaled."""
        intents = [
            MockIntent("ES", 5, 50.0, 1),
            MockIntent("6E", 2, 125000.0, 1),
        ]
        prices = {"ES": 5000.0, "6E": 1.10}
        nav = 500_000.0
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        assert result[1].qty == 2, "6E should not be affected by equity beta cap"

    def test_integer_truncation(self):
        """Scaled quantities must be integers (math.floor)."""
        intents = [MockIntent("ES", 10, 50.0, 1)]  # $2.5M
        prices = {"ES": 5000.0}
        nav = 500_000.0  # max = $600K, need to scale 10 → ~2.4 → floor to 2
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        assert isinstance(result[0].qty, int)
        assert result[0].qty == int(result[0].qty)  # No fractional quantities

    def test_directional_scaling_preserves_shorts(self):
        """When long-side breaches, short intents should NOT be scaled."""
        intents = [
            MockIntent("ES", 5, 50.0, 1),    # +$1.25M long beta
            MockIntent("SPY", 100, 1.0, -1),  # -$50K short beta
        ]
        prices = {"ES": 5000.0, "SPY": 500.0}
        nav = 500_000.0
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        # ES (long) should be scaled down
        assert result[0].qty < 5
        # SPY (short) should be untouched
        assert result[1].qty == 100

    def test_empty_intents_returns_empty(self):
        result = enforce_portfolio_beta_limits([], {}, 100_000.0)
        assert result == []

    def test_zero_nav_returns_unchanged(self):
        intents = [MockIntent("ES", 5, 50.0, 1)]
        result = enforce_portfolio_beta_limits(intents, {"ES": 5000.0}, 0.0)
        assert result[0].qty == 5

    def test_multiplier_correctly_applied(self):
        """Test that ES multiplier (50) creates 50× more exposure than SPY (1)."""
        # 1 ES at 5000 = $250K. 250 shares SPY at 500 = $125K.
        # ES should breach faster.
        intents_es = [MockIntent("ES", 3, 50.0, 1)]
        intents_spy = [MockIntent("SPY", 250, 1.0, 1)]
        prices = {"ES": 5000.0, "SPY": 500.0}
        nav = 500_000.0

        result_es = enforce_portfolio_beta_limits(intents_es, prices, nav, max_beta=1.2)
        result_spy = enforce_portfolio_beta_limits(intents_spy, prices, nav, max_beta=1.2)

        # 3 ES = $750K > $600K limit → scaled
        assert result_es[0].qty < 3
        # 250 SPY = $125K < $600K limit → not scaled
        assert result_spy[0].qty == 250

    def test_cross_sleeve_mixed_beta(self):
        """Realistic cross-sleeve scenario: COOC + StatArb + VRP."""
        intents = [
            MockIntent("ES", 2, 50.0, 1),     # COOC: $500K × 1.0 = $500K beta
            MockIntent("NQ", 1, 20.0, 1),      # COOC: $380K × 1.2 = $456K beta
            MockIntent("GC", 1, 100.0, -1),    # COOC: $200K × 0.0 = $0 beta
            MockIntent("XLF", 500, 1.0, 1),    # StatArb: $25K × 1.0 = $25K beta
            MockIntent("XLK", -300, 1.0, -1),  # StatArb: $63K × 1.1 = $69K beta (short)
        ]
        prices = {"ES": 5000.0, "NQ": 19000.0, "GC": 2000.0, "XLF": 50.0, "XLK": 210.0}
        nav = 500_000.0  # max = $600K

        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)

        # GC should be untouched (zero beta) — find by symbol since filtering may shift indices
        gc_intents = [i for i in result if i.symbol == "GC"]
        assert len(gc_intents) == 1
        assert gc_intents[0].qty == 1

    def test_zero_quantity_pennying_trap(self):
        """1-lot futures scaled by 0.526 → math.floor(0.526) = 0.

        qty=0 is a valid FLATTEN command that the execution engine must
        receive.  The IB Error 104 guard is applied downstream after
        computing differential orders (target_qty - current_qty).
        """
        intents = [MockIntent("NQ", 1, 20.0, 1)]  # 1 × 20 × 19000 × 1.2 = $456K
        prices = {"NQ": 19000.0}
        nav = 200_000.0  # max = 120% × 200K = $240K. Scale factor = 240K/456K ≈ 0.526
        # math.floor(1 * 0.526) = 0 → intent survives as qty=0 (flatten)
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        assert len(result) == 1, "qty=0 intents must survive (they are valid FLATTEN commands)"
        assert result[0].qty == 0

    def test_double_negative_exposure_trap(self):
        """Signed qty (-5) × signed direction (-1) must NOT invert risk profile.

        Without abs(qty), a short position with both negative qty and negative
        direction would appear as positive beta exposure, causing the netting
        engine to misclassify it and scale down the wrong side.
        """
        # Short 5 ES with signed qty AND signed direction
        intents = [
            MockIntent("ES", -5, 50.0, -1),   # Short: should be NEGATIVE beta
            MockIntent("NQ", 2, 20.0, 1),      # Long: should be POSITIVE beta
        ]
        prices = {"ES": 5000.0, "NQ": 19000.0}
        nav = 1_000_000.0  # Large NAV to avoid scaling

        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=5.0)

        # Both should survive (no breach at 5.0× cap with $1M NAV)
        es_intents = [i for i in result if i.symbol == "ES"]
        nq_intents = [i for i in result if i.symbol == "NQ"]
        assert len(es_intents) == 1
        assert len(nq_intents) == 1
        # ES short qty should be preserved as-is (negative)
        assert es_intents[0].qty == -5

    def test_zero_qty_preserves_all_intents(self):
        """When one intent is zeroed, all intents must survive (qty=0 is valid flatten)."""
        intents = [
            MockIntent("NQ", 1, 20.0, 1),     # Will be zeroed (pennying)
            MockIntent("GC", 5, 100.0, 1),     # Zero beta — untouched
        ]
        prices = {"NQ": 19000.0, "GC": 2000.0}
        nav = 200_000.0
        result = enforce_portfolio_beta_limits(intents, prices, nav, max_beta=1.2)
        # Both should survive: NQ with qty=0 (flatten), GC untouched
        assert len(result) == 2
        nq_intent = [i for i in result if i.symbol == "NQ"][0]
        gc_intent = [i for i in result if i.symbol == "GC"][0]
        assert nq_intent.qty == 0
        assert gc_intent.qty == 5

