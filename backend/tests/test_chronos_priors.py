"""Unit tests for ChronosPriors validation and infer_priors contract.

These tests are lightweight (no GPU, no network, no model loading).
They exercise the dataclass validation logic and schema invariants.
"""
import math
import pytest

from algae.models.foundation.chronos2_teacher import ChronosPriors


# ═══════════════════════════════════════════════════════════════════════════
# ChronosPriors.validate() tests
# ═══════════════════════════════════════════════════════════════════════════

def _make_priors(**overrides):
    """Build a valid ChronosPriors with sane defaults, overriding as needed."""
    defaults = dict(
        drift=0.05, vol_forecast=0.10, tail_risk=-0.02, trend_conf=0.7,
        metadata={"horizon": 10, "mode": "native_nll"},
        q10=-0.02, q50=0.05, q90=0.12, dispersion=0.14, prob_up=0.7,
    )
    defaults.update(overrides)
    return ChronosPriors(**defaults)


class TestPriorsValidateHappyPath:
    """Valid priors should pass validation unchanged."""

    def test_valid_priors_pass(self):
        p = _make_priors()
        result = p.validate(strict=True)
        assert result is p
        assert p.q10 == -0.02
        assert p.q50 == 0.05
        assert p.q90 == 0.12

    def test_zero_dispersion_valid(self):
        p = _make_priors(q10=0.0, q50=0.0, q90=0.0, dispersion=0.0, drift=0.0, tail_risk=0.0)
        p.validate(strict=True)

    def test_prob_up_boundaries(self):
        for pu in (0.0, 0.5, 1.0):
            p = _make_priors(prob_up=pu, trend_conf=pu)
            p.validate(strict=True)


class TestPriorsMonotonicEnforcement:
    """Quantile monotonicity (q10 <= q50 <= q90) enforcement."""

    def test_strict_raises_on_violation(self):
        p = _make_priors(q10=0.10, q50=0.05, q90=0.01)
        with pytest.raises(ValueError, match="monotonicity"):
            p.validate(strict=True)

    def test_lenient_auto_sorts(self):
        p = _make_priors(q10=0.10, q50=0.05, q90=0.01)
        p.validate(strict=False)
        assert p.q10 <= p.q50 <= p.q90
        assert p.q10 == 0.01
        assert p.q50 == 0.05
        assert p.q90 == 0.10


class TestPriorsProbClamp:
    """prob_up must be in [0, 1]."""

    def test_strict_raises_on_out_of_bounds(self):
        p = _make_priors(prob_up=1.5)
        with pytest.raises(ValueError, match="prob_up"):
            p.validate(strict=True)

    def test_lenient_clamps(self):
        p = _make_priors(prob_up=-0.2)
        p.validate(strict=False)
        assert p.prob_up == 0.0

        p2 = _make_priors(prob_up=1.5)
        p2.validate(strict=False)
        assert p2.prob_up == 1.0


class TestPriorsNanRejection:
    """NaN/Inf always raises regardless of strict mode."""

    def test_nan_drift(self):
        p = _make_priors(drift=float("nan"))
        with pytest.raises(ValueError, match="not finite"):
            p.validate(strict=False)

    def test_inf_vol(self):
        p = _make_priors(vol_forecast=float("inf"))
        with pytest.raises(ValueError, match="not finite"):
            p.validate(strict=False)

    def test_neg_inf_q10(self):
        p = _make_priors(q10=float("-inf"))
        with pytest.raises(ValueError, match="not finite"):
            p.validate(strict=True)


class TestPriorsNegativeDispersion:
    """Negative dispersion: strict raises, lenient recomputes."""

    def test_strict_raises(self):
        p = _make_priors(dispersion=-0.5)
        with pytest.raises(ValueError, match="dispersion"):
            p.validate(strict=True)

    def test_lenient_recomputes(self):
        p = _make_priors(q10=-0.02, q90=0.12, dispersion=-0.5)
        p.validate(strict=False)
        assert p.dispersion == pytest.approx(0.14)


class TestPriorsSchemaKeys:
    """All 10 fields (5 core + 5 extended) must be present."""

    def test_all_keys_present(self):
        p = _make_priors()
        required = {"drift", "vol_forecast", "tail_risk", "trend_conf", "metadata",
                     "q10", "q50", "q90", "dispersion", "prob_up"}
        actual = set(p.__dataclass_fields__.keys())
        assert required <= actual, f"Missing: {required - actual}"
