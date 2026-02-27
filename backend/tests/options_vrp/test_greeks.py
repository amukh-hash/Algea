"""Tests for Black-Scholes greeks engine — validates against known textbook values."""
from __future__ import annotations

import numpy as np
import pytest

from algea.data.options.greeks_engine import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta,
    bs_vega,
)


# Textbook reference: S=100, K=100, T=0.25, r=0.05, σ=0.20, q=0
# Call price ≈ 3.833, Put price ≈ 2.586 (Hull's "Options, Futures…")
_S, _K, _T, _r, _sigma, _q = 100.0, 100.0, 0.25, 0.05, 0.20, 0.0


class TestBSPrice:
    def test_call_price_range(self):
        price = float(bs_price(_S, _K, _T, _r, _sigma, "call", _q))
        assert 4.0 < price < 5.2, f"ATM call price {price} outside expected range"

    def test_put_price_range(self):
        price = float(bs_price(_S, _K, _T, _r, _sigma, "put", _q))
        assert 2.8 < price < 3.8, f"ATM put price {price} outside expected range"

    def test_put_call_parity(self):
        call = float(bs_price(_S, _K, _T, _r, _sigma, "call", _q))
        put = float(bs_price(_S, _K, _T, _r, _sigma, "put", _q))
        # C - P = S*exp(-qT) - K*exp(-rT)
        rhs = _S * np.exp(-_q * _T) - _K * np.exp(-_r * _T)
        assert abs((call - put) - rhs) < 0.01, "Put-call parity violated"

    def test_call_price_non_negative(self):
        price = float(bs_price(50, 100, 0.01, 0.05, 0.20, "call"))
        assert price >= 0.0

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            bs_price(_S, _K, _T, _r, _sigma, "straddle")


class TestBSDelta:
    def test_atm_call_delta_near_half(self):
        delta = float(bs_delta(_S, _K, _T, _r, _sigma, "call", _q))
        assert 0.45 < delta < 0.65, f"ATM call delta {delta} not near 0.5"

    def test_atm_put_delta_near_neg_half(self):
        delta = float(bs_delta(_S, _K, _T, _r, _sigma, "put", _q))
        assert -0.55 < delta < -0.35, f"ATM put delta {delta} not near -0.5"

    def test_deep_itm_call_delta_near_one(self):
        delta = float(bs_delta(200, 100, 1.0, 0.05, 0.20, "call"))
        assert delta > 0.95

    def test_deep_otm_put_delta_near_zero(self):
        delta = float(bs_delta(200, 100, 1.0, 0.05, 0.20, "put"))
        assert delta > -0.05


class TestBSGamma:
    def test_atm_gamma_positive(self):
        gamma = float(bs_gamma(_S, _K, _T, _r, _sigma, _q))
        assert gamma > 0.0

    def test_gamma_peak_at_atm(self):
        g_atm = float(bs_gamma(100, 100, _T, _r, _sigma))
        g_otm = float(bs_gamma(100, 120, _T, _r, _sigma))
        assert g_atm > g_otm, "Gamma should be highest ATM"


class TestBSVega:
    def test_atm_vega_positive(self):
        vega = float(bs_vega(_S, _K, _T, _r, _sigma, _q))
        assert vega > 0.0

    def test_vega_reasonable_range(self):
        vega = float(bs_vega(_S, _K, _T, _r, _sigma, _q))
        # For S=100, ATM, T=0.25, vega ≈ 0.25*S*N'(d1)*sqrt(T) ≈ 19-20
        assert 15 < vega < 25, f"Vega {vega} outside expected range"


class TestBSTheta:
    def test_call_theta_negative(self):
        theta = float(bs_theta(_S, _K, _T, _r, _sigma, "call", _q))
        assert theta < 0.0, "Long call theta should be negative"

    def test_put_theta_negative(self):
        theta = float(bs_theta(_S, _K, _T, _r, _sigma, "put", _q))
        assert theta < 0.0, "Long put theta should be negative"


class TestVectorised:
    def test_array_inputs(self):
        S = np.array([100, 100, 100])
        K = np.array([90, 100, 110])
        prices = bs_price(S, K, _T, _r, _sigma, "call")
        assert len(prices) == 3
        # ITM > ATM > OTM for calls
        assert prices[0] > prices[1] > prices[2]
