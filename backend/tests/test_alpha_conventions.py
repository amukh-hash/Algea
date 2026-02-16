"""Tests for canonical alpha conventions — polarity regression tests."""
import numpy as np
import pandas as pd
import pytest


def _build_panel(n_days=50, n_inst=6, seed=42):
    rng = np.random.RandomState(seed)
    base = pd.Timestamp("2024-01-02")
    rows = []
    for d in range(n_days):
        td = (base + pd.offsets.BDay(d)).date()
        r_co = rng.randn(n_inst) * 0.01
        r_oc = -0.7 * r_co + rng.randn(n_inst) * 0.003
        for i in range(n_inst):
            rows.append({
                "trading_day": td, "root": f"R{i}", "instrument": f"R{i}",
                "r_co": r_co[i], "r_oc": r_oc[i],
            })
    panel = pd.DataFrame(rows)
    panel["y"] = -panel["r_oc"]
    return panel


class TestLabelY:
    def test_label_is_negated_roc(self):
        from backend.app.portfolio.alpha_conventions import label_y
        panel = _build_panel()
        y = label_y(panel)
        np.testing.assert_array_almost_equal(y.values, -panel["r_oc"].values)

    def test_alpha_target_inverts_y(self):
        from backend.app.portfolio.alpha_conventions import alpha_target_from_y
        y = np.array([1.0, -2.0, 0.5])
        at = alpha_target_from_y(y)
        np.testing.assert_array_equal(at, -y)


class TestSigma:
    def test_sigma_positive(self):
        from backend.app.portfolio.alpha_conventions import sigma_from_log_sigma
        raw = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        sigma = sigma_from_log_sigma(raw, sigma_floor=1e-4)
        assert np.all(sigma > 0)
        assert np.all(sigma >= 1e-4)

    def test_sigma_torch(self):
        pytest.importorskip("torch")
        import torch
        from backend.app.portfolio.alpha_conventions import sigma_from_log_sigma
        raw = torch.tensor([-5.0, 0.0, 5.0])
        sigma = sigma_from_log_sigma(raw, sigma_floor=1e-4)
        assert isinstance(sigma, torch.Tensor)
        assert torch.all(sigma > 0)


class TestDerivedAndAlpha:
    def test_derived_score(self):
        from backend.app.portfolio.alpha_conventions import derived_score
        score_raw = np.array([1.0, -1.0, 0.5])
        sigma = np.array([0.5, 0.5, 0.5])
        d = derived_score(score_raw, sigma, eps=1e-6)
        np.testing.assert_allclose(d, score_raw / (1e-6 + sigma), atol=1e-8)

    def test_derived_to_alpha_negates(self):
        from backend.app.portfolio.alpha_conventions import derived_to_alpha
        d = np.array([2.0, -3.0, 0.0])
        alpha = derived_to_alpha(d)
        np.testing.assert_array_equal(alpha, -d)


class TestBaseline:
    def test_meanrevert_baseline(self):
        from backend.app.portfolio.alpha_conventions import baseline_alpha_from_r_co
        r_co = pd.Series([0.01, -0.02, 0.005])
        alpha = baseline_alpha_from_r_co(r_co, mode="meanrevert")
        np.testing.assert_array_almost_equal(alpha.values, -r_co.values)

    def test_momentum_baseline(self):
        from backend.app.portfolio.alpha_conventions import baseline_alpha_from_r_co
        r_co = np.array([0.01, -0.02])
        alpha = baseline_alpha_from_r_co(r_co, mode="momentum")
        np.testing.assert_array_equal(alpha, r_co)


class TestScoreToAlpha:
    def test_alpha_low_long_negates(self):
        from backend.app.portfolio.alpha_conventions import score_to_alpha
        s = np.array([1.0, -1.0, 0.5])
        a = score_to_alpha(s, semantics="alpha_low_long")
        np.testing.assert_array_equal(a, -s)

    def test_alpha_high_long_identity(self):
        from backend.app.portfolio.alpha_conventions import score_to_alpha
        s = np.array([1.0, -1.0, 0.5])
        a = score_to_alpha(s, semantics="alpha_high_long")
        np.testing.assert_array_equal(a, s)


class TestOracleSanity:
    """Oracle pred=y must produce positive proxy Sharpe; anti-oracle negative."""

    def test_oracle_positive_sharpe(self):
        from backend.app.portfolio.alpha_conventions import label_y, score_to_alpha
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel()
        preds = panel["y"].values  # oracle
        report = evaluate_trade_proxy(
            dataset=panel, preds=preds,
            config={"cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                    "slippage_bps_close": 0.0},
        )
        assert report.sharpe_model > 0, f"Oracle Sharpe should be positive, got {report.sharpe_model}"

    def test_anti_oracle_negative_sharpe(self):
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel()
        preds = -panel["y"].values  # anti-oracle
        report = evaluate_trade_proxy(
            dataset=panel, preds=preds,
            config={"cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                    "slippage_bps_close": 0.0},
        )
        assert report.sharpe_model < 0, f"Anti-oracle Sharpe should be negative, got {report.sharpe_model}"

    def test_oracle_anti_oracle_symmetric(self):
        from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

        panel = _build_panel()
        oracle = evaluate_trade_proxy(
            dataset=panel, preds=panel["y"].values,
            config={"cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                    "slippage_bps_close": 0.0},
        )
        anti = evaluate_trade_proxy(
            dataset=panel, preds=-panel["y"].values,
            config={"cost_per_contract": 0.0, "slippage_bps_open": 0.0,
                    "slippage_bps_close": 0.0},
        )
        assert abs(oracle.sharpe_model + anti.sharpe_model) < 0.5, (
            f"Not symmetric: oracle={oracle.sharpe_model:.3f}, anti={anti.sharpe_model:.3f}"
        )


class TestDictConfigDefaults:
    """Config defaults preserved when dict keys missing."""

    def test_missing_semantics_preserves_default(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyConfig
        cfg = TradeProxyConfig()
        assert cfg.score_semantics == "alpha_low_long"
