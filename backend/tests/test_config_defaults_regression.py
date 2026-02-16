"""Tests for config from_dict() preserving defaults — F6 regression."""
import pytest


class TestTradeProxyConfigFromDict:
    def test_empty_dict_returns_default(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyConfig
        cfg = TradeProxyConfig.from_dict({})
        default = TradeProxyConfig()
        assert cfg == default

    def test_missing_score_semantics_preserves_default(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyConfig
        cfg = TradeProxyConfig.from_dict({"cost_per_contract": 0.0})
        assert cfg.score_semantics == "alpha_low_long"
        assert cfg.cost_per_contract == 0.0

    def test_explicit_semantics_overrides(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyConfig
        cfg = TradeProxyConfig.from_dict({"score_semantics": "alpha_high_long"})
        assert cfg.score_semantics == "alpha_high_long"

    def test_unknown_keys_ignored(self):
        from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyConfig
        cfg = TradeProxyConfig.from_dict({"bogus_key": 42, "top_k": 3})
        assert cfg.top_k == 3


class TestLossConfig:
    def test_from_dict_defaults(self):
        from sleeves.cooc_reversal_futures.config import LossConfig
        cfg = LossConfig.from_dict({})
        assert cfg.rank_loss_weight == 1.0
        assert cfg.sigma_floor == 1e-4

    def test_from_dict_override(self):
        from sleeves.cooc_reversal_futures.config import LossConfig
        cfg = LossConfig.from_dict({"rank_loss_weight": 0.5, "temp_y": 2.0})
        assert cfg.rank_loss_weight == 0.5
        assert cfg.temp_y == 2.0
        assert cfg.reg_loss_weight == 0.1  # default preserved


class TestTrainingConfig:
    def test_from_dict_defaults(self):
        from sleeves.cooc_reversal_futures.config import TrainingConfig
        cfg = TrainingConfig.from_dict({})
        assert len(cfg.seeds) == 7

    def test_list_seeds_converted_to_tuple(self):
        from sleeves.cooc_reversal_futures.config import TrainingConfig
        cfg = TrainingConfig.from_dict({"seeds": [1, 2, 3]})
        assert isinstance(cfg.seeds, tuple)
        assert cfg.seeds == (1, 2, 3)


class TestPromotionConfig:
    def test_from_dict_new_fields(self):
        from sleeves.cooc_reversal_futures.config import PromotionConfig
        cfg = PromotionConfig.from_dict({})
        assert cfg.ic_tail_floor == -0.05
        assert cfg.catastrophic_sharpe_floor == -0.50
        assert cfg.tier_for_gates == "TIER2"

    def test_from_dict_override(self):
        from sleeves.cooc_reversal_futures.config import PromotionConfig
        cfg = PromotionConfig.from_dict({
            "min_sharpe_delta": 0.20,
            "catastrophic_sharpe_floor": -1.0,
        })
        assert cfg.min_sharpe_delta == 0.20
        assert cfg.catastrophic_sharpe_floor == -1.0
        assert cfg.min_hit_rate == 0.45  # default preserved


class TestCOOCReversalConfig:
    def test_from_dict_nested(self):
        from sleeves.cooc_reversal_futures.config import COOCReversalConfig
        cfg = COOCReversalConfig.from_dict({
            "model": {"lr": 1e-3},
            "loss": {"rank_loss_weight": 0.5},
        })
        assert cfg.model.lr == 1e-3
        assert cfg.loss.rank_loss_weight == 0.5
        assert cfg.model.d_model == 128  # default preserved

    def test_unknown_top_level_key_ignored(self):
        from sleeves.cooc_reversal_futures.config import COOCReversalConfig
        cfg = COOCReversalConfig.from_dict({"totally_bogus": 42})
        assert cfg.gross_target == 0.8  # default
