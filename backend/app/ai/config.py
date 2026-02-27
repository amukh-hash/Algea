"""Central feature-flag configuration for AI-native modules.

All flags default to ``False`` (Shadow Mode) so new modules are inert
until explicitly promoted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AIFeatureFlags:
    """Master toggle for every AI-native subsystem."""

    # ── Regime Detection ──────────────────────────────────────────────
    use_wasserstein_clustering: bool = field(
        default_factory=lambda: os.getenv("AI_USE_WASSERSTEIN", "0") == "1"
    )

    # ── Equity Sleeve ─────────────────────────────────────────────────
    use_mera_equity: bool = field(
        default_factory=lambda: os.getenv("AI_USE_MERA_EQUITY", "0") == "1"
    )

    # ── VRP Options Sleeve ────────────────────────────────────────────
    use_vrp_lstm_cnn: bool = field(
        default_factory=lambda: os.getenv("AI_USE_VRP_LSTM_CNN", "0") == "1"
    )

    # ── Futures Sleeve ────────────────────────────────────────────────
    use_kronos_futures: bool = field(
        default_factory=lambda: os.getenv("AI_USE_KRONOS_FUTURES", "0") == "1"
    )

    # ── Meta-Allocator ────────────────────────────────────────────────
    use_rl_allocator: bool = field(
        default_factory=lambda: os.getenv("AI_USE_RL_ALLOCATOR", "0") == "1"
    )

    # ── Execution ─────────────────────────────────────────────────────
    use_signature_execution: bool = field(
        default_factory=lambda: os.getenv("AI_USE_SIGNATURE_EXEC", "0") == "1"
    )

    # ── Hyperparameters (shared) ──────────────────────────────────────
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_buffer_size: int = 100_000
