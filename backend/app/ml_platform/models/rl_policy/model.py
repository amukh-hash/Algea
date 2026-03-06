"""
RL Policy Model — Production TD3 Execution Agent wrapper with state projection.

Replaces the ``torch.zeros(1, 256)`` dummy tensor fallback with a learnable
``RLStateProjector`` network that maps arbitrary live feature vectors into
the strict ``[1, 256]`` state space expected by the ``TD3Actor``.

Falls back to a pass-through ``(size_multiplier=1.0, veto=False)`` if no
trained weights are loaded, instead of generating random actions.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from algae.models.rl.td3 import TD3Actor

logger = logging.getLogger(__name__)


class RLStateProjector(nn.Module):
    """Projects raw, variable-length feature vectors into the TD3 state space.

    Parameters
    ----------
    raw_dim : int
        Dimensionality of the live feature vector from the orchestrator.
    embed_dim : int
        Output dimension matching ``TD3Actor.state_dim`` (default 256).
    """

    def __init__(self, raw_dim: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLPolicyModel:
    """TD3 Execution Agent Wrapper with state projection.

    If trained weights are loaded, maps live features through
    ``RLStateProjector → TD3Actor`` to get continuous actions.
    If no weights are available, returns safe pass-through defaults.

    Parameters
    ----------
    hidden_size : int
        Unused in production; retained for backward compatibility.
    raw_feature_dim : int
        Expected imensionality of live feature vectors.  If ``None``,
        the legacy fallback path is used.
    weights_path : str or None
        Path to a ``.pt`` file containing ``{"actor": ..., "projector": ...}``.
    device : str
        CUDA device string.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        raw_feature_dim: Optional[int] = None,
        weights_path: Optional[str] = None,
        device: str = "cuda:1",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = 256
        self.action_dim = 2
        self.is_active = False

        self.actor = TD3Actor(state_dim=self.state_dim, action_dim=self.action_dim)
        self.actor.to(self.device, dtype=torch.bfloat16)
        self.actor.eval()

        self.projector: Optional[RLStateProjector] = None
        if raw_feature_dim is not None and raw_feature_dim > 0:
            self.projector = RLStateProjector(raw_dim=raw_feature_dim, embed_dim=self.state_dim)
            self.projector.to(self.device, dtype=torch.bfloat16)
            self.projector.eval()

        if weights_path:
            self._load_weights(weights_path)
        else:
            logger.warning(
                "RLPolicyModel: no trained weights.  Returning safe pass-through "
                "(size_multiplier=1.0, veto=False) instead of random TD3 actions."
            )

    def _load_weights(self, path: str) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            if "actor" in state_dict:
                self.actor.load_state_dict(state_dict["actor"], strict=True)
            else:
                self.actor.load_state_dict(state_dict, strict=False)
            if self.projector is not None and "projector" in state_dict:
                self.projector.load_state_dict(state_dict["projector"], strict=True)
            self.is_active = True
            logger.info("RL Policy weights loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load RL weights from %s: %s", path, e)
            self.is_active = False

    def act(self, state_embedding: torch.Tensor | list[float]) -> tuple[float, bool, float]:
        """Compute actions from state.

        Parameters
        ----------
        state_embedding : Tensor or list[float]
            If a list, projects through ``RLStateProjector`` first.
            If a Tensor of shape ``[1, 256]``, feeds directly to TD3Actor.

        Returns
        -------
        tuple[float, bool, float]
            ``(size_multiplier, veto, confidence)``
        """
        # ── Fail-safe: untrained → pass-through ──
        if not self.is_active:
            return 1.0, False, 0.0

        # ── Project raw features → state embedding ──
        if isinstance(state_embedding, list):
            raw = state_embedding
            # NaN / Inf guard
            arr = np.array(raw, dtype=np.float32)
            if np.isnan(arr).any() or np.isinf(arr).any():
                logger.warning("RL Policy: NaN/Inf in input features.  Vetoing.")
                return 0.0, True, 0.0

            if self.projector is not None:
                raw_tensor = torch.tensor(
                    [raw], dtype=torch.bfloat16, device=self.device
                )
                with torch.inference_mode():
                    state_tensor = self.projector(raw_tensor)
            else:
                # Legacy fallback: pad/truncate to state_dim
                padded = raw[:self.state_dim] + [0.0] * max(0, self.state_dim - len(raw))
                state_tensor = torch.tensor(
                    [padded], dtype=torch.bfloat16, device=self.device
                )
        else:
            state_tensor = state_embedding
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

        if state_tensor is None or state_tensor.numel() == 0:
            return 1.0, False, 0.0

        with torch.inference_mode():
            action = self.actor(state_tensor.to(self.device, dtype=torch.bfloat16))

        act_val = action[0].cpu().float().numpy()

        # Linear map Tanh [-1, 1] → [0, 1] for size multiplier
        multiplier = max(0.01, min((float(act_val[0]) + 1.0) / 2.0, 1.0))
        # Second dimension: positive → veto, magnitude → confidence
        veto = float(act_val[1]) > 0.0
        confidence = min(1.0, max(0.0, float(abs(act_val[1]))))

        return multiplier, veto, confidence
