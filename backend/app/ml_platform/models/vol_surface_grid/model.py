"""
Vol Surface Grid Forecaster — Production inference.

Loads the trained SpatialTemporalTransformer + regression head from
``vrp_st_transformer.pt`` and provides state embeddings + VRP forecasts
for the Orchestrator DAG.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from algae.models.tsfm.st_transformer import SpatialTemporalTransformer

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = Path("backend/artifacts/model_weights/vrp_st_transformer.pt")


class VolSurfaceGridForecaster:
    """Spatial-Temporal Transformer for Options Grid Sequences.

    Processes ``[Lookback=10, 5 Tenors, 25 Deltas]`` vol surface snapshots
    and outputs:
      1. A 256-dim state embedding (for TD3 Agent S_t)
      2. A scalar VRP forecast (for Put Credit Spread sizing)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda:1",
        scale: float = 0.05,
    ):
        self.scale = scale
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

        # Build architecture
        self.model = SpatialTemporalTransformer(
            spatial_dim=125,   # 5 tenors × 25 deltas flattened
            temporal_dim=10,   # 10-day lookback
            d_model=128,
            nhead=4,
            num_layers=3,
        ).to(self.device)

        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.0),  # No dropout at inference
            nn.Linear(64, 1),
        ).to(self.device)

        # Load weights
        wp = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if wp.exists():
            self._load_weights(wp)
        else:
            logger.warning(
                "VRP model weights not found at %s. "
                "Forecaster will return zero embeddings.", wp,
            )

    def _load_weights(self, path: Path) -> None:
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["transformer"])
            self.regression_head.load_state_dict(state["head"])
            self.model.eval()
            self.regression_head.eval()
            self.is_loaded = True
            logger.info("VRP ST-Transformer loaded from %s", path.name)
        except Exception as e:
            logger.error("Failed to load VRP weights: %s", e)

    def get_state_embedding(self, grid_history: list[dict]) -> torch.Tensor:
        """Convert grid history dicts → 256-dim embedding for TD3.

        Parameters
        ----------
        grid_history : list[dict]
            Last 10 daily grid snapshots. Each dict has key ``"grid"``
            containing a ``[5, 25]`` IV grid (np.ndarray or list).

        Returns
        -------
        torch.Tensor
            Shape ``[1, 256]``.
        """
        if not self.is_loaded or len(grid_history) < 10:
            return torch.zeros(1, 256, device=self.device)

        try:
            # Stack last 10 grids → [10, 5, 25] → flatten → [10, 125]
            grids = []
            for snap in grid_history[-10:]:
                g = snap.get("grid")
                if g is None:
                    g = np.zeros((5, 25), dtype=np.float32)
                if isinstance(g, list):
                    g = np.array(g, dtype=np.float32)
                grids.append(g.reshape(-1))  # [125]

            x = torch.tensor(np.stack(grids), dtype=torch.float32)
            x = x.unsqueeze(0).to(self.device)  # [1, 10, 125]

            with torch.inference_mode():
                return self.model(x)

        except Exception as e:
            logger.error("State embedding failed: %s", e)
            return torch.zeros(1, 256, device=self.device)

    @torch.inference_mode()
    def forecast(self, grid_history: list[dict]) -> tuple[float, float, float]:
        """Forecast the Volatility Risk Premium.

        Returns
        -------
        tuple[float, float, float]
            (vrp_prediction, uncertainty, drift_score)
        """
        if not self.is_loaded or len(grid_history) < 10:
            return 0.0, 1.0, 1.0

        try:
            embed = self.get_state_embedding(grid_history)
            vrp_pred = self.regression_head(embed).item()

            # Simple uncertainty: magnitude of prediction
            uncertainty = abs(vrp_pred) / 10.0

            # Drift: how much the latest grid changed
            if len(grid_history) >= 2:
                g_last = np.array(grid_history[-1].get("grid", np.zeros((5, 25))))
                g_prev = np.array(grid_history[-2].get("grid", np.zeros((5, 25))))
                drift = float(np.mean(np.abs(g_last - g_prev)))
            else:
                drift = 0.0

            return vrp_pred, uncertainty, drift

        except Exception as e:
            logger.error("VRP forecast failed: %s", e)
            return 0.0, 1.0, 1.0
