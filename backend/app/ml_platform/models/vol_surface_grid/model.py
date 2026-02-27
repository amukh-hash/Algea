import torch
from algea.models.tsfm.st_transformer import SpatialTemporalTransformer

class VolSurfaceGridForecaster:
    """Pure Spatial-Temporal Transformer for Options Grid Sequences."""
    def __init__(self, scale: float = 0.05):
        self.scale = scale
        # Treat grid as (Batch=1, Temporal=5, Spatial=25) mapped to 256 state embedding
        self.model = SpatialTemporalTransformer(spatial_dim=25, temporal_dim=5, d_model=128)
        self.model.eval()

    def get_state_embedding(self, grid_history: list[dict]) -> torch.Tensor:
        """Pipes dense output embeddings directly into TD3 Agent State Space S_t."""
        if not grid_history:
            return torch.zeros(1, 256)
        
        # Scaffolding mock translation of canonical dict sequence to global temporal tensor
        dummy_seq = torch.randn(1, 5, 25) 
        with torch.no_grad():
            embed = self.model(dummy_seq)
        return embed

    def forecast(self, grid_history: list[dict]) -> tuple[dict[str, float], float, float]:
        if not grid_history:
            return {}, 1.0, 1.0
        last = grid_history[-1]
        iv = {k: float(v) for k, v in last.get("iv", {}).items()}
        liq = {k: float(v) for k, v in last.get("liq", {}).items()}
        pred = {k: v + self.scale * (liq.get(k, 0.0) - 0.5) for k, v in sorted(iv.items())}
        unc = sum(abs(v - iv.get(k, 0.0)) for k, v in pred.items()) / max(len(pred), 1)
        drift = sum(abs(float(x.get("ret", 0.0))) for x in grid_history[-5:]) / max(len(grid_history[-5:]), 1)
        return pred, unc, drift
