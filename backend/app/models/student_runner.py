import torch
import polars as pl
import numpy as np
from typing import Optional, Type
from backend.app.preprocessing.preproc import Preprocessor
from backend.app.models import model_io
from backend.app.models.signal_types import ModelSignal
from backend.app.models.baseline import BaselineMLP

class StudentRunner:
    def __init__(self, model_path: str, preproc_path: str, model_class: Type[torch.nn.Module] = BaselineMLP, device: str = "cpu"):
        self.device = device
        self.preproc = Preprocessor.load(preproc_path)
        state_dict, self.metadata = model_io.load_model(model_path, device=device)
        model_io.verify_preproc_compatibility(self.metadata, self.preproc.version_hash)
        
        # Hardcoded dims for Phase 1
        input_dim = 4
        lookback = 128
        
        self.model = model_class(input_dim=input_dim, lookback=lookback)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
    def infer(self, df_window: pl.DataFrame) -> ModelSignal:
        df_trans = self.preproc.transform(df_window)
        cols = ["log_ret", "volume_norm", "ad_line_norm", "bpi_norm"]
        data = df_trans.select(cols).to_numpy().astype(np.float32)
        x = torch.from_numpy(data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x)
            
        q_vals = out.cpu().numpy()[0]
        quantiles = {}
        horizons = ["1D", "3D"]
        q_levels = ["0.05", "0.50", "0.95"]
        
        for i, h in enumerate(horizons):
            quantiles[h] = {ql: float(q_vals[i, j]) for j, ql in enumerate(q_levels)}
            
        direction_probs = {}
        for h in horizons:
            median = quantiles[h]["0.50"]
            direction_probs[h] = 0.6 if median > 0 else 0.4
            
        return ModelSignal(
            horizons=horizons,
            quantiles=quantiles,
            direction_probs=direction_probs,
            metadata=self.metadata
        )
