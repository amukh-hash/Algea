import torch
import polars as pl
import numpy as np
from typing import Optional, Type
from backend.app.preprocessing.preproc import Preprocessor
from backend.app.models import model_io
from backend.app.models.signal_types import ModelSignal, ModelMetadata
from backend.app.models.baseline import BaselineMLP

class TeacherERunner:
    def __init__(self, model_path: str, preproc_path: str, model_class: Type[torch.nn.Module] = BaselineMLP, device: str = "cpu"):
        self.device = device

        # Load Preproc
        self.preproc = Preprocessor.load(preproc_path)

        # Load Model
        state_dict, self.metadata = model_io.load_model(model_path, device=device)

        # Verify compatibility
        model_io.verify_preproc_compatibility(self.metadata, self.preproc.version_hash)

        # Init model
        # We need to know input dims to init model if it's dynamic.
        # For Phase 1, let's assume specific dimensions or save them in metadata?
        # Metadata should store model config.
        # For now, hardcode or try to infer from state_dict shape?
        # BaselineMLP args: input_dim, lookback.
        # We'll assume these are standard for Phase 1 (e.g. Lookback=512).
        # Or store in metadata.

        # Hack for Phase 1 Baseline: Hardcoded dims matching test/train script
        # Input dim depends on preproc output columns.
        # Preproc config has 5 columns: timestamp, log_ret, volume_norm, ad_line_norm, bpi_norm.
        # Timestamp is not feature. So 4 features.
        input_dim = 4
        lookback = 128 # Default for now, should be in metadata

        self.model = model_class(input_dim=input_dim, lookback=lookback)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def infer(self, df_window: pl.DataFrame) -> ModelSignal:
        # 1. Preprocess
        df_trans = self.preproc.transform(df_window)

        # 2. Tensorize
        # Select feature columns (exclude timestamp)
        # Order matters! Must match training.
        # Preproc transform returns specific columns.
        cols = ["log_ret", "volume_norm", "ad_line_norm", "bpi_norm"]

        # Check if columns exist
        data = df_trans.select(cols).to_numpy().astype(np.float32)

        x = torch.from_numpy(data).unsqueeze(0).to(self.device) # (1, L, F)

        # 3. Inference
        with torch.no_grad():
            # Output: (1, Horizons, Quantiles)
            out = self.model(x)

        # 4. Pack Signal
        # Assume output is 3 quantiles: 0.05, 0.5, 0.95
        q_vals = out.cpu().numpy()[0] # (H, Q)

        quantiles = {}
        horizons = ["1D", "3D"]
        q_levels = ["0.05", "0.50", "0.95"]

        for i, h in enumerate(horizons):
            quantiles[h] = {ql: float(q_vals[i, j]) for j, ql in enumerate(q_levels)}

        # Direction: Prob(Up).
        # Crude approximation from median? Or if model output prob.
        # Baseline outputs Quantiles.
        # If Median > 0 -> Up?
        # We can calculate probability mass > 0 if we assume distribution (e.g. Gaussian from quantiles).
        # For Phase 1, just use Median sign.
        direction_probs = {}
        for h in horizons:
            median = quantiles[h]["0.50"]
            direction_probs[h] = 0.6 if median > 0 else 0.4 # Dummy prob

        return ModelSignal(
            horizons=horizons,
            quantiles=quantiles,
            direction_probs=direction_probs,
            metadata=self.metadata
        )
