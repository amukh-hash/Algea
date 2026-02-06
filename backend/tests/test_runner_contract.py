import pytest
import torch
import polars as pl
import numpy as np
from unittest.mock import MagicMock, patch
from backend.app.models.teacher_equity_inference import TeacherERunner
from backend.app.models.signal_types import ModelSignal, ModelMetadata

@patch("backend.app.models.teacher_equity_inference.Preprocessor")
@patch("backend.app.models.teacher_equity_inference.model_io")
def test_teacher_runner_contract(mock_io, mock_preproc_cls):
    # Setup Mocks
    mock_preproc = MagicMock()
    mock_preproc.version_hash = "abc"
    mock_preproc_cls.load.return_value = mock_preproc
    
    # Mock Transform: returns df with required columns
    def mock_transform(df):
        # Return dummy DF with same length but with required feature cols
        # cols = ["log_ret", "volume_norm", "ad_line_norm", "bpi_norm"]
        n = df.height
        return pl.DataFrame({
            "log_ret": np.zeros(n),
            "volume_norm": np.zeros(n),
            "ad_line_norm": np.zeros(n),
            "bpi_norm": np.zeros(n)
        })
    mock_preproc.transform.side_effect = mock_transform
    
    # Mock Model IO
    mock_meta = ModelMetadata(
        model_version="v1", preproc_id="abc", training_start="", training_end=""
    )
    # State dict: empty for mock model
    mock_io.load_model.return_value = ({}, mock_meta)
    
    # Mock Model Class
    mock_model = MagicMock(spec=torch.nn.Module)
    # Output: (Batch, Horizons=2, Quantiles=3)
    mock_model.return_value = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]) 
    
    # Instantiate
    # Pass mock_model as model_class (constructor returns instance)
    mock_model_cls = MagicMock()
    mock_model_cls.return_value = mock_model
    
    runner = TeacherERunner("path", "path", model_class=mock_model_cls)
    
    # Infer
    df_in = pl.DataFrame({"timestamp": [1], "close": [100]}) # Dummy input
    signal = runner.infer(df_in)
    
    # Verify Contract
    assert isinstance(signal, ModelSignal)
    assert signal.horizons == ["1D", "3D"]
    assert "1D" in signal.quantiles
    assert signal.quantiles["1D"]["0.05"] == pytest.approx(0.1)
    assert signal.quantiles["1D"]["0.50"] == pytest.approx(0.2)
    assert signal.quantiles["1D"]["0.95"] == pytest.approx(0.3)
    
    assert signal.metadata.model_version == "v1"
