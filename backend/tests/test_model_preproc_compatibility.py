import pytest
from unittest.mock import MagicMock, patch
from backend.app.models.teacher_equity_inference import TeacherERunner
from backend.app.models.signal_types import ModelMetadata

@patch("backend.app.models.teacher_equity_inference.Preprocessor")
def test_preproc_id_mismatch(mock_preproc_cls):
    # Setup Mocks
    mock_preproc = MagicMock()
    mock_preproc.version_hash = "abc"
    mock_preproc_cls.load.return_value = mock_preproc
    
    # Mock Model IO with Different ID
    mock_meta = ModelMetadata(
        model_version="v1", preproc_id="xyz", training_start="", training_end=""
    )
    
    # We patch load_model ONLY, so verify_preproc_compatibility is real
    with patch("backend.app.models.teacher_equity_inference.model_io.load_model") as mock_load:
        mock_load.return_value = ({}, mock_meta)
        
        # We need to pass a mock model class to avoid real model instantiation/loading
        mock_model_cls = MagicMock()
        
        with pytest.raises(ValueError, match="Model requires preproc xyz, but active is abc"):
            TeacherERunner("path", "path", model_class=mock_model_cls)
