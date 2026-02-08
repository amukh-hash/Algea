import os
import pytest
from unittest.mock import patch, MagicMock
from backend.app.core import artifacts, config

def test_resolve_priors_path():
    # Test valid path resolution
    with patch("os.path.exists", return_value=True):
        path = artifacts.resolve_priors_path("2024-01-01", "v1")
        expected = os.path.join(config.PRIORS_DIR, "v1", "2024-01-01.parquet")
        assert path == expected

def test_resolve_priors_path_missing():
    # Test missing file returns None
    with patch("os.path.exists", return_value=False):
        path = artifacts.resolve_priors_path("2024-01-01")
        assert path is None

def test_resolve_leaderboard_path():
    with patch("os.path.exists", return_value=True):
        path = artifacts.resolve_leaderboard_path("2024-01-01", "v1")
        expected = os.path.join(config.SIGNALS_DIR, "selector", "v1", "2024-01-01.parquet")
        assert path == expected

def test_ensure_compatibility_missing_file():
    # Should raise if any path is None or missing
    with pytest.raises(FileNotFoundError):
        artifacts.ensure_artifact_compatibility(None, "path/to/scaler", "path/to/ckpt")
    
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            artifacts.ensure_artifact_compatibility("path/priors", "path/scaler", "path/ckpt")

def test_manifest_path_generation():
    path = "backend/data/priors/v1/file.parquet"
    expected = "backend/data/priors/v1/file.parquet.manifest.json"
    assert artifacts.get_manifest_path(path) == expected

def test_load_manifest():
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("json.load", return_value={"version": "v1"}):
        
        man = artifacts.load_manifest("some/path")
        assert man == {"version": "v1"}

def test_load_manifest_missing():
    with patch("os.path.exists", return_value=False):
        man = artifacts.load_manifest("some/path")
        assert man == {}
