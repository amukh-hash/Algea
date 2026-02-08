"""
Experiment configuration loader.
Reads the frozen YAML config and provides typed access to all parameters.
"""
import yaml
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

_DEFAULT_CONFIG_PATH = Path("backend/config/experiment_v1.yaml")
_cached_config = None


def load_experiment_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and cache the experiment configuration."""
    global _cached_config
    if _cached_config is not None and path is None:
        return _cached_config

    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if path is None:
        _cached_config = config
    return config


def config_hash(path: Optional[Path] = None) -> str:
    """Compute a stable hash of the experiment config for reproducibility tracking."""
    config_path = path or _DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def get_split_dates(config: Optional[Dict] = None) -> Dict[str, Dict[str, str]]:
    """Extract train/val/test date ranges from config."""
    if config is None:
        config = load_experiment_config()
    return config["splits"]


def get_chronos_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Extract Chronos teacher configuration."""
    if config is None:
        config = load_experiment_config()
    return config["chronos_teacher"]


def get_selector_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Extract Selector training configuration."""
    if config is None:
        config = load_experiment_config()
    return config["selector"]


def get_priors_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Extract Priors generation configuration."""
    if config is None:
        config = load_experiment_config()
    return config["priors"]


def get_simulation_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Extract simulation configuration."""
    if config is None:
        config = load_experiment_config()
    return config["simulation"]
