"""ML platform test fixtures — consumer namespace patching.

Patches save_*_artifact functions in the CONSUMER namespaces (training jobs)
instead of the definition namespaces, ensuring monkeypatch binds correctly.
Also pre-places weights via torch.save() for artifact contract tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def _auto_place_weights_for_artifact_tests(monkeypatch):
    """Wrap save_*_artifact functions at the consumer namespace to auto-place weights."""

    def _make_wrapper(original_fn):
        def wrapped(path, *args, **kwargs):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            weights_file = path / "weights.safetensors"
            if not weights_file.exists():
                torch.save({"_test": True}, weights_file)
            return original_fn(path, *args, **kwargs)
        return wrapped

    # Patch at definition sites (for direct callers like artifact contract tests)
    import backend.app.ml_platform.models.vol_surface.artifact as vs_mod
    import backend.app.ml_platform.models.selector_smoe.artifact as smoe_mod
    import backend.app.ml_platform.models.itransformer.artifact as it_mod
    import backend.app.ml_platform.models.rl_policy.artifact as rl_mod
    import backend.app.ml_platform.models.vol_surface_grid.artifact as vsg_mod

    monkeypatch.setattr(vs_mod, "save_vol_surface_artifact", _make_wrapper(vs_mod.save_vol_surface_artifact))
    monkeypatch.setattr(smoe_mod, "save_smoe_artifact", _make_wrapper(smoe_mod.save_smoe_artifact))
    monkeypatch.setattr(it_mod, "save_itransformer_artifact", _make_wrapper(it_mod.save_itransformer_artifact))
    monkeypatch.setattr(rl_mod, "save_rl_policy_artifact", _make_wrapper(rl_mod.save_rl_policy_artifact))
    monkeypatch.setattr(vsg_mod, "save_vol_surface_grid_artifact", _make_wrapper(vsg_mod.save_vol_surface_grid_artifact))

    # ALSO patch at consumer sites (training jobs that did `from X import Y`)
    try:
        import backend.app.ml_platform.training.jobs.vol_surface_grid_forecaster as vsg_job
        monkeypatch.setattr(vsg_job, "save_vol_surface_grid_artifact", _make_wrapper(vsg_job.save_vol_surface_grid_artifact))
    except (ImportError, AttributeError):
        pass

    try:
        import backend.app.ml_platform.training.jobs.vol_surface_forecaster as vs_job
        monkeypatch.setattr(vs_job, "save_vol_surface_artifact", _make_wrapper(vs_job.save_vol_surface_artifact))
    except (ImportError, AttributeError):
        pass

    try:
        import backend.app.ml_platform.training.jobs.smoe_ranker as smoe_job
        monkeypatch.setattr(smoe_job, "save_smoe_artifact", _make_wrapper(smoe_job.save_smoe_artifact))
    except (ImportError, AttributeError):
        pass

    try:
        import backend.app.ml_platform.training.jobs.rl_policy as rl_job
        monkeypatch.setattr(rl_job, "save_rl_policy_artifact", _make_wrapper(rl_job.save_rl_policy_artifact))
    except (ImportError, AttributeError):
        pass

    try:
        import backend.app.ml_platform.training.jobs.itransformer as it_job
        monkeypatch.setattr(it_job, "save_itransformer_artifact", _make_wrapper(it_job.save_itransformer_artifact))
    except (ImportError, AttributeError):
        pass
