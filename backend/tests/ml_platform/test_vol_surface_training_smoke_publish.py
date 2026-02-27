from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.ml_platform.training.jobs.vol_surface_forecaster import TrainVolSurfaceForecasterJob
from backend.app.ml_platform.training.trainerd import TrainerDaemon


def test_vol_surface_training_publish(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    d = TrainerDaemon(store)
    job = TrainVolSurfaceForecasterJob(
        job_type="train_vol_surface_forecaster",
        model_name="vol_surface",
        version="v1",
        history={7: [{"rv_hist_20": 0.2}] * 10, 30: [{"rv_hist_20": 0.3}] * 10},
        labels={7: 0.2, 30: 0.3},
    )
    d.run_job(job)
    assert store.resolve_alias("vol_surface", "staging") == "v1"
