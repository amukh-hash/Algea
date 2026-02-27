from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.ml_platform.training.jobs.vol_surface_grid_forecaster import TrainVolSurfaceGridForecasterJob
from backend.app.ml_platform.training.trainerd import TrainerDaemon


def test_vol_surface_grid_training_smoke_publish(tmp_path):
    store = ModelRegistryStore(tmp_path / "registry.db", tmp_path / "models")
    daemon = TrainerDaemon(registry=store)
    sha = daemon.run_job(
        TrainVolSurfaceGridForecasterJob(
            job_type="train_vol_surface_grid_forecaster",
            model_name="vol_surface_grid",
            version="v1",
            grid_history=[{"iv": {"7:ATM": 0.2}, "target": {"7:ATM": 0.21}}],
        )
    )
    assert sha
    assert store.resolve_alias("vol_surface_grid", "staging") == "v1"
