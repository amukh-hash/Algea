from backend.app.ml_platform.training.jobs.vol_surface_grid_forecaster import TrainVolSurfaceGridForecasterJob


def test_vol_surface_grid_artifact_contract(tmp_path):
    job = TrainVolSurfaceGridForecasterJob(job_type="train_vol_surface_grid_forecaster", model_name="vol_surface_grid", version="v1", grid_history=[])
    out = job.run(tmp_path / "x")
    for f in ["model_config.json", "metrics.json", "drift_baseline.json", "weights.safetensors"]:
        assert (out["artifact_dir"] / f).exists()
