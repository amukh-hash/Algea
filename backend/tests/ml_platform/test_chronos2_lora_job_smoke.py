from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.ml_platform.training.jobs.chronos2_lora import TrainChronos2LoRAJob
from backend.app.ml_platform.training.trainerd import TrainerDaemon


def test_chronos2_lora_job_publishes_staging(tmp_path: Path) -> None:
    store = ModelRegistryStore(tmp_path / "reg.sqlite", tmp_path / "models")
    daemon = TrainerDaemon(store)
    job = TrainChronos2LoRAJob(
        job_type="train_chronos2_lora",
        model_name="chronos2",
        version="v2",
        universe_id="fut",
        instrument_ids=["ES"],
        freq="1d",
        context_length=4,
        prediction_length=2,
        train_start="2024-01-01",
        train_end="2024-06-01",
        val_start="2024-06-02",
        val_end="2024-07-01",
        series=[100, 101, 102, 103, 104, 105, 106],
    )
    daemon.run_job(job)
    assert store.resolve_alias("chronos2", "staging") == "v2"
    model_dir = tmp_path / "models" / "chronos2" / "v2"
    assert (model_dir / "weights.safetensors").exists()
    assert (model_dir / "metrics.json").exists()
