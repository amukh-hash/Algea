from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.ml_platform.training.jobs.chronos2_lora import TrainChronos2LoRAJob
from backend.app.ml_platform.training.trainerd import TrainerDaemon


def test_training_job_only_sets_staging_alias(tmp_path: Path) -> None:
    store = ModelRegistryStore(tmp_path / "reg.sqlite", tmp_path / "models")
    daemon = TrainerDaemon(store)
    job = TrainChronos2LoRAJob(
        job_type="train_chronos2_lora",
        model_name="chronos2",
        version="v-staging",
        universe_id="u",
        instrument_ids=["ES"],
        freq="1d",
        context_length=3,
        prediction_length=1,
        train_start="2024-01-01",
        train_end="2024-01-10",
        val_start="2024-01-11",
        val_end="2024-01-20",
        series=[1, 2, 3, 4, 5],
    )
    daemon.run_job(job)
    assert store.resolve_alias("chronos2", "staging") == "v-staging"
    assert store.resolve_alias("chronos2", "prod") is None
