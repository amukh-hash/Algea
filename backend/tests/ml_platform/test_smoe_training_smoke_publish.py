from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.ml_platform.training.jobs.smoe_ranker import TrainSMoERankerJob
from backend.app.ml_platform.training.trainerd import TrainerDaemon


def test_smoe_training_smoke_publish(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    d = TrainerDaemon(store)
    job = TrainSMoERankerJob(
        job_type="train_smoe_ranker",
        model_name="selector_smoe",
        version="v1",
        feature_matrix=[[0.1, 0.2], [0.2, -0.1], [0.3, 0.1]],
        labels=[0.01, -0.02, 0.03],
    )
    d.run_job(job)
    assert store.resolve_alias("selector_smoe", "staging") == "v1"
