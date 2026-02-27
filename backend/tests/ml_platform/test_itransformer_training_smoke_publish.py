from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.ml_platform.training.jobs.itransformer import TrainITransformerJob
from backend.app.ml_platform.training.trainerd import TrainerDaemon


def test_itransformer_training_publish(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    daemon = TrainerDaemon(store)
    job = TrainITransformerJob(
        job_type="train_itransformer",
        model_name="itransformer",
        version="v1",
        feature_matrix=[[0.1, 0.2], [0.2, 0.1]],
        labels=[0.01, -0.01],
    )
    daemon.run_job(job)
    assert store.resolve_alias("itransformer", "staging") == "v1"
