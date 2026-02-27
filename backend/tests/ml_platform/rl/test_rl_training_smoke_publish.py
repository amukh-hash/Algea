from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.ml_platform.training.jobs.rl_policy import TrainRLPolicyJob
from backend.app.ml_platform.training.trainerd import TrainerDaemon


def test_rl_training_smoke_publish(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "tr")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    trainer = TrainerDaemon(store, cfg)
    job = TrainRLPolicyJob(job_type="train_rl_policy", model_name="rl_policy", version="v1", steps=8)
    trainer.run_job(job)
    assert store.resolve_alias("rl_policy", "staging") == "v1"
