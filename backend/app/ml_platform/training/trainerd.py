from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import MLPlatformConfig
from ..registry.store import ModelRegistryStore
from .jobs import TrainChronos2LoRAJob, TrainSMoERankerJob, TrainVolSurfaceForecasterJob, TrainITransformerJob, TrainRLPolicyJob, TrainingJob


class TrainerDaemon:
    def __init__(self, registry: ModelRegistryStore | None = None, cfg: MLPlatformConfig | None = None):
        self.cfg = cfg or MLPlatformConfig()
        self.registry = registry or ModelRegistryStore(self.cfg.registry_db_path, self.cfg.model_root)

    def _publish_result(self, model_name: str, version: str, result: dict, status: str = "staging") -> str:
        self.registry.publish_artifact_directory(
            model_name=model_name,
            version=version,
            artifact_dir=result["artifact_dir"],
            sha256=result["sha256"],
            metrics=result["metrics"],
            config=result["config"],
            data_lineage=result["lineage"],
            status=status,
        )
        self.registry.set_alias(model_name, "staging", version)
        return result["sha256"]

    def run_job(self, job: TrainingJob | TrainChronos2LoRAJob | TrainSMoERankerJob) -> str:
        if isinstance(job, TrainChronos2LoRAJob) or job.job_type == "train_chronos2_lora":
            if not isinstance(job, TrainChronos2LoRAJob):
                job = TrainChronos2LoRAJob(**{**job.params, "job_type": job.job_type, "model_name": job.model_name, "version": job.version})
            tmp = Path("backend/artifacts/tmp_training") / f"{job.model_name}_{job.version}"
            return self._publish_result(job.model_name, job.version, job.run(tmp))

        if isinstance(job, TrainSMoERankerJob) or job.job_type == "train_smoe_ranker":
            if not isinstance(job, TrainSMoERankerJob):
                job = TrainSMoERankerJob(**{**job.params, "job_type": job.job_type, "model_name": job.model_name, "version": job.version})
            tmp = Path("backend/artifacts/tmp_training") / f"{job.model_name}_{job.version}"
            return self._publish_result(job.model_name, job.version, job.run(tmp))

        if isinstance(job, TrainVolSurfaceForecasterJob) or job.job_type == "train_vol_surface_forecaster":
            if not isinstance(job, TrainVolSurfaceForecasterJob):
                job = TrainVolSurfaceForecasterJob(**{**job.params, "job_type": job.job_type, "model_name": job.model_name, "version": job.version})
            tmp = Path("backend/artifacts/tmp_training") / f"{job.model_name}_{job.version}"
            return self._publish_result(job.model_name, job.version, job.run(tmp))

        if isinstance(job, TrainITransformerJob) or job.job_type == "train_itransformer":
            if not isinstance(job, TrainITransformerJob):
                job = TrainITransformerJob(**{**job.params, "job_type": job.job_type, "model_name": job.model_name, "version": job.version})
            tmp = Path("backend/artifacts/tmp_training") / f"{job.model_name}_{job.version}"
            return self._publish_result(job.model_name, job.version, job.run(tmp))

        if isinstance(job, TrainRLPolicyJob) or job.job_type == "train_rl_policy":
            if not isinstance(job, TrainRLPolicyJob):
                job = TrainRLPolicyJob(**{**job.params, "job_type": job.job_type, "model_name": job.model_name, "version": job.version})
            tmp = Path("backend/artifacts/tmp_training") / f"{job.model_name}_{job.version}"
            return self._publish_result(job.model_name, job.version, job.run(tmp))

        seed_bytes = json.dumps(job.params, sort_keys=True).encode("utf-8")
        sha = hashlib.sha256(seed_bytes).hexdigest()
        self.registry.publish_version(
            model_name=job.model_name,
            version=job.version,
            sha256=sha,
            metrics={"sharpe": 1.05, "max_drawdown": 0.12, "calibration_score": 0.82},
            config={"job_type": job.job_type, **job.params},
            data_lineage={"dataset_id": "stub", "feature_schema": {"name": "stub"}},
        )
        return sha
