from pathlib import Path

import pytest

from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_contract_validation_fails_when_missing_required_file(tmp_path: Path) -> None:
    store = ModelRegistryStore(tmp_path / "reg.sqlite", tmp_path / "models")
    artifact_dir = tmp_path / "incomplete"
    artifact_dir.mkdir()
    (artifact_dir / "model_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        store.publish_artifact_directory(
            model_name="chronos2",
            version="v0",
            artifact_dir=artifact_dir,
            sha256="x",
            metrics={},
            config={},
            data_lineage={},
        )
