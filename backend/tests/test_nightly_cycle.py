import json
from datetime import date
from pathlib import Path

import pandas as pd

from backend.scripts.run.run_nightly_cycle import run


def test_nightly_cycle_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    artifact_root = tmp_path / "artifacts"
    config_path.write_text(json.dumps({"artifact_root": str(artifact_root)}), encoding="utf-8")

    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "AAA"],
            "open": [10.0, 11.0],
            "high": [12.0, 12.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "volume": [100.0, 110.0],
        }
    )
    input_path = tmp_path / "canonical.parquet"
    df.to_parquet(input_path, index=False)

    run(str(config_path), [input_path], date.fromisoformat("2024-01-02"))
    summary_path = artifact_root / "reports" / "nightly" / "2024-01-02" / "summary.json"
    assert summary_path.exists()
