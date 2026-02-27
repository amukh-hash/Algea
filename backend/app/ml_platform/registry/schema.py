from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ModelVersionRecord:
    model_name: str
    version: str
    sha256: str
    created_at: datetime
    status: str
    metrics_json: str
    config_json: str
    data_lineage_json: str
