import json
import os
from datetime import datetime
from typing import List, Dict
from backend.app.core.config import OPTIONS_DATA_VERSION

class GateTuner:
    def __init__(self):
        pass

    def tune(self, dataset: List[Dict], metric: str = "precision") -> Dict:
        # Mock tuning: Find threshold for 'student_p50_3d' that maximizes precision
        # scan range -0.05 to 0.05

        best_thresh = -0.01
        best_score = 0.0

        candidates = [-0.05, -0.02, -0.01, 0.0, 0.01]

        for thresh in candidates:
            tp = 0
            fp = 0
            for row in dataset:
                pred = row["features"]["student_p50_3d"] > thresh
                label = row["label"]

                if pred and label: tp += 1
                if pred and not label: fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if precision > best_score and (tp+fp) > 10: # Min support
                best_score = precision
                best_thresh = thresh

        return {
            "min_student_p50": best_thresh,
            "min_bpi": 40.0, # fixed
            "score": best_score
        }

    def save_artifacts(self, params: Dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "gate_params_v1.json")

        artifact = {
            "params": params,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "options_data_version": OPTIONS_DATA_VERSION,
                "marketframe_schema_hash": "mock_hash",
                "options_schema_version": 1
            }
        }

        with open(path, "w") as f:
            json.dump(artifact, f, indent=2)
