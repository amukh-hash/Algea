import os
import sys
import json
from backend.app.options.gate.tuning import GateTuner

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

def main():
    print("Tuning options gate...")

    path = "backend/data/options/gate/dataset.json"
    if not os.path.exists(path):
        print("Dataset not found. Run build_gate_dataset.py first.")
        return

    with open(path, "r") as f:
        dataset = json.load(f)

    tuner = GateTuner()
    params = tuner.tune(dataset)

    print("Best Params:", params)

    tuner.save_artifacts(params, "backend/models/options_gate")
    print("Artifacts saved.")

if __name__ == "__main__":
    main()
