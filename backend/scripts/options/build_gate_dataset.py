import os
import sys
import json
import random
import numpy as np

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

def main():
    print("Building synthetic gate dataset...")

    dataset = []
    for _ in range(1000):
        # Synthetic feature
        p50 = np.random.normal(0.001, 0.02)
        bpi = np.random.uniform(20, 80)

        # Synthetic label logic: correlated with p50
        prob_success = 0.5 + (p50 * 5) # simple linear
        label = 1 if random.random() < prob_success else 0

        row = {
            "features": {
                "student_p50_3d": p50,
                "bpi": bpi
            },
            "label": label
        }
        dataset.append(row)

    os.makedirs("backend/data/options/gate", exist_ok=True)
    with open("backend/data/options/gate/dataset.json", "w") as f:
        json.dump(dataset, f)

    print(f"Saved {len(dataset)} rows.")

if __name__ == "__main__":
    main()
