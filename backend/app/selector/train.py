
import os
from typing import Dict
import torch
from backend.app.ops import pathmap
from backend.app.models.rank_transformer import RankTransformer
from backend.app.models.selector_scaler import SelectorFeatureScaler
from backend.app.models.calibration import ScoreCalibrator
from backend.app.ops import promotion_gate, artifact_registry

def train_selector(run_cfg: Dict) -> Dict:
    print(f"Wrapper: train_selector({run_cfg})")

    dataset_path = pathmap.resolve("dataset_selector")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset artifact: {dataset_path}")

    dataset = torch.load(dataset_path)
    X_list = dataset["X"]
    y_list = dataset["y"]

    if not X_list or not y_list:
        raise ValueError("Empty dataset artifacts.")

    feature_cols = dataset.get("feature_cols", [])
    prior_cols = dataset.get("prior_cols", [])

    X_flat = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_list], dim=0).numpy()
    scaler = SelectorFeatureScaler(version="v1", feature_names=feature_cols + prior_cols)
    scaler.fit(X_flat)

    input_dim = X_list[0].shape[-1]
    model = RankTransformer(d_input=input_dim, d_model=64, n_head=2, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = run_cfg.get("epochs", 5)
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in zip(X_list, y_list):
            X_scaled = scaler.transform(X).unsqueeze(0)
            scores = model(X_scaled)["score"].squeeze(0)

            y_soft = torch.softmax(y, dim=0)
            s_soft = torch.softmax(scores, dim=0)
            loss = -(y_soft * torch.log(s_soft + 1e-9)).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(X_list):.6f}")

    model_path = pathmap.resolve("model_selector", version="v1")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    meta = {
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "prior_cols": prior_cols,
        "sequence_len": dataset.get("sequence_len", None)
    }
    artifact_registry.write_metadata(model_path + ".meta", meta)

    paths = pathmap.get_paths()
    scaler_path = os.path.join(paths.calibration, "selector_scaler_v1.joblib")
    scaler.save(scaler_path)

    all_scores = []
    all_targets = []
    with torch.no_grad():
        for X, y in zip(X_list, y_list):
            X_scaled = scaler.transform(X).unsqueeze(0)
            scores = model(X_scaled)["score"].squeeze(0)
            all_scores.append(scores)
            all_targets.append(y)
    all_scores = torch.cat(all_scores).numpy()
    all_targets = torch.cat(all_targets).numpy()

    calibrator = ScoreCalibrator(version="v1")
    calibrator.fit(all_scores, all_targets)
    calib_path = os.path.join(paths.calibration, "selector_calibration_v1.joblib")
    calibrator.save(calib_path)

    versions = {
        "model_version": "v1",
        "feature_version": "v1",
        "prior_version": "v1",
        "cal_version": "v1"
    }
    promotion_gate.promote_model(run_id="v1", metrics={}, versions=versions)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "calibration_path": calib_path
    }
