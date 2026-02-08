
import os
import pandas as pd
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
    dates = dataset.get("dates", [])

    if not X_list or not y_list:
        raise ValueError("Empty dataset artifacts.")

    feature_cols = dataset.get("feature_cols", [])
    prior_cols = dataset.get("prior_cols", [])
    label_horizon = int(dataset.get("label_horizon", 0))

    def split_indices():
        if not dates:
            return list(range(len(X_list))), [], []
        date_index = pd.to_datetime(pd.Series(dates)).sort_values().tolist()
        n = len(date_index)
        if n < 10:
            return list(range(n)), [], []

        val_pct = float(run_cfg.get("val_pct", 0.1))
        test_pct = float(run_cfg.get("test_pct", 0.1))
        train_end = int(n * (1.0 - val_pct - test_pct))
        val_end = int(n * (1.0 - test_pct))
        embargo = int(run_cfg.get("embargo_td", label_horizon))

        train_idx = list(range(0, max(train_end - embargo, 0)))
        val_idx = list(range(min(train_end + embargo, n), max(val_end - embargo, 0)))
        test_idx = list(range(min(val_end + embargo, n), n))
        return train_idx, val_idx, test_idx

    train_idx, val_idx, test_idx = split_indices()
    if not train_idx:
        raise ValueError("No training samples after split/embargo.")

    X_flat = torch.cat([X_list[i].reshape(-1, X_list[i].shape[-1]) for i in train_idx], dim=0).numpy()
    scaler = SelectorFeatureScaler(version="v1", feature_names=feature_cols + prior_cols)
    scaler.fit(X_flat)

    input_dim = X_list[0].shape[-1]
    model = RankTransformer(d_input=input_dim, d_model=64, n_head=2, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = run_cfg.get("epochs", 5)
    for epoch in range(epochs):
        total_loss = 0.0
        for idx in train_idx:
            X = X_list[idx]
            y = y_list[idx]
            X_scaled = scaler.transform(X).unsqueeze(0)
            scores = model(X_scaled)["score"].squeeze(0).squeeze(-1)

            y_soft = torch.softmax(y, dim=0)
            s_soft = torch.softmax(scores, dim=0)
            loss = -(y_soft * torch.log(s_soft + 1e-9)).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_idx):.6f}")

    model_path = pathmap.resolve("model_selector", version="v1")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    meta = {
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "prior_cols": prior_cols,
        "sequence_len": dataset.get("sequence_len", None),
        "label_horizon": label_horizon,
        "val_pct": run_cfg.get("val_pct", 0.1),
        "test_pct": run_cfg.get("test_pct", 0.1),
        "embargo_td": run_cfg.get("embargo_td", label_horizon)
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
            scores = model(X_scaled)["score"].squeeze(0).squeeze(-1)
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
