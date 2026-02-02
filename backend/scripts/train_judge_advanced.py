import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = "backend/data/judge_training_data.csv"
MODEL_PATH = "backend/models/judge_xgb.pkl"
METRICS_PATH = "backend/models/judge_metrics.json"

def train_judge():
    print("--- Training The Judge (XGBoost Meta-Model) ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}. Run generate_judge_data.py first.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Check basics
    print(f"Total rows: {len(df)}")
    print(df.head())
    
    # Filter for active predictions if needed?
    # Our data gen produces rows for every prediction.
    # Prediction label 0 = Neutral.
    # Judge usually only evaluates Buy/Sell signals (1, 2) to see if they are correct.
    # If pred=0, we don't really trade, so Judge is moot?
    # Usually Judge is "Should execute?" 
    
    # If data gen didn't predict, pred_label is 0.
    # But gen script generates predictions for valid indices.
    # Some might be Neutral.
    # Let's verify 'pred_label' in df
    
    if 'pred_label' not in df.columns:
        # data gen calculates pred_ret (params_q50) and label implicitly?
        # My new generate_judge_data.py writes 'true_label,true_ret,params_q50,uncertainty_spread,volatility,is_correct'
        # Wait, I didn't write 'pred_label' explicitly in the header, BUT 'is_correct' is derived from it.
        # And params_q50 is pred_ret.
        pass
        
    # We train on ALL samples or just active?
    # 'is_correct' = 1 if pred matches true for Buy/Sell.
    # If pred was 0, is_correct=0.
    # But if pred=0, we wouldn't call Judge live.
    # We should train only on cases where pred != 0?
    # BUT data gen logic: "if pred_ret > thresh: pred_label = 1".
    # So we can reconstruct pred_label or filter by params_q50 magnitude.
    
    # Let's filter for significant predictions (Buy/Sell intent)
    df['pred_label'] = 0
    df.loc[df['params_q50'] > 0.001, 'pred_label'] = 1
    df.loc[df['params_q50'] < -0.001, 'pred_label'] = 2
    
    active_mask = df['pred_label'].isin([1, 2])
    print(f"Active Signals: {active_mask.sum()}")
    
    if active_mask.sum() < 50:
         print("Not enough signals to train.")
         # Train on all for demo stability?
         df_train = df
    else:
         df_train = df[active_mask]
    
    # Features
    # Must match run_live_bolt.py extraction
    features = ['uncertainty_spread', 'volatility'] # Minimal set
    # 'params_q50' (pred return) is arguably a feature too (magnitude confidence)
    # Let's add it.
    features.append('params_q50')
    
    X = df_train[features]
    y = df_train['is_correct']
    
    print(f"Training on {len(X)} samples. features={features}")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='auc',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    model.fit(X, y)
    
    # Eval
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    print(f"In-Sample AUC: {auc:.4f}")
    
    # Save Feature Importance
    print(pd.Series(model.feature_importances_, index=features).sort_values(ascending=False))
    
    # Save as PKL
    joblib.dump(model, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_judge()
