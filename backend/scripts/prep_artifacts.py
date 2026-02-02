"""
prep_artifacts.py - Generate Required Model Artifacts

Run this BEFORE training to:
1. Fit and save RobustScaler
2. Train and save HMM regime detector
3. Standardize TBM labels (0=Neutral, 1=Buy, 2=Sell)
4. Save processed training data

Usage: python prep_artifacts.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM

# --- CONFIGURATION ---
CONFIG = {
    'output_dir': 'backend/models/',
    'processed_data_path': 'backend/data/processed/train_ready.parquet',
    'seq_len': 60,
    'input_dim': 64,
}


def map_labels(triple_barrier_label):
    """
    Standardize TBM labels to 0, 1, 2
    Input (Triple Barrier): -1 (Sell), 0 (Neutral), 1 (Buy)
    Output (Model):          0 (Neutral), 1 (Buy), 2 (Sell)
    """
    if triple_barrier_label == 0:
        return 0
    if triple_barrier_label == 1:
        return 1
    if triple_barrier_label == -1:
        return 2
    return 0  # Default to neutral


def main():
    print("=" * 50)
    print("ARTIFACT GENERATION")
    print("=" * 50)
    
    # Ensure directories exist
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path('backend/data/processed').mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data (using mock client for demo)
    print("\n1. Loading data...")
    try:
        from app.api.databento_client import DatabentoClient
        from app.features.signal_processing import triple_barrier_labels
        
        client = DatabentoClient(mock_mode=True)
        df = client.get_historical_range("BTC-USD", "2023-01-01", "2023-03-01", schema='mbp-10')
        
        if 'price' not in df.columns:
            df['price'] = (df['bid_px_00'] + df['ask_px_00']) / 2.0
        
        print(f"   Loaded {len(df)} rows")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Creating mock data...")
        df = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(10000) * 0.01),
            'bid_px_00': 99.95 + np.cumsum(np.random.randn(10000) * 0.01),
            'ask_px_00': 100.05 + np.cumsum(np.random.randn(10000) * 0.01),
        })
    
    # 2. Generate Features
    print("\n2. Generating features...")
    prices = df['price'].values
    
    # Returns
    returns = np.zeros(len(prices))
    returns[1:] = np.diff(prices) / prices[:-1]
    df['returns'] = returns
    
    # Volatility (rolling 20)
    df['volatility'] = df['returns'].rolling(20).std().bfill()
    
    # RSI (simplified)
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].bfill()
    
    # Spread
    if 'bid_px_00' in df.columns and 'ask_px_00' in df.columns:
        df['spread'] = df['ask_px_00'] - df['bid_px_00']
    else:
        df['spread'] = 0.1
    
    print(f"   Features: returns, volatility, rsi, spread")
    
    # 3. Train HMM
    print("\n3. Training HMM regime detector...")
    valid_returns = df['returns'].replace([np.inf, -np.inf], np.nan).dropna()
    X_hmm = valid_returns.values.reshape(-1, 1)
    
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    hmm.fit(X_hmm)
    
    hmm_path = os.path.join(CONFIG['output_dir'], 'hmm_regime.pkl')
    joblib.dump(hmm, hmm_path)
    print(f"   ✅ HMM saved to {hmm_path}")
    
    # Add regime to features
    df['regime'] = 0
    valid_mask = ~df['returns'].isna()
    df.loc[valid_mask, 'regime'] = hmm.predict(df.loc[valid_mask, 'returns'].values.reshape(-1, 1))
    
    # 4. Fit Scaler
    print("\n4. Fitting RobustScaler...")
    features_to_scale = ['returns', 'volatility', 'rsi', 'spread', 'regime']
    
    scaler = RobustScaler()
    scaler.fit(df[features_to_scale].fillna(0))
    
    scaler_path = os.path.join(CONFIG['output_dir'], 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler saved to {scaler_path}")
    
    # 5. Generate Labels
    print("\n5. Generating Triple Barrier labels...")
    try:
        vol_series = df['price'].rolling(20).std().bfill()
        from app.features.signal_processing import triple_barrier_labels
        tbm = triple_barrier_labels(df['price'], vol_series)
        df['tbm_label'] = tbm.values
    except Exception as e:
        print(f"   Warning: {e}")
        print("   Using random labels for demo...")
        df['tbm_label'] = np.random.choice([-1, 0, 1], size=len(df), p=[0.2, 0.6, 0.2])
    
    # Standardize labels
    df['target'] = df['tbm_label'].apply(map_labels)
    
    print(f"   Label distribution:")
    print(f"   {df['target'].value_counts().to_dict()}")
    
    # 6. Create Sequences
    print("\n6. Creating sequences...")
    seq_len = CONFIG['seq_len']
    feature_cols = ['returns', 'volatility', 'rsi', 'spread', 'regime']
    
    X_list = []
    y_list = []
    
    for i in range(len(df) - seq_len):
        seq = df[feature_cols].iloc[i:i + seq_len].values
        label = df['target'].iloc[i + seq_len]
        X_list.append(seq)
        y_list.append(label)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"   Created {len(X)} sequences of shape {X.shape}")
    
    # 7. Save Processed Data
    print("\n7. Saving processed data...")
    
    # Save as NPZ for faster loading
    npz_path = os.path.join('backend/data/processed', 'train_sequences.npz')
    np.savez(npz_path, X=X, y=y)
    print(f"   ✅ Sequences saved to {npz_path}")
    
    # Also save flat version for reference
    df.to_parquet(CONFIG['processed_data_path'])
    print(f"   ✅ DataFrame saved to {CONFIG['processed_data_path']}")
    
    print("\n" + "=" * 50)
    print("ARTIFACT GENERATION COMPLETE")
    print("=" * 50)
    print("\nGenerated artifacts:")
    print(f"  - {hmm_path}")
    print(f"  - {scaler_path}")
    print(f"  - {npz_path}")
    print(f"  - {CONFIG['processed_data_path']}")
    print("\nNext: Run train_teacher_t5.py")


if __name__ == "__main__":
    main()
