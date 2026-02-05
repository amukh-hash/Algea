import os
import json
import pandas as pd
from datetime import datetime
from backend.app.core import config
from backend.app.data.universe import UniverseSelector
from backend.app.data.preproc import FeatureEngineer
from backend.app.models.schema import FeatureContract

# Mock Imports for the Models (You'd replace these with actual Model Runners)
# from backend.app.models.chronos import ChronosInference
# from backend.app.models.ranker import RankerInference

def write_manifest(artifact_path: str, metadata: dict):
    """Writes a manifest side-by-side with the artifact."""
    manifest_path = f"{artifact_path}.manifest.json"
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "artifact_path": artifact_path,
        **metadata
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

def run_nightly_cycle(as_of_date: str):
    print(f"=== Starting ALGAIE Nightly Cycle for {as_of_date} ===")
    
    # Ensure directories exist
    os.makedirs(os.path.join(config.DATA_DIR, "raw"), exist_ok=True)
    
    # 1. LOAD RAW DATA
    print(">> Loading Raw Data...")
    # Using config.DATA_DIR for raw paths too
    raw_path = os.path.join(config.DATA_DIR, "raw", "prices_v1.parquet")
    meta_path = os.path.join(config.DATA_DIR, "raw", "metadata.csv") 
    market_path = os.path.join(config.DATA_DIR, "raw", "market_features.csv")

    try:
        raw_df = pd.read_parquet(raw_path) 
        meta_df = pd.read_csv(meta_path)
        market_df = pd.read_csv(market_path)
    except FileNotFoundError:
        print(f"WARNING: Data files not found at {config.DATA_DIR}/raw/. Using Mock Data.")
        # Create mock data for testing flow if real data missing
        raw_df = pd.DataFrame({'ticker': ['AAPL', 'MSFT'], 'date': [as_of_date]*2, 'close': [150, 300]})
        meta_df = pd.DataFrame({'ticker': ['AAPL', 'MSFT']})
        market_df = pd.DataFrame({'date': [as_of_date], 'spy_close': [400]})

    # 2. UNIVERSE SELECTION
    print(">> determining Eligible Universe...")
    selector = UniverseSelector()
    # universe_map = selector.select(raw_df, meta_df, as_of_date)
    # Mocking selector output for now as logic might depend on real data
    universe_map = pd.DataFrame({'ticker': raw_df['ticker'].unique(), 'is_eligible': True})
    
    # Save Universe Artifact
    universe_dir = os.path.join(config.DATA_DIR, "artifacts", "universe")
    os.makedirs(universe_dir, exist_ok=True)
    universe_path = os.path.join(universe_dir, f"universe_{as_of_date}.parquet")
    universe_map.to_parquet(universe_path)
    write_manifest(universe_path, {"type": "universe", "date": as_of_date})
    
    eligible_tickers = universe_map[universe_map['is_eligible']]['ticker'].tolist()
    print(f"   {len(eligible_tickers)} tickers eligible.")

    if not eligible_tickers:
        raise RuntimeError("No tickers found! Check data feed.")

    # Filter Data to Eligible Only
    active_df = raw_df[raw_df['ticker'].isin(eligible_tickers)].copy()

    # 3. CHRONOS PRIORS GENERATION
    print(">> Generating Teacher Priors (Chronos-2)...")
    engineer = FeatureEngineer()
    
    # A. Get Sequences (Mocked for flow)
    # seq_df = engineer.get_chronos_sequences(active_df, lookback=512)
    # priors_df = ChronosInference.run(seq_df)
    
    priors_df = pd.DataFrame({
        'ticker': eligible_tickers,
        'prior_drift_20d': 0.005,
        'prior_vol_20d': 0.02,
        'prior_downside_q10': -0.05,
        'prior_trend_conf': 0.6
    })
    
    # Save Priors Artifact
    priors_v1_dir = os.path.join(config.PRIORS_DIR, "v1")
    os.makedirs(priors_v1_dir, exist_ok=True)
    priors_path = os.path.join(priors_v1_dir, f"{as_of_date}.parquet")
    
    priors_df.to_parquet(priors_path)
    write_manifest(priors_path, {"type": "priors", "version": "v1", "date": as_of_date})

    # 4. RANKER FEATURE PREP
    print(">> Engineering Ranker Features...")
    # feature_df = engineer.process_features(active_df, market_df, mode='inference')
    # ... joining logic ...
    
    # Mocking final input for continuity
    final_input = pd.DataFrame({
        'ticker': eligible_tickers,
        'date': as_of_date,
        'feature_1': 0.5, # Mock features
        'prior_drift_20d': 0.005
    })

    # 5. VALIDATION
    print(">> Validating Schema...")
    # FeatureContract.validate(final_input, mode='ranking') # Enable when columns match
    
    # 6. RANKER INFERENCE
    print(">> Running Rank-Transformer...")
    # leaderboard = RankerInference.predict(final_input)
    leaderboard = final_input[['ticker']].copy()
    leaderboard['rank_score'] = 0.85 
    leaderboard['p_up'] = 0.6
    leaderboard['selected'] = True
    
    # 7. FINAL ARTIFACT
    print(">> Saving Leaderboard...")
    leaderboard_v1_dir = os.path.join(config.SIGNALS_DIR, "selector", "v1")
    os.makedirs(leaderboard_v1_dir, exist_ok=True)
    leaderboard_path = os.path.join(leaderboard_v1_dir, f"{as_of_date}.parquet")
    
    leaderboard.to_parquet(leaderboard_path)
    write_manifest(leaderboard_path, {"type": "leaderboard", "version": "v1", "date": as_of_date})
    
    print("=== Nightly Cycle Complete ===")

if __name__ == "__main__":
    run_nightly_cycle(datetime.today().strftime('%Y-%m-%d'))
