import os
import json
import pandas as pd
from datetime import datetime
from backend.app.core import config
from backend.app.ops import pathmap
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
    # Using pathmap for consistent paths if possible, or config.DATA_DIR for raw
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
        raw_df = pd.DataFrame({'symbol': ['AAPL', 'MSFT'], 'date': [as_of_date]*2, 'close': [150, 300]})
        meta_df = pd.DataFrame({'symbol': ['AAPL', 'MSFT']})
        market_df = pd.DataFrame({'date': [as_of_date], 'spy_close': [400]})

    # 2. UNIVERSE SELECTION
    print(">> determining Eligible Universe...")
    selector = UniverseSelector()
    # universe_map = selector.select(raw_df, meta_df, as_of_date)
    # Mocking selector output for now as logic might depend on real data
    universe_map = pd.DataFrame({'symbol': raw_df['symbol'].unique(), 'is_eligible': True})
    
    # Save Universe Artifact
    universe_path = pathmap.resolve("manifest", date=as_of_date)
    os.makedirs(os.path.dirname(universe_path), exist_ok=True)
    universe_map.to_parquet(universe_path)
    write_manifest(universe_path, {"type": "universe", "date": as_of_date})
    
    eligible_symbols = universe_map[universe_map['is_eligible']]['symbol'].tolist()
    print(f"   {len(eligible_symbols)} symbols eligible.")

    if not eligible_symbols:
        raise RuntimeError("No symbols found! Check data feed.")

    # Filter Data to Eligible Only
    active_df = raw_df[raw_df['symbol'].isin(eligible_symbols)].copy()

    # 3. CHRONOS PRIORS GENERATION
    print(">> Generating Teacher Priors (Chronos-2)...")
    # In real flow: engineer.get_chronos_sequences -> ChronosInference.run
    # For now, we construct the DF manually but adhering to Schema B7
    priors_df = pd.DataFrame({
        'symbol': eligible_symbols,
        'date': pd.to_datetime(as_of_date),
        'prior_drift_20d': 0.005,
        'prior_vol_20d': 0.02,
        'prior_downside_q10_20d': -0.05,
        'prior_trend_conf_20d': 0.6,
        'prior_version': "v1",
        'chronos_model_id': "stub-v1",
        'context_len': 512,
        'horizon': 20
    })
    
    # Save Priors Artifact
    priors_path = pathmap.resolve("priors_date", date=as_of_date, version="v1")
    os.makedirs(os.path.dirname(priors_path), exist_ok=True)
    
    priors_df.to_parquet(priors_path)
    write_manifest(priors_path, {"type": "priors", "version": "v1", "date": as_of_date})

    # 4. RANKER FEATURE PREP
    print(">> Engineering Ranker Features...")
    from backend.app.features import featureframe
    
    # Build FeatureFrame (B6)
    # This mocks the heavy lifting of joining OHLCV + Covariates + Computing Features
    # and returns a valid B6-schema dataframe.
    ff_df = featureframe.build_featureframe(as_of_date, as_of_date, {"symbols": eligible_symbols})
    
    # Join with Priors and Market Features? 
    # FeatureContract expects: CORE (B6) + MARKET + PRIORS
    # FF provides CORE + MARKET (spy_ret, vix).
    # We just need to join Priors.
    
    # Ensure date is matching for merge (ff_df date might be datetime64, priors date datetime64)
    # Merge keys: date, symbol
    final_input = ff_df.merge(priors_df[['symbol', 'date', 'prior_drift_20d', 'prior_vol_20d', 'prior_downside_q10_20d', 'prior_trend_conf_20d']], 
                              on=['symbol', 'date'], how='inner')
    
    # Rename priors columns to match FeatureContract if they differ? 
    # Contract: prior_drift_20d (Checked in Schema Alignment). 
    # DF: prior_drift_20d (Set above). 
    # Match.

    # Rename B6 columns to Contract if needed?
    # Contract: ret_1d, vol_20d, volume_z_20d (Updated in Schema Alignment).
    # FF (B6): ret_1d, vol_20d, volume_z_20d (Updated in builds).
    # Match.

    # 5. VALIDATION
    print(">> Validating Schema...")
    FeatureContract.validate(final_input, mode='ranking')
    
    # 6. RANKER INFERENCE
    print(">> Running Rank-Transformer...")
    # leaderboard = RankerInference.predict(final_input)
    leaderboard = final_input[['symbol']].copy()
    leaderboard['rank_score'] = 0.85 
    leaderboard['p_up'] = 0.6
    leaderboard['selected'] = True
    
    # 7. FINAL ARTIFACT
    print(">> Saving Leaderboard...")
    leaderboard_path = pathmap.resolve("leaderboard", date=as_of_date)
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
    
    leaderboard.to_parquet(leaderboard_path)
    write_manifest(leaderboard_path, {"type": "leaderboard", "version": "v1", "date": as_of_date})
    
    print("=== Nightly Cycle Complete ===")

if __name__ == "__main__":
    run_nightly_cycle(datetime.today().strftime('%Y-%m-%d'))
