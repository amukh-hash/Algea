import pandas as pd
from datetime import datetime
from backend.app.data.universe import UniverseSelector
from backend.app.data.preproc import FeatureEngineer
from backend.app.models.schema import FeatureContract

# Mock Imports for the Models (You'd replace these with actual Model Runners)
# from backend.app.models.chronos import ChronosInference
# from backend.app.models.ranker import RankerInference

def run_nightly_cycle(as_of_date: str):
    print(f"=== Starting ALGAIE Nightly Cycle for {as_of_date} ===")
    
    # 1. LOAD RAW DATA
    # Assume we load a large window (e.g. 2 years) to ensure full history for rolling stats + Chronos
    print(">> Loading Raw Data...")
    raw_df = pd.read_parquet("data/raw/prices_v1.parquet") 
    meta_df = pd.read_csv("data/raw/metadata.csv")
    market_df = pd.read_csv("data/raw/market_features.csv") # Already pre-calculated SPY/VIX

    # 2. UNIVERSE SELECTION (Dynamic)
    print(">> determining Eligible Universe...")
    selector = UniverseSelector()
    universe_map = selector.select(raw_df, meta_df, as_of_date)
    
    # Save Membership Artifact
    universe_map.to_parquet(f"data/artifacts/universe/universe_{as_of_date}.parquet")
    
    eligible_tickers = universe_map[universe_map['is_eligible']]['ticker'].tolist()
    print(f"   {len(eligible_tickers)} tickers eligible.")

    if not eligible_tickers:
        raise RuntimeError("No tickers found! Check data feed.")

    # Filter Data to Eligible Only (Optimization)
    active_df = raw_df[raw_df['ticker'].isin(eligible_tickers)].copy()

    # 3. CHRONOS PRIORS GENERATION
    print(">> Generating Teacher Priors (Chronos-2)...")
    engineer = FeatureEngineer()
    
    # A. Get Sequences (Last 512 days)
    seq_df = engineer.get_chronos_sequences(active_df, lookback=512)
    
    # B. Run Inference (Mock)
    # priors_df = ChronosInference.run(seq_df)
    # Mocking output for structure demonstration:
    priors_df = pd.DataFrame({
        'ticker': seq_df['ticker'],
        'prior_drift_20d': 0.005,
        'prior_vol_20d': 0.02,
        'prior_downside_q10': -0.05,
        'prior_trend_conf': 0.6
    })
    
    # Save Priors Artifact
    priors_df.to_parquet(f"data/artifacts/priors/priors_{as_of_date}.parquet")

    # 4. RANKER FEATURE PREP
    print(">> Engineering Ranker Features...")
    # We need the last ~60 days of features for the Ranker's internal lookback
    feature_df = engineer.process_features(active_df, market_df, mode='inference')
    
    # Isolate the relevant window (e.g. last 60 days per ticker)
    # For simplicity, let's assume the Ranker just needs TODAY's features + History is internal
    # But usually, we save the last batch.
    
    # Align Priors to Features (Join on Ticker)
    # Note: Priors are "As Of Today", so we map them to the latest date rows
    today_features = feature_df[feature_df['date'] == as_of_date].copy()
    
    final_input = today_features.merge(priors_df, on='ticker', how='left')

    # 5. VALIDATION
    print(">> Validating Schema...")
    FeatureContract.validate(final_input, mode='ranking')
    
    # 6. RANKER INFERENCE
    print(">> Running Rank-Transformer...")
    # leaderboard = RankerInference.predict(final_input)
    # Mock output
    leaderboard = final_input[['ticker']].copy()
    leaderboard['raw_score'] = 0.85 # Mock
    leaderboard['rank_pct'] = leaderboard['raw_score'].rank(pct=True)
    leaderboard['ev_10d'] = 0.02
    
    # 7. FINAL ARTIFACT
    print(">> Saving Leaderboard...")
    leaderboard.to_parquet(f"data/artifacts/leaderboard/leaderboard_{as_of_date}.parquet")
    
    print("=== Nightly Cycle Complete ===")

if __name__ == "__main__":
    run_nightly_cycle("2026-02-04")
