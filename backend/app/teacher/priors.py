
import os
import pandas as pd
import logging
from typing import Dict, List, Callable, Optional
from backend.app.ops import pathmap, artifact_registry, config
from backend.app.features import schemas, validators

logger = logging.getLogger(__name__)

def generate_priors_for_date(
    asof_date, 
    symbols: List[str], 
    prior_spec: Dict,
    # Injected runner to decouple
    runner = None,
    load_fn: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Generates priors for the list of symbols.
    Returns Schema B7.
    """
    if not runner:
        raise ValueError("Chronos Runner instance required")
        
    if load_fn is None:
        raise ValueError("load_fn is required to load series data for priors.")

    df = runner.infer_batch(symbols, asof_date, load_fn=load_fn)
    df["date"] = pd.to_datetime(asof_date)
    df["chronos_model_id"] = runner.model_id
    df["context_len"] = runner.context_len
    df["horizon"] = runner.horizon
    
    return df

def write_priors(asof_date, df: pd.DataFrame, prior_spec: Dict) -> str:
    # Compute Version
    code_version = "v1"
    pv = artifact_registry.compute_prior_version(prior_spec, code_version)
    
    df["prior_version"] = pv
    
    # Path: date=YYYY-MM-DD/priors_v{PV}.parquet
    path = pathmap.resolve("priors_date", date=asof_date, version=pv)
    
    # Validate
    validators.validate_df(df, schemas.SCHEMA_PRIORS, context="Priors Write")
    
    # Check Coverage
    # If partial write? No, failing coverage usually at script level before write.
    
    # Ensure dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    
    # Metadata
    meta = {
        "version": pv,
        "spec": prior_spec,
        "rows": len(df),
        "date": str(asof_date)
    }
    artifact_registry.write_metadata(path + ".meta", meta)
    
    return path

def load_priors(asof_date, prior_version: str = "latest") -> pd.DataFrame:
    # Resolve handling 'latest'? Pathmap resolve handles name construction.
    # If version='latest', we might need to list dir and find newest?
    # For now, assume explicit version or default 'v1' from logic.
    path = pathmap.resolve("priors_date", date=asof_date, version=prior_version)
    return pd.read_parquet(path)
