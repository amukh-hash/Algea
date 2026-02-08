
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PathSet:
    data_raw: str
    data_canonical: str
    features: str
    priors: str
    datasets: str
    models: str
    calibration: str
    manifests: str
    outputs: str
    logs: str
    legacy_artifacts_root: str = "backend/data/artifacts"

def get_paths() -> PathSet:
    """
    Returns the authoritative PathSet.
    """
    # Assuming run from project root, or flexible?
    # We'll use relative paths from repo root (CWD).
    return PathSet(
        data_raw="backend/data_raw",
        data_canonical="backend/data_canonical",
        features="backend/features",
        priors="backend/priors",
        datasets="backend/datasets",
        models="backend/models",
        calibration="backend/calibration",
        manifests="backend/manifests",
        outputs="backend/outputs",
        logs="backend/logs",
        legacy_artifacts_root="backend/data/artifacts"
    )

def resolve(kind: str, *, legacy_ok: bool = True, **kwargs) -> str:
    """
    Resolves a path based on kind and partition keys.
    kind in {'security_master','ohlcv_adj','covariates','breadth',
             'manifest','featureframe','priors_date','labels','leaderboard',...}
    kwargs provides partition keys (symbol/date/version).
    """
    paths = get_paths()
    
    # 1. B1 Security Master
    if kind == "security_master":
        return os.path.join(paths.data_canonical, "security_master.parquet")
        
    # 2. B2 Canonical Daily Bars (Partitioned)
    elif kind == "ohlcv_adj":
        ticker = kwargs.get("ticker") or kwargs.get("symbol")
        if not ticker:
            raise ValueError("ohlcv_adj requires 'ticker'")
        return os.path.join(paths.data_canonical, "ohlcv_adj", f"ticker={ticker}", "data.parquet")
        
    # 3. B3/B4 Covariates & Breadth
    elif kind == "covariates":
        return os.path.join(paths.data_canonical, "covariates_daily.parquet")
    elif kind == "breadth":
        return os.path.join(paths.data_canonical, "breadth_daily.parquet")
        
    # 4. B5 Universe Manifest
    elif kind == "manifest":
        asof_date = kwargs.get("asof_date") or kwargs.get("date")
        if not asof_date:
            raise ValueError("manifest requires 'asof_date'")
        # Normalize date
        dstr = str(asof_date).split(" ")[0]
        return os.path.join(paths.manifests, f"universe_asof={dstr}.parquet")
        
    # 5. B6 FeatureFrame
    elif kind == "featureframe":
        version = kwargs.get("version", "latest")
        return os.path.join(paths.features, f"featureframe_v{version}.parquet")
        
    # 6. B7 Priors (Date Partitioned)
    elif kind == "priors_date":
        date = kwargs.get("date")
        version = kwargs.get("version", "latest")
        if not date:
            raise ValueError("priors_date requires 'date'")
        dstr = str(date).split(" ")[0]
        return os.path.join(paths.priors, f"date={dstr}", f"priors_v{version}.parquet")
        
    # 7. B8 Labels
    elif kind == "labels":
        horizon = kwargs.get("horizon", 10)
        return os.path.join(paths.datasets, f"labels_fwd{horizon}d.parquet")
        
    # 8. B9 Leaderboard
    elif kind == "leaderboard":
        date = kwargs.get("date")
        if not date:
            raise ValueError("leaderboard requires 'date'")
        dstr = str(date).split(" ")[0]
        return os.path.join(paths.outputs, f"leaderboard_date={dstr}.parquet")
    
    # Dataset tensors (Selector)
    elif kind == "dataset_selector":
        return os.path.join(paths.datasets, "selector_dataset.pt") 

    elif kind == "dataset_selector_groups":
        return os.path.join(paths.datasets, "selector_dataset_groups.parquet")
        
    # Models
    elif kind == "model_selector":
         version = kwargs.get("version", "latest")
         return os.path.join(paths.models, f"rank_tf_{version}.pt")
         
    # Legacy Fallback?
    # If legacy_ok=True and file missing in primary, check legacy?
    # Here we just resolve NEW paths. Legacy logic handled by caller or specific bridge.
    
    else:
        raise ValueError(f"Unknown path kind: {kind}")

def get_gold_daily_root(paths: PathSet = None) -> Path:
    """
    Canonical location for Gold Daily Parquet files (OHLCV).
    Env override: GOLD_DAILY_ROOT
    Default: backend/data_canonical/daily_parquet/
    """
    env = os.getenv("GOLD_DAILY_ROOT")
    if env: return Path(env)
    p = paths or get_paths()
    return Path(p.data_canonical) / "daily_parquet"

def get_universe_frame_root(paths: PathSet = None, version: str = "v2") -> Path:
    """
    Canonical location for UniverseFrame artifacts (Hive partitioned).
    Env override: UNIVERSE_FRAME_ROOT
    Default: backend/data_canonical/universe/universe_frame_v2/
    """
    env = os.getenv("UNIVERSE_FRAME_ROOT")
    if env: return Path(env)
    p = paths or get_paths()
    return Path(p.data_canonical) / "universe" / f"universe_frame_{version}"

def get_selector_features_root(paths: PathSet = None, version: str = "v2", horizon: Optional[str] = None) -> Path:
    """
    Canonical location for SelectorFeatureFrame artifacts (Hive partitioned).
    Env override: SELECTOR_FEATURES_ROOT
    Default: backend/data_canonical/features/selector_features_v2/
    If horizon is specified, appends /horizon={horizon} (though implementation plan distincts this?)
    Actually plan says: default: backend/data_canonical/features/selector_features_v2
    if horizon provided, append f"horizon={horizon}"?
    Let's follow plan:
    """
    env = os.getenv("SELECTOR_FEATURES_ROOT")
    if env: 
        root = Path(env)
    else:
        p = paths or get_paths()
        root = Path(p.data_canonical) / "features" / f"selector_features_{version}"
    
    if horizon:
        return root / f"horizon={horizon}"
    return root

def get_silver_daily_root(paths: PathSet = None) -> Path:
    """
    Canonical location for Silver MarketFrame files (daily aligned).
    Env override: SILVER_DAILY_ROOT
    Default: backend/data_canonical/marketframe_daily/
    """
    env = os.getenv("SILVER_DAILY_ROOT")
    if env: return Path(env)
    p = paths or get_paths()
    return Path(p.data_canonical) / "marketframe_daily"

def get_priors_root(paths: PathSet = None) -> Path:
    """
    Canonical location for Priors artifacts.
    Env override: PRIORS_ROOT
    Default: backend/priors/
    """
    env = os.getenv("PRIORS_ROOT")
    if env: return Path(env)
    p = paths or get_paths()
    return Path(p.priors)
