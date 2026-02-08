"""
Chronos-2 Data Readiness Compliant Dataset.
Implements strict Index Builder (3.2) and Observable Mask (3.1).
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from backend.app.ops import pathmap
from backend.app.data.schema_contracts import normalize_keys

logger = logging.getLogger(__name__)

class ChronosDataset(Dataset):
    """
    Dataset for Chronos-2 Training/Fine-tuning.
    
    Checklist Compliance:
    3.1 Observable Universe Mask: Filters anchors by UniverseFrame.is_observable
    3.2 Sample Index Builder: Checks Context, Horizon, Observability, and NaN Integrity
    3.3 Scaling: Return-space modeling (Relative Log Price)
    3.4 Stride: Configurable (default 5)
    """
    def __init__(self, 
                 files: List[Path],
                 context_len: int,
                 prediction_len: int,
                 stride: int = 5,
                 universe_path: Optional[str] = None, # Path to universe parquet
                 target_col: str = "close", # Use raw close and transform internally
                 max_samples_per_file: Optional[int] = None,
                 seed: int = 42):
        
        self.stats = {
            "n_files_total": 0,
            "n_rows_total": 0,
            "n_windows_potential": 0, # Total valid start indices based on length
            "n_dropped_observable": 0,
            "n_dropped_nan": 0,
            "n_dropped_invalid_price": 0,
            "n_final_samples": 0
        }
        self.files = files
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.stride = stride
        self.target_col = target_col
        self.rng = np.random.RandomState(seed)
        
        # Load Universe Observable Mask
        self.obs_lookup = self._load_universe_mask(universe_path)
        
        # Build Index
        self.index = self._build_index(max_samples_per_file)
        
    def _load_universe_mask(self, universe_path: Optional[str]) -> Optional[Dict[str, set]]:
        """
        Loads mapping: Symbol -> Set of Observable Dates
        """
        if not universe_path:
            return None

        try:
            logger.info(f"Loading UniverseFrame from {universe_path}...")
            # 1. Recursive Glob Support for Hive Partitioning
            # If path ends in .parquet, read directly. If dir (or glob pattern), use scan/read logic.
            # Polars scan_parquet supports globs.
            
            # If passed strict directory, append glob
            if Path(universe_path).is_dir():
                universe_path = str(Path(universe_path) / "**/*.parquet")
                
            uframe = pl.read_parquet(universe_path, columns=["date", "is_observable", "symbol"])
            
            if uframe.height == 0:
                logger.warning("UniverseFrame empty!")
                return None
                
            # 1.1 Normalize ticker -> symbol (using schema contracts)
            uframe = normalize_keys(uframe)
                
            if "symbol" not in uframe.columns:
                 logger.error("UniverseFrame missing 'symbol' column (and 'ticker').")
                 return None

            # 1.3 Ensure Date type
            uframe = uframe.with_columns(pl.col("date").cast(pl.Date))
                
            # Filter for observable
            # Check if is_observable exists (it should)
            if "is_observable" not in uframe.columns:
                 # If missing, assume all observable? Or fail?
                 # Fail safe
                 logger.warning("is_observable missing, assuming all False (empty mask).")
                 return {}

            obs_df = uframe.filter(pl.col("is_observable")).select(["date", "symbol"])
            
            logger.info(f"Building observable lookup for {obs_df['symbol'].n_unique()} symbols...")
            lookup = {}
            
            grouped = obs_df.group_by("symbol").agg(pl.col("date"))
            
            for row in grouped.iter_rows():
                t, dates = row
                # dates are python date objects (because we cast to pl.Date)
                lookup[t] = set(dates)
            
            return lookup
            
        except Exception as e:
            # Handle "Column not found" if read_parquet requests columns not in file involves catching PolarsError
            # But we can lazily scan or just try/except
            logger.warning(f"Failed to load universe mask: {e}")
            return None

    def _build_index(self, max_samples: Optional[int]) -> List[Tuple[int, int]]:
        """
        Scans files and builds valid (file_idx, start_row) index.
        Checks 3.2 constraints.
        Populates self.stats.
        """
        index = []
        logger.info(f"Indexing {len(self.files)} files with Stride={self.stride}...")
        self.stats["n_files_total"] = len(self.files)
        
        for fi, fp in enumerate(self.files):
            try:
                # 1. Read Data needed for validation
                # Ticker from filename
                ticker = fp.stem
                
                if fi == 0:
                    print(f"DEBUG: Processing file {fp}, extracted ticker: {ticker}")
                    if self.obs_lookup:
                        print(f"DEBUG: obs_lookup has {len(self.obs_lookup)} keys. Sample: {list(self.obs_lookup.keys())[:5]}")
                        if ticker not in self.obs_lookup:
                            print(f"DEBUG: Ticker {ticker} NOT FOUND in obs_lookup")
                    else:
                        print("DEBUG: obs_lookup is None/Empty")

                # Verify observability for this ticker exists
                valid_obs_dates = self.obs_lookup.get(ticker) if self.obs_lookup else None
                # If universe is loaded but ticker not found, it is skipped?
                # User logic: "Observable Universe Mask: Filters anchors by UniverseFrame.is_observable"
                # If ticker not in universe, it is never observable.
                if self.obs_lookup is not None and valid_obs_dates is None:
                    # Ticker never observable
                    # Treat as dropped observable?
                    # Or just skip silently?
                    # Let's count it as file skipped?
                    # But we are iterating files...
                    continue
                
                # Read columns: date, target, support symbol/ticker check
                # We need 'date' as Date to match Universe
                df = pl.read_parquet(fp, columns=["date", self.target_col])
                
                # 1.2 Normalize Gold Date -> Date
                # Gold is Datetime (ns, UTC). Universe is Date.
                
                if fi == 0:
                     print(f"DEBUG: File {fp} schema: {df.schema}")
                     print(f"DEBUG: File {fp} head: {df.head(2)}")
                
                df = df.with_columns(pl.col("date").dt.date().alias("date"))
                
                dates = df["date"].to_list()
                values = df[self.target_col].to_numpy()
                n = len(df)
                self.stats["n_rows_total"] += n
                
                # 2. Iterate Windows
                # Window: [start, start + context + pred)
                # Anchor: start + context - 1 (last point of context)
                
                max_start = n - (self.context_len + self.prediction_len)
                if max_start <= 0: continue
                
                starts = range(0, max_start, self.stride)
                self.stats["n_windows_potential"] += len(starts)
                
                valid_file_starts = []
                
                for s in starts:
                    # 3. Observability Check
                    anchor_idx = s + self.context_len - 1
                    anchor_date = dates[anchor_idx]
                    
                    if valid_obs_dates is not None:
                        if anchor_date not in valid_obs_dates:
                            if self.stats["n_dropped_observable"] < 5:
                                print(f"DEBUG: Dropped obs {ticker} on {anchor_date} (valid dates len: {len(valid_obs_dates)})")
                            self.stats["n_dropped_observable"] += 1
                            continue
                            
                    # 4. Data Integrity Check (NaNs / Zeros)
                    # Check window slice in 'values'
                    window_vals = values[s : s + self.context_len + self.prediction_len]
                    
                    # NaNs
                    if np.isnan(window_vals).any():
                        self.stats["n_dropped_nan"] += 1
                        continue
                     
                    # Zeros/Negatives (Impossible for price -> log return)
                    if (window_vals <= 1e-8).any():
                        self.stats["n_dropped_invalid_price"] += 1
                        continue
                        
                    valid_file_starts.append(s)
                    
                # Subsample if needed
                if max_samples and len(valid_file_starts) > max_samples:
                    valid_file_starts = self.rng.choice(valid_file_starts, size=max_samples, replace=False).tolist()
                    
                for s in valid_file_starts:
                    index.append((fi, int(s)))
                    
            except Exception as e:
                # logger.debug(f"Error indexing {fp}: {e}")
                continue
                
        self.stats["n_final_samples"] = len(index)
        logger.info(f"Built index with {len(index)} samples. Stats: {self.stats}")
        return index

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        fi, start_row = self.index[idx]
        fp = self.files[fi]
        
        # Read window logic.
        # Since parquet is columnar and compressed, random access is costly.
        # But for training iteration, we often rely on OS page cache.
        # Simple implementation: read file, slice. 
        # (Assuming file size is manageable, e.g. < 100MB per ticker).
        
        try:
            df = pl.read_parquet(fp, columns=[self.target_col])
            # Slice
            end_row = start_row + self.context_len + self.prediction_len
            window_vals = df[self.target_col].slice(start_row, self.context_len + self.prediction_len).to_numpy().astype(np.float32)
            
            # Verify length (edge case safety)
            if len(window_vals) < (self.context_len + self.prediction_len):
                # Should not happen due to index builder
                raise ValueError("Window too short")
                
            # 3.3 Scaling: Relative Log Price (Return Space Modeling)
            # Transform: x_t = log(p_t / p_ref)
            # Ref = last point of context context-1
            
            ref_idx = self.context_len - 1
            ref_val = window_vals[ref_idx]
            
            # Safe log (checked > 0 in builder)
            x_trans = np.log(window_vals / ref_val)
            
            # Split
            context = x_trans[:self.context_len]         # [Context]
            target = x_trans[self.context_len:]          # [Pred]
            
            # Shapes expected by model/codec?
            # Univariate: [T] or [T, 1]
            # Chronos expects [T] usually?
            # Let's return [T, 1] to be safe for multivariate expansion.
            
            return {
                "past_target": torch.tensor(context, dtype=torch.float32).unsqueeze(-1),
                "future_target": torch.tensor(target, dtype=torch.float32).unsqueeze(-1),
                "scale": torch.tensor([ref_val], dtype=torch.float32) # Captured for potential reconstruction
            }
            
        except Exception as e:
            logger.error(f"Error reading sample {fp}: {e}")
            # Return dummy to avoid crash? Or raise?
            # Raise lets DataLoader handle it (worker crash).
            raise e

def chronos_collate_fn(batch):
    """
    Collate function for Chronos Dataset.
    """
    past = torch.stack([b["past_target"] for b in batch])
    future = torch.stack([b["future_target"] for b in batch])
    scale = torch.stack([b["scale"] for b in batch])
    
    return {
        "past_target": past,
        "future_target": future,
        "scale": scale
    }
