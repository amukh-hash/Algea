
from typing import Dict
from backend.app.training.trainer import train_model

def train_selector(run_cfg: Dict) -> Dict:
    print(f"Wrapper: train_selector({run_cfg})")
    # Delegate to legacy trainer if needed, or stub
    return {}
