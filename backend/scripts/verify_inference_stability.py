"""
Verification script for Chronos-2 Teacher Inference Stability.
Target: Validate NLL-trained models (Teacher-10d, Teacher-30d) produce stable priors.

Approach: Load pipeline -> inject LoRA -> load checkpoint -> use pipeline.predict directly.
pipeline.predict uses pipeline.model internally, which is the same object we modified.

Usage:
    python backend/scripts/verify_inference_stability.py RUN-2026-02-09-175844 RUN-2026-02-09-181337
"""
import argparse
import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.scripts._cli_utils import detect_ffn_modules

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_inference")


# -------------------------------------------------------------------------
# 1. Model Loading — Direct Pipeline Approach
# -------------------------------------------------------------------------
def load_production_model(run_dir: Path):
    """
    Load trained Chronos-2 model via pipeline + LoRA checkpoint.
    Returns (pipeline, device, config) — use pipeline.predict() for inference.
    """
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "best" / "model.pt"

    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config or checkpoint in {run_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model_id = config.get("model_id", "amazon/chronos-2")
    from algaie.core.device import get_device
    device = get_device()

    logger.info(f"Loading {model_id} pipeline...")

    # 1. Load Pipeline (base model)
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        model_id, device_map=None, dtype=torch.float32
    )

    # 2. Inject LoRA into pipeline.model (in-place modification)
    from peft import LoraConfig, inject_adapter_in_model

    target_modules = ["q", "v", "k", "o"]
    if config.get("lora_ffn", False):
        target_modules += detect_ffn_modules(pipeline.model)

    peft_cfg = LoraConfig(
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=target_modules,
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
    )

    inject_adapter_in_model(peft_cfg, pipeline.model)

    # 3. Load checkpoint weights into pipeline.model
    logger.info(f"Loading weights from {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)...")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

    incompatible = pipeline.model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.warning(f"Missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:3]}...")
    if incompatible.unexpected_keys:
        logger.warning(f"Unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:3]}...")

    n_lora = sum(1 for k in state_dict if "lora_" in k)
    logger.info(f"Loaded {n_lora} LoRA parameters out of {len(state_dict)} total keys.")

    pipeline.model.eval()
    pipeline.model.to(device)

    return pipeline, device, config


# -------------------------------------------------------------------------
# 2. Compute Priors from Sampled Forecasts
# -------------------------------------------------------------------------
def compute_priors_from_forecasts(forecasts, current_price):
    """
    forecasts: tensor [B, n_samples, prediction_length] or similar
    current_price: scalar or [B] tensor
    Returns dict of prior features per batch element.
    """
    # forecasts from pipeline.predict returns [B, S, P] typically
    if isinstance(forecasts, list):
        forecasts = torch.stack(forecasts)
    forecasts = torch.as_tensor(forecasts, dtype=torch.float32)

    # Handle various output shapes
    if forecasts.ndim == 4:
        forecasts = forecasts.squeeze(-1)  # [B, S, P, 1] -> [B, S, P]

    # Terminal returns: (forecast_terminal / current_price) - 1
    terminal_forecasts = forecasts[:, :, -1]  # [B, S] — last time step
    cum_returns = (terminal_forecasts / current_price.unsqueeze(1)) - 1.0

    results = []
    for b in range(cum_returns.shape[0]):
        r = cum_returns[b]
        results.append({
            "drift": float(torch.median(r)),
            "vol_forecast": float(torch.std(r)),
            "tail_risk": float(torch.quantile(r, 0.10)),
            "prob_up": float((r > 0).float().mean()),
            "q10": float(torch.quantile(r, 0.10)),
            "q50": float(torch.quantile(r, 0.50)),
            "q90": float(torch.quantile(r, 0.90)),
            "dispersion": float(torch.quantile(r, 0.90) - torch.quantile(r, 0.10)),
        })
    return results


# -------------------------------------------------------------------------
# 3. Verification Logic
# -------------------------------------------------------------------------
def _run_trials(pipeline, rw_tensor, prediction_len, current_price, n_trials, fixed_seed=None):
    """Run n_trials forecast trials, optionally with a fixed seed each time."""
    results = []
    for i in range(n_trials):
        if fixed_seed is not None:
            torch.manual_seed(fixed_seed)
            np.random.seed(fixed_seed)
        with torch.no_grad():
            forecasts = pipeline.predict(
                rw_tensor, prediction_length=prediction_len,
                limit_prediction_length=False,
            )
        priors = compute_priors_from_forecasts(forecasts, current_price)
        results.append(priors[0])
        if fixed_seed is None:
            logger.info(f"  Trial {i}: drift={priors[0]['drift']:.4f}, vol={priors[0]['vol_forecast']:.4f}")
    return results


REQUIRED_PRIOR_KEYS = {"drift", "vol_forecast", "tail_risk", "prob_up", "q10", "q50", "q90", "dispersion"}


def verify_stability(run_name: str, run_dir: Path, n_trials=5):
    logger.info(f"=== Verifying {run_name} ===")
    pipeline, device, config = load_production_model(run_dir)

    context_len = config["context_len"]
    prediction_len = config["prediction_len"]

    # Generate a random walk as test input
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(context_len)) + 100.0
    rw_tensor = torch.tensor(rw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    current_price = torch.tensor([rw[-1]], dtype=torch.float32)

    logger.info(f"Input shape: {rw_tensor.shape}, prediction_len: {prediction_len}")

    # A. Stochastic stability (no seed control)
    logger.info(f"Running {n_trials} stochastic trials...")
    stochastic_results = _run_trials(pipeline, rw_tensor, prediction_len, current_price, n_trials)

    drifts = [r["drift"] for r in stochastic_results]
    vols = [r["vol_forecast"] for r in stochastic_results]
    logger.info(f"Stochastic Stability:")
    logger.info(f"  Drift: {np.mean(drifts):.4f} +/- {np.std(drifts):.4f}")
    logger.info(f"  Vol:   {np.mean(vols):.4f} +/- {np.std(vols):.4f}")

    # B. Fixed-seed determinism
    logger.info(f"Running {n_trials} fixed-seed trials...")
    fixed_results = _run_trials(pipeline, rw_tensor, prediction_len, current_price, n_trials, fixed_seed=12345)

    fixed_drifts = [r["drift"] for r in fixed_results]
    logger.info(f"Fixed-Seed Stability: drift StdDev = {np.std(fixed_drifts):.6f}")

    # C. Output Schema Check
    example = stochastic_results[0]
    missing = REQUIRED_PRIOR_KEYS - set(example.keys())
    if missing:
        logger.error(f"  MISSING KEYS: {missing}")
    else:
        logger.info("  All required keys present")

    logger.info(f"Example output: {example}")
    logger.info("=" * 60)

    return np.std(drifts) < 0.05  # Pass if stochastic variance is reasonable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify inference stability of trained Chronos-2 models")
    parser.add_argument("run_ids", nargs="+", help="Run IDs to verify (e.g. RUN-2026-02-09-175844)")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per run")
    args = parser.parse_args()

    runs_dir = ROOT / "backend" / "data" / "runs"
    verdicts = {}
    for run_id in args.run_ids:
        run_dir = runs_dir / run_id
        ok = verify_stability(run_id, run_dir, n_trials=args.trials)
        verdicts[run_id] = "PASS" if ok else "FAIL"

    logger.info("=" * 60)
    verdict_str = ", ".join(f"{k}={v}" for k, v in verdicts.items())
    logger.info(f"VERDICT: {verdict_str}")
