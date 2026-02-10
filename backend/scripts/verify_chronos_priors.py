"""
Integration verification: Chronos-2 Teacher Inference Stability & Validation.

Loads real Teacher-10d and Teacher-30d checkpoints, runs infer_priors via the
hardened API, and asserts determinism + schema + contract validation.
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_priors")


# -------------------------------------------------------------------------
# 1. Model Loading — Direct Pipeline + LoRA
# -------------------------------------------------------------------------
def load_teacher(run_dir: Path):
    """Load trained Chronos-2 model and wrap for infer_priors."""
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "best" / "model.pt"

    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config or checkpoint in {run_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model_id = config.get("model_id", "amazon/chronos-2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=None, dtype=torch.float32)

    from peft import LoraConfig, inject_adapter_in_model

    def _detect_ffn_modules(m):
        candidates = {"wi", "wo", "wi_0", "wi_1", "dense", "fc1", "fc2", "mlp"}
        found = set()
        for name, mod in m.named_modules():
            if hasattr(mod, "weight"):
                short = name.split(".")[-1]
                if short in candidates or any(c in short for c in candidates):
                    found.add(short)
        return sorted(found)

    target_modules = ["q", "v", "k", "o"]
    if config.get("lora_ffn", False):
        target_modules += _detect_ffn_modules(pipeline.model)

    peft_cfg = LoraConfig(
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=target_modules,
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
    )
    inject_adapter_in_model(peft_cfg, pipeline.model)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    pipeline.model.load_state_dict(state_dict, strict=False)
    pipeline.model.eval().to(device)

    # Wrap for infer_priors
    from algaie.models.foundation.chronos2_teacher import Chronos2NativeWrapper
    model = pipeline.model
    model.predict = pipeline.predict
    native = Chronos2NativeWrapper(model, model_type="chronos-2",
                                    forward_params=["context", "future_target"])
    return native, pipeline, device, config


# -------------------------------------------------------------------------
# 2. Verification
# -------------------------------------------------------------------------
def verify_teacher(name: str, run_dir: Path) -> bool:
    from algaie.models.foundation.chronos2_teacher import infer_priors, ChronosPriors

    logger.info(f"=== {name} ===")
    model, pipeline, device, config = load_teacher(run_dir)
    ctx_len = config["context_len"]
    pred_len = config["prediction_len"]

    np.random.seed(42)
    rw = np.cumsum(np.random.randn(ctx_len)) + 100.0
    input_tensor = torch.tensor(rw, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]

    # A. Determinism check — two calls must be identical
    priors_a = infer_priors(model, input_tensor, horizon_days=pred_len, n_samples=20, mode="native_nll")
    priors_b = infer_priors(model, input_tensor, horizon_days=pred_len, n_samples=20, mode="native_nll")
    assert len(priors_a) == len(priors_b) == 1
    a, b = priors_a[0], priors_b[0]
    assert a.drift == b.drift, f"Drift mismatch: {a.drift} vs {b.drift}"
    assert a.q10 == b.q10, f"q10 mismatch: {a.q10} vs {b.q10}"
    assert a.q90 == b.q90, f"q90 mismatch: {a.q90} vs {b.q90}"
    logger.info(f"  ✓ Determinism: identical outputs on repeated calls")

    # B. Schema check — all required keys
    required = {"drift", "vol_forecast", "tail_risk", "trend_conf", "metadata",
                 "q10", "q50", "q90", "dispersion", "prob_up"}
    actual = set(a.__dataclass_fields__.keys())
    missing = required - actual
    assert not missing, f"Missing schema keys: {missing}"
    logger.info(f"  ✓ Schema: all {len(required)} keys present")

    # C. Validation — contract invariants hold
    a.validate(strict=True)  # Should not raise
    logger.info(f"  ✓ Validation: strict mode passed")

    # D. Monotonic quantiles
    assert a.q10 <= a.q50 <= a.q90, f"Quantile order: {a.q10}, {a.q50}, {a.q90}"
    logger.info(f"  ✓ Quantiles: q10={a.q10:.4f} ≤ q50={a.q50:.4f} ≤ q90={a.q90:.4f}")

    # E. Dispersion non-negative
    assert a.dispersion >= 0, f"Negative dispersion: {a.dispersion}"
    logger.info(f"  ✓ Dispersion: {a.dispersion:.4f} ≥ 0")

    # F. prob_up in [0, 1]
    assert 0.0 <= a.prob_up <= 1.0, f"prob_up out of range: {a.prob_up}"
    logger.info(f"  ✓ prob_up: {a.prob_up:.4f} ∈ [0, 1]")

    # G. Metadata mode tag
    assert a.metadata.get("mode") == "native_nll"
    logger.info(f"  ✓ Metadata: mode={a.metadata['mode']}, horizon={a.metadata['horizon']}")

    # H. Backward compat: trend_conf == prob_up
    assert a.trend_conf == a.prob_up
    logger.info(f"  ✓ Backward compat: trend_conf == prob_up")

    logger.info(f"  OUTPUT: drift={a.drift:.4f}, vol={a.vol_forecast:.4f}, "
                f"q10={a.q10:.4f}, q50={a.q50:.4f}, q90={a.q90:.4f}, "
                f"disp={a.dispersion:.4f}, prob_up={a.prob_up:.2f}")
    return True


if __name__ == "__main__":
    runs_dir = ROOT / "backend" / "data" / "runs"
    ok_10d = verify_teacher("Teacher-10d", runs_dir / "RUN-2026-02-09-175844")
    ok_30d = verify_teacher("Teacher-30d", runs_dir / "RUN-2026-02-09-181337")
    verdict = "PASS" if (ok_10d and ok_30d) else "FAIL"
    logger.info(f"\nVERDICT: {verdict}")
