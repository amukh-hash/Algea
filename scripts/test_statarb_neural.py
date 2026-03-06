"""Direct test of StatArb V3 neural overlay — bypasses orchestrator idempotency."""
import sys, json, logging, os
from pathlib import Path
from datetime import date, datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Build a minimal context matching what orchestrator.run_once() provides
day_root = Path("backend/artifacts/orchestrator") / date.today().isoformat()
day_root.mkdir(parents=True, exist_ok=True)

# Mock the config that _statarb_enabled checks
class MockConfig:
    mode = "paper"

ctx = {
    "asof_date": date.today().isoformat(),
    "session": "premarket",
    "artifact_root": str(day_root),
    "mode": "paper",
    "tick_id": "direct-test",
    "dry_run": False,
    "config": MockConfig(),
}

from backend.app.orchestrator.job_defs import handle_signals_generate_statarb

print("\n" + "=" * 60)
print("STATARB V3 NEURAL OVERLAY — DIRECT TEST")
print("=" * 60)

result = handle_signals_generate_statarb(ctx)

print("\n" + "=" * 60)
print("RESULT:")
print(json.dumps(result, indent=2, default=str))
print("=" * 60)

# Show the signal file
sig_str = result.get("artifacts", {}).get("statarb_signals", "")
if sig_str:
    sig_path = Path(sig_str)
    if sig_path.exists():
        signals = json.loads(sig_path.read_text(encoding="utf-8"))
        print("\n=== CONFIRMED PAIRS ===")
        for cp in signals.get("confirmed_pairs", []):
            print(f"  {cp['direction'].upper():5s}  {cp['pair']:12s}  Z={cp['z']:.2f}  Pred={cp['pred']:.3f}  Conv={cp['conviction']:.3f}")

        if not signals.get("confirmed_pairs"):
            print("  (none)")

        print("\n=== VETOED PAIRS (no edge) ===")
        for sig in signals.get("signals", []):
            pair = sig["pair"]
            if not any(cp["pair"] == pair for cp in signals.get("confirmed_pairs", [])):
                print(f"  {pair:12s}  Z={sig['z_score']:.2f}  Pred={sig['pred_delta']:.3f}")

        print("\n=== BETA-NEUTRAL WEIGHTS ===")
        weights = signals.get("beta_neutral_weights", {})
        if weights:
            for sym, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
                bar = "█" * int(abs(w) * 100)
                sign = "+" if w > 0 else "-"
                print(f"  {sym:5s}  {w:+.4f}  {sign}{bar}")
        else:
            print("  (flat — no pairs passed alpha gate)")

        print(f"\n  Ensemble loaded: {signals.get('ensemble_loaded')}")
        print(f"  Folds used:     {signals.get('n_folds')}")
        if weights:
            print(f"  Sum weights:    {sum(weights.values()):.6f}")
            print(f"  Gross exposure: {sum(abs(v) for v in weights.values()):.4f}")
else:
    print("No signal path in result — handler returned:", result)
