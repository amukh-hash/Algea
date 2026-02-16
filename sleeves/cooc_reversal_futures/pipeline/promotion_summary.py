"""R3: Promotion Summary assembler.

Collects results from validation gates, R1 (provider invariance), and
R2 (Tier2 calibration) into a single go/no-go document.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sleeves.cooc_reversal_futures.pipeline.types import (
    PromotionSummary,
    ProviderInvarianceReport,
    Tier2CalibrationReport,
    ValidationReport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate classification
# ---------------------------------------------------------------------------

# "Hard" gates: must all pass for PROMOTE
_HARD_GATES = frozenset({
    "data_completeness", "leakage_oracle", "leakage_permutation",
    "leakage_timestamp", "model_sanity", "strategy_polarity",
})

# "Soft" gates: can fail → HOLD (not FAIL)
_SOFT_GATES = frozenset({
    "ic_distribution", "multi_seed_stability", "stress_window",
    "contiguous_oos", "promotion_windows",
})


def _classify_gates(
    validation: ValidationReport,
) -> Tuple[Dict[str, bool], str]:
    """Classify ValidationReport gates and determine decision.

    Returns (integrity_checks dict, decision string).
    """
    integrity: Dict[str, bool] = {}
    hard_fail = False
    soft_fail = False

    for gate in validation.gates:
        integrity[gate.name] = gate.passed
        if not gate.passed:
            if gate.name in _HARD_GATES:
                hard_fail = True
            elif gate.name in _SOFT_GATES:
                soft_fail = True
            else:
                # Unknown gate → treat as soft
                soft_fail = True

    if hard_fail:
        return integrity, "FAIL"
    if soft_fail:
        return integrity, "HOLD"
    return integrity, "PROMOTE"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_promotion_summary(
    validation: ValidationReport,
    *,
    provider_invariance: Optional[ProviderInvarianceReport] = None,
    tier2_calibration: Optional[Tier2CalibrationReport] = None,
    feature_list: Optional[List[str]] = None,
    features_dropped: Optional[List[str]] = None,
    config_hash: str = "",
    data_version_hash: str = "",
    seed_list: Optional[List[int]] = None,
    run_id: str = "",
) -> PromotionSummary:
    """Assemble a promotion summary from all upstream results.

    Decision logic
    --------------
    - **PROMOTE**: all hard gates pass, all soft gates pass,
      provider invariance is consistent (if provided), Tier2 not degraded.
    - **HOLD**: hard gates pass but some soft gates fail or provider/Tier2
      has warnings.
    - **FAIL**: any hard gate fails.
    """
    integrity, decision = _classify_gates(validation)

    # Provider invariance override
    if provider_invariance is not None and not provider_invariance.overall_consistent:
        if decision == "PROMOTE":
            decision = "HOLD"

    # Tier2 gates — extract summary-level metrics
    tier2_gates: Dict[str, Any] = {}
    if tier2_calibration is not None and tier2_calibration.ladder:
        summary = tier2_calibration.ladder.get("summary", {})
        tier2_gates = {
            "tier0_sharpe": summary.get("tier0_sharpe"),
            "tier1_sharpe": summary.get("tier1_sharpe"),
            "tier2_sharpe": summary.get("tier2_sharpe"),
            "cost_erosion_t0_t1": summary.get("cost_erosion_t0_t1"),
            "cost_erosion_t1_t2": summary.get("cost_erosion_t1_t2"),
        }

    return PromotionSummary(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        integrity_checks=integrity,
        tier2_gates=tier2_gates,
        provider_invariance=provider_invariance,
        tier2_calibration=tier2_calibration,
        feature_list_used=tuple(feature_list or []),
        features_dropped=tuple(features_dropped or []),
        config_hash=config_hash,
        seed_list=tuple(seed_list or []),
        data_version_hash=data_version_hash,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_promotion_summary(
    summary: PromotionSummary,
    output_dir: str | Path,
) -> Path:
    """Write promotion summary as JSON and markdown to output_dir.

    Returns the path to the JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "promotion_summary.json"
    json_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # Markdown
    md_lines = [
        f"# Promotion Summary — {summary.run_id}",
        "",
        f"**Decision: {summary.decision}**",
        f"- Timestamp: {summary.timestamp}",
        f"- Config hash: `{summary.config_hash}`",
        f"- Data version: `{summary.data_version_hash}`",
        f"- Seeds: {list(summary.seed_list)}",
        "",
        "## Integrity Checks",
        "",
    ]
    for name, passed in sorted(summary.integrity_checks.items()):
        icon = "✅" if passed else "❌"
        md_lines.append(f"- {icon} {name}")

    if summary.provider_invariance is not None:
        pi = summary.provider_invariance
        icon = "✅" if pi.overall_consistent else "⚠️"
        md_lines.extend([
            "",
            "## Provider Invariance",
            f"- {icon} Overall consistent: {pi.overall_consistent}",
        ])
        if pi.flags:
            for f in pi.flags:
                md_lines.append(f"  - ⚠️ {f}")

    if summary.tier2_gates:
        md_lines.extend([
            "",
            "## Tier2 Gates",
        ])
        for k, v in sorted(summary.tier2_gates.items()):
            md_lines.append(f"- {k}: {v}")

    if summary.feature_list_used:
        md_lines.extend([
            "",
            f"## Features ({len(summary.feature_list_used)} used, "
            f"{len(summary.features_dropped)} dropped)",
        ])

    md_lines.append("")
    md_path = out / "promotion_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Promotion summary written to %s", out)
    return json_path
