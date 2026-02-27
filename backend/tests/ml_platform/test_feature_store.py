from pathlib import Path

from backend.app.ml_platform.feature_store.builders.cross_sectional import build_cross_sectional
from backend.app.ml_platform.feature_store.lineage import build_lineage_manifest


def test_stable_ordering_and_deterministic_lineage(tmp_path: Path) -> None:
    rows = [
        {"asof": "2026-01-02", "symbol": "MSFT", "universe_id": "u"},
        {"asof": "2026-01-02", "symbol": "AAPL", "universe_id": "u"},
    ]
    built = build_cross_sectional(rows)
    assert [r["symbol"] for r in built] == ["AAPL", "MSFT"]

    f = tmp_path / "input.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    m1 = build_lineage_manifest("d1", ["2020:2021"], [f], "codehash", "univhash")
    m2 = build_lineage_manifest("d1", ["2020:2021"], [f], "codehash", "univhash")
    assert m1["lineage_hash"] == m2["lineage_hash"]
