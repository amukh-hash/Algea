from pathlib import Path


def test_chronos_no_svd_pca_guard():
    root = Path("backend/app/ml_platform")
    bad = []
    for p in (root / "models" / "chronos2").glob("*.py"):
        t = p.read_text(encoding="utf-8").lower()
        if "truncatedsvd" in t or "pca" in t or "svd" in t:
            bad.append(str(p))
    assert not bad
