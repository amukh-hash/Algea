from pathlib import Path


def test_selector_no_grip_guard():
    root = Path("backend/app/ml_platform/models/selector_smoe")
    for p in root.glob("*.py"):
        assert "grip" not in p.read_text(encoding="utf-8").lower()
