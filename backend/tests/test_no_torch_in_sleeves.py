from pathlib import Path


def test_no_torch_imports_in_sleeves() -> None:
    strategies_dir = Path("backend/app/strategies")
    offenders: list[str] = []
    for py_file in strategies_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "import torch" in text or "from torch" in text:
            offenders.append(str(py_file))
    assert not offenders, f"Sleeves must use inference gateway; torch imports found: {offenders}"
