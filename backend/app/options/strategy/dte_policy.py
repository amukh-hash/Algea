from typing import Tuple
from backend.app.options.config import OPTIONS_DTE_BUCKET

def get_target_dte_range() -> Tuple[int, int]:
    """
    Parses OPTIONS_DTE_BUCKET config.
    e.g. "3-7" -> (3, 7)
    """
    parts = OPTIONS_DTE_BUCKET.split("-")
    if len(parts) != 2:
        return (3, 7) # Default
    return (int(parts[0]), int(parts[1]))

def is_dte_valid(dte: int) -> bool:
    min_d, max_d = get_target_dte_range()
    return min_d <= dte <= max_d
