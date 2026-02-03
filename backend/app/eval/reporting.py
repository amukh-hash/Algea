import json
import os
from typing import Dict, Any

def save_report(metrics: Dict[str, Any], decision: bool, reasons: list, path: str):
    report = {
        "metrics": metrics,
        "decision": "PROMOTE" if decision else "REJECT",
        "reasons": reasons
    }
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {path}")
