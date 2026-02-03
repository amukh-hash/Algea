import json
from datetime import datetime

def generate_daily_report(events: list, output_path: str):
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_events": len(events),
        "triggers": len([e for e in events if e.get("type") == "TRIGGER"]),
        "vetoes": {}
    }

    for e in events:
        if e.get("type") == "VETO":
            reason = e.get("reason")
            report["vetoes"][reason] = report["vetoes"].get(reason, 0) + 1

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
