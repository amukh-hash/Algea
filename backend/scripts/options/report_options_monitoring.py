import os
import sys
from backend.app.options.monitoring.reporting import generate_daily_report

def main():
    # Mock events for demonstration
    events = [
        {"type": "VETO", "reason": "REJECT_REGIME"},
        {"type": "VETO", "reason": "REJECT_IV"},
        {"type": "TRIGGER", "ticker": "AAPL"}
    ]

    os.makedirs("backend/data/options/reports", exist_ok=True)
    out_path = "backend/data/options/reports/daily_report.json"
    generate_daily_report(events, out_path)
    print(f"Report generated at {out_path}")

if __name__ == "__main__":
    main()
