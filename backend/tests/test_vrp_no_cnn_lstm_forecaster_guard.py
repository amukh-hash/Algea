from pathlib import Path


def test_vrp_no_cnn_lstm_forecaster_guard():
    txt = Path("backend/app/strategies/vrp/vrp_sleeve.py").read_text(encoding="utf-8").lower()
    assert "lstm" not in txt and "cnn" not in txt
    assert "vol_surface_forecast" in txt
