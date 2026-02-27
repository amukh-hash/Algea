from __future__ import annotations

APP_NAME = "Algae"
APP_VERSION = "4.0.0"
APP_DISPLAY = f"{APP_NAME} 4.0"


def app_metadata() -> dict[str, str]:
    return {"app": APP_NAME, "version": APP_VERSION, "display": APP_DISPLAY}


def with_app_metadata(payload: dict) -> dict:
    stamped = dict(payload)
    stamped.setdefault("app", APP_NAME)
    stamped.setdefault("version", APP_VERSION)
    return stamped
