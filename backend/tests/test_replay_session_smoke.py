import json

from backend.app.ml_platform.replay.replay_session import replay_session


def test_replay_session_smoke(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"order": "buy", "qty": 1}), encoding="utf-8")
    out = replay_session([p])
    assert out["count"] == 1
    assert out["decision_hash"]
