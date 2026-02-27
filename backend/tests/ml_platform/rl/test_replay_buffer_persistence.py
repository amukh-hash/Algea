from backend.app.ml_platform.rl.replay_buffer import ReplayBuffer


def test_replay_buffer_persistence(tmp_path):
    path = tmp_path / "rb.json"
    rb = ReplayBuffer(path)
    rb.append([{"state": {"x": 1.0}, "reward": 0.1}])
    rb2 = ReplayBuffer(path)
    payload = rb2.load()
    assert len(payload["transitions"]) == 1
