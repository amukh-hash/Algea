import time

from backend.app.ml_platform.inference_gateway.batching import MicroBatchQueue


def test_microbatch_queue_sla():
    q = MicroBatchQueue(max_batch_size=4, max_queue_ms=1)
    q.push({"id": 1, "critical": False})
    time.sleep(0.01)
    q.push({"id": 2, "critical": True})
    batch = q.pop_batch()
    ids = [x["id"] for x in batch]
    assert 1 not in ids
    assert 2 in ids
