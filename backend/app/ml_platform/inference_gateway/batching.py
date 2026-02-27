from __future__ import annotations

import time
from collections import deque


class MicroBatchQueue:
    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 50, max_queue_ms: int = 200, drop_noncritical: bool = True):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.max_queue_ms = max_queue_ms
        self.drop_noncritical = drop_noncritical
        self._q: deque = deque()

    def push(self, item: dict) -> None:
        item["_enqueued_at"] = time.time()
        self._q.append(item)

    def pop_batch(self) -> list[dict]:
        batch: list[dict] = []
        start = time.time()
        now = time.time()
        while self._q and (now - self._q[0]["_enqueued_at"]) * 1000 > self.max_queue_ms:
            item = self._q[0]
            if item.get("critical", True) or not self.drop_noncritical:
                break
            self._q.popleft()
        while self._q and len(batch) < self.max_batch_size:
            if (time.time() - start) * 1000 > self.timeout_ms and batch:
                break
            batch.append(self._q.popleft())
        return batch
