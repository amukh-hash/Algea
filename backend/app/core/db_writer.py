"""Dedicated SQLite single-writer thread (Outbox pattern).

SQLite WAL supports concurrent readers but strictly one writer.  This
daemon thread batches high-frequency writes (telemetry, ECE logs) off
the ``asyncio`` event loop to prevent ``SQLITE_BUSY`` starvation.

Resolves **F6** (ASGI event loop starvation — write side).
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class SQLiteOutboxWriter(threading.Thread):
    """Batched background SQLite writer.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    maxsize : int
        Maximum queue depth.  Drops writes if the queue is full (fail-open
        for telemetry; orchestrator writes use direct connections).
    batch_size : int
        Maximum statements per ``BEGIN IMMEDIATE`` transaction.
    """

    def __init__(
        self,
        db_path: str,
        maxsize: int = 10_000,
        batch_size: int = 500,
    ) -> None:
        super().__init__(daemon=True, name="sqlite-outbox-writer")
        self.db_path = db_path
        self.q: queue.Queue[tuple[str, tuple[Any, ...]]] = queue.Queue(
            maxsize=maxsize
        )
        self.batch_size = batch_size
        self._stop_event = threading.Event()

    def enqueue(self, sql: str, params: tuple[Any, ...] = ()) -> bool:
        """Submit a SQL statement for background execution.

        Returns ``False`` if the queue is full (statement dropped).
        """
        try:
            self.q.put_nowait((sql, params))
            return True
        except queue.Full:
            logger.warning("Outbox queue full — dropping: %s", sql[:80])
            return False

    def stop(self) -> None:
        """Signal the writer thread to drain and exit."""
        self._stop_event.set()

    def run(self) -> None:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        while not self._stop_event.is_set():
            batch: list[tuple[str, tuple[Any, ...]]] = []
            try:
                batch.append(self.q.get(timeout=1.0))
                while len(batch) < self.batch_size and not self.q.empty():
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                # Idle tick — compact the WAL
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
                continue

            if batch:
                for attempt in range(5):
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        for sql, params in batch:
                            conn.execute(sql, params)
                        conn.execute("COMMIT")
                        break
                    except sqlite3.OperationalError:
                        # SQLITE_BUSY — exponential backoff
                        try:
                            conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        time.sleep(0.1 * (2**attempt))
                else:
                    logger.error(
                        "Outbox: failed to commit batch of %d after 5 retries",
                        len(batch),
                    )

                for _ in batch:
                    self.q.task_done()

        # Drain remaining items before exit
        while not self.q.empty():
            try:
                sql, params = self.q.get_nowait()
                conn.execute(sql, params)
                self.q.task_done()
            except queue.Empty:
                break
            except Exception:
                pass

        conn.close()
        logger.info("SQLiteOutboxWriter stopped")
