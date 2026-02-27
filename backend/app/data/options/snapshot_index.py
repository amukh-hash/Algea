from __future__ import annotations

import sqlite3
from pathlib import Path


class SnapshotIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                create table if not exists option_snapshots(
                    snapshot_id text primary key,
                    underlying_symbol text,
                    asof text,
                    path text
                )
                """
            )

    def insert(self, snapshot_id: str, symbol: str, asof: str, path: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "insert or replace into option_snapshots(snapshot_id, underlying_symbol, asof, path) values(?,?,?,?)",
                (snapshot_id, symbol, asof, path),
            )

    def latest(self, symbol: str, asof: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "select path from option_snapshots where underlying_symbol=? and asof<=? order by asof desc limit 1",
                (symbol, asof),
            ).fetchone()
        return row[0] if row else None
