"""Tiny thread-safe SQLite key/value cache for tool and LLM calls (makes runs resumable and reproducible)."""
import hashlib
import json
import sqlite3
import threading

from config import CACHE_DIR


class Cache:
    def __init__(self, name: str):
        self._conn = sqlite3.connect(CACHE_DIR / f"{name}.sqlite", check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts) -> str:
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, k: str):
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, k: str, v) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (k, json.dumps(v, ensure_ascii=False))
            )
            self._conn.commit()
