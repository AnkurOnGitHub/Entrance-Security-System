"""
storage.py — Feature 16: SQLite event database and CSV export.

The camera loop writes here and the dashboard thread reads from
here at the same time, which is why WAL mode and a lock are used.
"""

import csv
import json
import sqlite3
import threading
import time
from datetime import datetime

from config import DASHBOARD_HISTORY_LIMIT, DB_ENABLED, DB_FILE

# ==============================================================================
# ### FILE: storage.py
# Feature 16 — SQLite database + CSV export (NEW).
#
# Why: the daily CSV was fine for a demo but cannot be queried, has no ids,
# and cannot be read safely while the camera is writing it. The dashboard
# (Feature 18) needs to read history while the loop keeps running, which is
# exactly what SQLite in WAL mode is for.
#
# The original daily CSV is STILL written, so anything you already built on
# top of it keeps working. The database is an addition, not a replacement.
# ==============================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_epoch    REAL    NOT NULL,
    ts_iso      TEXT    NOT NULL,
    day         TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    severity    TEXT    NOT NULL,
    subject     TEXT,
    message     TEXT,
    snapshot    TEXT,
    clip        TEXT,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_day  ON events(day);
"""


class EventStore:
    """
    Thin wrapper over SQLite. Deliberately small — no ORM, no migrations.

    Thread safety: the camera loop writes and the dashboard thread reads, so
    the connection is opened with check_same_thread=False and every call is
    wrapped in a lock. WAL mode lets a reader work while a writer is active.
    """

    def __init__(self, path=None, enabled=None):
        self.path = path or DB_FILE
        self.enabled = DB_ENABLED if enabled is None else enabled
        self.conn = None
        self._lock = threading.Lock()
        if not self.enabled:
            return
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA_SQL)
            self.conn.commit()

    def add_event(self, kind, severity, subject, message,
                  snapshot=None, clip=None, meta=None, ts=None):
        """Insert one event and return its row id (None if the DB is off)."""
        if not self.enabled or self.conn is None:
            return None
        ts = ts or time.time()
        stamp = datetime.fromtimestamp(ts)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO events (ts_epoch, ts_iso, day, kind, severity,"
                " subject, message, snapshot, clip, meta)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    ts,
                    stamp.isoformat(timespec="seconds"),
                    stamp.strftime("%Y-%m-%d"),
                    kind,
                    severity,
                    subject,
                    message,
                    snapshot,
                    clip,
                    json.dumps(meta or {}),
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    def attach_clip(self, event_id, clip_path):
        """A clip finishes recording after the event row is written."""
        if not self.enabled or self.conn is None or event_id is None:
            return
        with self._lock:
            self.conn.execute(
                "UPDATE events SET clip=? WHERE id=?", (clip_path, event_id)
            )
            self.conn.commit()

    def recent_events(self, limit=None, kind=None, day=None):
        """Newest first. Used by the dashboard and by CSV export."""
        if not self.enabled or self.conn is None:
            return []
        limit = limit or DASHBOARD_HISTORY_LIMIT
        sql = "SELECT * FROM events"
        where, params = [], []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if day:
            where.append("day = ?")
            params.append(day)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_epoch DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def counts_by_kind(self, since_epoch=None):
        """Small summary for the dashboard header."""
        if not self.enabled or self.conn is None:
            return {}
        sql = "SELECT kind, COUNT(*) AS n FROM events"
        params = []
        if since_epoch:
            sql += " WHERE ts_epoch >= ?"
            params.append(since_epoch)
        sql += " GROUP BY kind"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return {r["kind"]: r["n"] for r in rows}

    def export_csv(self, path, day=None, limit=1000000):
        """Write the history out as CSV — for reports, or to hand to a judge."""
        rows = self.recent_events(limit=limit, day=day)
        rows.reverse()  # oldest first reads better in a report
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "time", "day", "kind", "severity",
                "subject", "message", "snapshot", "clip",
            ])
            for r in rows:
                writer.writerow([
                    r["id"], r["ts_iso"], r["day"], r["kind"], r["severity"],
                    r["subject"], r["message"], r["snapshot"] or "", r["clip"] or "",
                ])
        return path

    def close(self):
        if self.conn is not None:
            with self._lock:
                self.conn.close()
            self.conn = None
