"""
alerts.py — Feature 14: the one place that decides what is an alert.

Takes an EventStore (storage.py) and an EvidenceRecorder
(evidence.py) as arguments rather than importing them, so this
module stays testable with fakes and has no circular imports.
"""

import time
from collections import deque
from datetime import datetime

from config import (
    ALERT_COOLDOWNS,
    ALERT_COOLDOWN_SECONDS,
    ALERT_PRINT,
    CSV_ENABLED,
)

# ==============================================================================
# ### FILE: alerts.py
# Feature 14 — Centralized alert manager (NEW).
#
# Before this, every feature had its own cooldown variable, its own print
# line, its own CSV write, and its own banner entry. Five copies of the same
# logic is five places for a bug to hide, and adding an alert type meant
# touching the camera loop in four places.
#
# Now a feature just says, once per frame:
#     alerts.report("LOITERING", active=True, message=..., subject=...)
# and the manager decides whether that is a NEW event worth recording.
#
# Two ideas are kept separate on purpose:
#   CONDITION — "this is true right now"  → drives the on-screen banner
#   EVENT     — "this became true"        → recorded once, with evidence
# ==============================================================================

SEVERITY_COLORS = {
    "INFO": (120, 120, 120),
    "OK": (0, 140, 0),
    "WARN": (0, 120, 200),
    "ALERT": (0, 0, 180),
    "CRITICAL": (0, 0, 120),
}


class AlertManager:
    """One place that decides what counts as an alert and what happens next."""

    def __init__(self, store=None, evidence=None, csv_writer=None):
        self.store = store
        self.evidence = evidence
        self.csv_writer = csv_writer
        self._last_fired = {}      # (kind, subject) → epoch of last event
        self._banners = []         # conditions true in the current frame
        self._recent = deque(maxlen=50)   # for the dashboard "live" list
        self.frame_for_evidence = None
        if self.evidence is not None and self.store is not None:
            self.evidence.on_clip_done = self._clip_done

    # -- per frame ---------------------------------------------------------
    def begin_frame(self, frame=None):
        """Clear last frame's banners and remember the frame for evidence."""
        self._banners = []
        self.frame_for_evidence = frame

    def banners(self):
        """[(text, colour)] for everything currently true. Drawn by main."""
        return list(self._banners)

    def _cooldown_for(self, kind):
        return ALERT_COOLDOWNS.get(kind, ALERT_COOLDOWN_SECONDS)

    # -- the one call every feature uses -----------------------------------
    def report(self, kind, active, message, subject="-", severity="ALERT",
               color=None, meta=None, capture=True, banner=True):
        """
        Tell the manager the current state of one condition.

        active=True  → show a banner, and fire an event if the cooldown for
                       this (kind, subject) has passed.
        active=False → nothing; the condition simply is not happening.

        Returns True only when a NEW event was actually recorded, so callers
        can print extra detail without repeating the manager's own logging.
        """
        if not active:
            return False

        if banner:
            self._banners.append((message, color or SEVERITY_COLORS.get(severity, (0, 0, 180))))

        key = (kind, str(subject))
        now = time.time()
        if now - self._last_fired.get(key, 0.0) < self._cooldown_for(kind):
            return False
        self._last_fired[key] = now

        return self._fire(kind, severity, subject, message, meta, capture, now)

    def event(self, kind, message, subject="-", severity="INFO", meta=None,
              capture=False):
        """
        Record a one-off event that is not a lasting condition — a doorbell
        ring, a known face arriving, the system starting up.
        """
        key = (kind, str(subject))
        now = time.time()
        if now - self._last_fired.get(key, 0.0) < self._cooldown_for(kind):
            return False
        self._last_fired[key] = now
        return self._fire(kind, severity, subject, message, meta, capture, now)

    # -- internals ---------------------------------------------------------
    def _fire(self, kind, severity, subject, message, meta, capture, now):
        snapshot_path = None
        event_id = None

        if self.store is not None:
            event_id = self.store.add_event(
                kind=kind, severity=severity, subject=str(subject),
                message=message, meta=meta, ts=now,
            )

        if capture and self.evidence is not None and self.frame_for_evidence is not None:
            snapshot_path = self.evidence.snapshot(
                self.frame_for_evidence, kind, event_id
            )
            if snapshot_path and self.store is not None and event_id is not None:
                with self.store._lock:
                    self.store.conn.execute(
                        "UPDATE events SET snapshot=? WHERE id=?",
                        (snapshot_path, event_id),
                    )
                    self.store.conn.commit()
            self.evidence.start_clip(kind, event_id)

        # The original daily CSV, kept exactly as before so nothing built on
        # it breaks. Columns: Name, Time, Status, Event.
        if CSV_ENABLED and self.csv_writer is not None:
            self.csv_writer.writerow([
                str(subject), datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                severity, kind,
            ])

        self._recent.appendleft({
            "id": event_id, "kind": kind, "severity": severity,
            "subject": str(subject), "message": message,
            "ts": now, "snapshot": snapshot_path,
        })

        if ALERT_PRINT:
            icon = "🚨" if severity in ("ALERT", "CRITICAL") else "ℹ️"
            stamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
            print(f"{icon} [{stamp}] {kind}: {message}")

        return True

    def _clip_done(self, event_id, clip_path):
        if self.store is not None:
            self.store.attach_clip(event_id, clip_path)

    def recent(self, limit=20):
        return list(self._recent)[:limit]
