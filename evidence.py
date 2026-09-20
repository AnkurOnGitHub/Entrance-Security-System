"""
evidence.py — Feature 15: snapshots and video clips for alerts.

Keeps a rolling buffer of recent frames so a saved clip can show
the seconds BEFORE the alert, not just the aftermath.
"""

import os
import time
from collections import deque
from datetime import datetime

import cv2

from config import (
    EVIDENCE_CLIPS,
    EVIDENCE_DIR,
    EVIDENCE_ENABLED,
    EVIDENCE_FALLBACK_FPS,
    EVIDENCE_JPEG_QUALITY,
    EVIDENCE_POST_SECONDS,
    EVIDENCE_PRE_SECONDS,
    EVIDENCE_RETENTION_DAYS,
    EVIDENCE_SNAPSHOTS,
)

# ==============================================================================
# ### FILE: evidence.py
# Feature 15 — Evidence snapshots and video clips (NEW).
#
# Why a rolling buffer: if you only start recording when the alert fires, the
# clip begins AFTER the interesting moment. Keeping the last few seconds of
# frames in memory means the saved clip shows the approach, not just the
# aftermath. That is the difference between useful evidence and a video of
# someone already walking away.
# ==============================================================================

class EvidenceRecorder:
    """
    Holds a rolling buffer of recent frames and writes snapshots/clips.

    add_frame() must be called every loop iteration — that is what fills the
    pre-roll buffer. Everything else is triggered by the AlertManager.
    """

    def __init__(self, base_dir=None, enabled=None):
        self.base_dir = base_dir or EVIDENCE_DIR
        self.enabled = EVIDENCE_ENABLED if enabled is None else enabled
        self.fps = EVIDENCE_FALLBACK_FPS
        buffer_len = max(1, int(EVIDENCE_PRE_SECONDS * EVIDENCE_FALLBACK_FPS))
        self.buffer = deque(maxlen=buffer_len)
        self.writer = None
        self.clip_path = None
        self.clip_until = 0.0
        self.clip_event_id = None
        self.on_clip_done = None  # set by AlertManager to update the DB
        if self.enabled:
            os.makedirs(self.base_dir, exist_ok=True)

    # -- housekeeping ------------------------------------------------------
    def _day_dir(self):
        day = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.base_dir, day)
        os.makedirs(path, exist_ok=True)
        return path

    def set_fps(self, fps):
        """Keep the buffer length matched to the real measured frame rate."""
        if not self.enabled or fps <= 0:
            return
        self.fps = max(1.0, min(60.0, fps))
        wanted = max(1, int(EVIDENCE_PRE_SECONDS * self.fps))
        if wanted != self.buffer.maxlen:
            self.buffer = deque(self.buffer, maxlen=wanted)

    def add_frame(self, frame):
        """Call once per loop. Fills pre-roll and feeds any active clip."""
        if not self.enabled:
            return
        if EVIDENCE_CLIPS:
            self.buffer.append(frame.copy())
        if self.writer is not None:
            self.writer.write(frame)
            if time.time() >= self.clip_until:
                self._finish_clip()

    # -- capture -----------------------------------------------------------
    def snapshot(self, frame, kind, event_id=None):
        """Save one JPEG. Returns the path, or None."""
        if not self.enabled or not EVIDENCE_SNAPSHOTS or frame is None:
            return None
        stamp = datetime.now().strftime("%H%M%S")
        tag = f"{event_id}" if event_id else stamp
        name = f"{stamp}_{kind.lower()}_{tag}.jpg"
        path = os.path.join(self._day_dir(), name)
        try:
            cv2.imwrite(
                path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), EVIDENCE_JPEG_QUALITY]
            )
        except Exception as error:  # a full disk should not kill the camera
            print(f"⚠️ Could not save snapshot: {error}")
            return None
        return path

    def start_clip(self, kind, event_id=None):
        """
        Begin (or extend) a video clip.

        If a clip is already recording we just push its end time out, instead
        of starting a second file. One incident should be one video.
        """
        if not self.enabled or not EVIDENCE_CLIPS:
            return None
        now = time.time()
        if self.writer is not None:
            self.clip_until = max(self.clip_until, now + EVIDENCE_POST_SECONDS)
            return self.clip_path

        if not self.buffer:
            return None
        height, width = self.buffer[-1].shape[:2]
        stamp = datetime.now().strftime("%H%M%S")
        tag = f"{event_id}" if event_id else stamp
        name = f"{stamp}_{kind.lower()}_{tag}.mp4"
        path = os.path.join(self._day_dir(), name)

        # mp4v is the codec most likely to already be available with the
        # pip build of OpenCV. If it fails we fall back to .avi / MJPG.
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            path = path[:-4] + ".avi"
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (width, height)
            )
        if not writer.isOpened():
            print("⚠️ No usable video codec — clips disabled for this run.")
            return None

        # Write the pre-roll first: this is the whole point of the buffer.
        for buffered in list(self.buffer):
            writer.write(buffered)

        self.writer = writer
        self.clip_path = path
        self.clip_event_id = event_id
        self.clip_until = now + EVIDENCE_POST_SECONDS
        return path

    def _finish_clip(self):
        if self.writer is None:
            return
        self.writer.release()
        finished_path = self.clip_path
        finished_id = self.clip_event_id
        self.writer = None
        self.clip_path = None
        self.clip_event_id = None
        if self.on_clip_done and finished_path:
            self.on_clip_done(finished_id, finished_path)

    def purge_old(self, days=None):
        """Delete evidence folders older than the retention window."""
        days = EVIDENCE_RETENTION_DAYS if days is None else days
        if not self.enabled or not days:
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        if not os.path.isdir(self.base_dir):
            return 0
        for entry in os.listdir(self.base_dir):
            full = os.path.join(self.base_dir, entry)
            if not os.path.isdir(full):
                continue
            try:
                day = datetime.strptime(entry, "%Y-%m-%d").timestamp()
            except ValueError:
                continue  # not one of our day folders — leave it alone
            if day < cutoff:
                for name in os.listdir(full):
                    try:
                        os.remove(os.path.join(full, name))
                        removed += 1
                    except OSError:
                        pass
                try:
                    os.rmdir(full)
                except OSError:
                    pass
        return removed

    def close(self):
        self._finish_clip()
