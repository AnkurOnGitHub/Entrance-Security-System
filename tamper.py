"""
tamper.py — Feature 6 checks plus the Feature 13 baseline watcher.

Catches a covered lens AND a camera that has been turned to face
somewhere else, without false-alarming every evening at dusk.
"""

import time

import cv2
import numpy as np

from config import (
    TAMPER_BASELINE_ALPHA,
    TAMPER_BASELINE_WARMUP_SECONDS,
    TAMPER_BLUR_VARIANCE,
    TAMPER_BRIGHT_MEAN,
    TAMPER_CONFIRM_SECONDS,
    TAMPER_DARK_MEAN,
    TAMPER_FLAT_STD,
    TAMPER_SCENE_CHANGE_THRESHOLD,
    TAMPER_SCENE_CONFIRM_SECONDS,
)

# ==============================================================================
# ### FILE: tamper.py
# Feature 6 — camera tamper, now upgraded by Feature 13.
#
# The original four checks (dark / bright / flat / blurry) are unchanged and
# still run first — they catch a hand or a cloth over the lens.
#
# What Feature 13 adds is the case those checks CANNOT see: the camera is
# still producing a perfectly normal-looking picture, but it is pointing
# somewhere else because someone turned it. To notice that, we have to know
# what the normal view looks like, so we keep a slowly-updated baseline.
#
# The slow update is the important part. A sunset changes the picture too,
# but over minutes; the baseline follows it and nothing fires. A hand
# turning the camera changes it in a second, and the baseline has not moved,
# so the difference is large.
# ==============================================================================

def is_camera_tampered(frame):
    """
    Feature 6 — camera tamper (the original absolute checks).

    If someone puts a hand, cloth, or sticker over the lens, the image is no
    longer a normal scene. We treat it as tamper when ANY of these hold:
      - too dark  (covered lens)
      - too bright (flashlight / overexposure attack)
      - almost no contrast (flat color covering the camera)
      - almost no edges / blur (Laplacian variance is low)

    A single dark frame is not enough; the caller waits TAMPER_CONFIRM_SECONDS.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return (
        mean < TAMPER_DARK_MEAN
        or mean > TAMPER_BRIGHT_MEAN
        or std < TAMPER_FLAT_STD
        or sharpness < TAMPER_BLUR_VARIANCE
    )


def describe_tamper(frame):
    """Same checks, but says WHICH one tripped — useful in the alert text."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if mean < TAMPER_DARK_MEAN:
        return "lens covered (view is dark)"
    if mean > TAMPER_BRIGHT_MEAN:
        return "bright light aimed at lens"
    if std < TAMPER_FLAT_STD:
        return "lens blocked (flat colour)"
    if sharpness < TAMPER_BLUR_VARIANCE:
        return "lens blurred or sprayed"
    return None


class TamperWatcher:
    """
    Feature 13 — keeps a memory of the normal scene and watches for changes.

    Two independent alarms:
      BLOCKED : the original absolute checks, confirmed over time.
      MOVED   : the picture is fine but it no longer matches the baseline,
                so the camera is probably pointing somewhere new.

    The baseline is only updated while the view is NORMAL. That matters: if
    we kept updating during tampering, the covered lens would quietly become
    the new "normal" and the alert would switch itself off after a while.
    """

    def __init__(self):
        self.baseline = None          # small grayscale picture of normal view
        self.baseline_started = 0.0
        self.blocked_since = None
        self.changed_since = None
        self.last_reason = None
        self.last_similarity = 1.0

    @staticmethod
    def _thumb(frame):
        """Small blurred grayscale copy — compares structure, ignores noise."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (96, 72))
        return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)

    def _similarity(self, thumb):
        """
        0..1, how much this view still looks like the baseline.

        We compare edge maps rather than raw brightness, because edges
        describe WHERE things are. A light turning on changes brightness
        everywhere but leaves the edges of the door frame in place; turning
        the camera moves every edge.
        """
        base_edges = cv2.Laplacian(self.baseline, cv2.CV_32F)
        now_edges = cv2.Laplacian(thumb, cv2.CV_32F)
        base_edges = cv2.normalize(base_edges, None, 0, 1, cv2.NORM_MINMAX)
        now_edges = cv2.normalize(now_edges, None, 0, 1, cv2.NORM_MINMAX)
        result = cv2.matchTemplate(now_edges, base_edges, cv2.TM_CCOEFF_NORMED)
        return float(result[0][0])

    def update(self, frame, now=None):
        """
        Call once per frame.

        Returns (tampered, reason) where reason is None when all is well.
        """
        now = now or time.time()
        thumb = self._thumb(frame)

        if self.baseline is None:
            self.baseline = thumb
            self.baseline_started = now
            return False, None

        # ---- alarm 1: the original blocked-lens checks ----
        reason = describe_tamper(frame)
        if reason is not None:
            if self.blocked_since is None:
                self.blocked_since = now
            elif now - self.blocked_since >= TAMPER_CONFIRM_SECONDS:
                self.last_reason = reason
                return True, reason
            # While blocked we deliberately do NOT update the baseline.
            return False, None
        self.blocked_since = None

        # ---- alarm 2: scene change (camera moved / redirected) ----
        similarity = self._similarity(thumb)
        self.last_similarity = similarity
        warm = (now - self.baseline_started) >= TAMPER_BASELINE_WARMUP_SECONDS

        if warm and similarity < (1.0 - TAMPER_SCENE_CHANGE_THRESHOLD):
            if self.changed_since is None:
                self.changed_since = now
            elif now - self.changed_since >= TAMPER_SCENE_CONFIRM_SECONDS:
                self.last_reason = "camera moved or view changed"
                # Do not learn the new view while we are alerting on it.
                return True, self.last_reason
            return False, None

        self.changed_since = None

        # ---- normal: let the baseline drift slowly toward what we see ----
        # This is what absorbs sunset, clouds and a light being switched on.
        alpha = TAMPER_BASELINE_ALPHA
        self.baseline = (1.0 - alpha) * self.baseline + alpha * thumb
        return False, None

    def relearn(self):
        """Forget the baseline — call after intentionally moving the camera."""
        self.baseline = None
        self.blocked_since = None
        self.changed_since = None
