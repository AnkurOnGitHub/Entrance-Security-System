"""
tracking.py — Feature 9: stable identities on top of YOLO track ids.

YOLO's own ids break when a person is briefly hidden. These
identities survive that, so the loiter timer never restarts by
accident. Also holds the Feature 12 face cache for each person.
"""

import math

import cv2

from config import (
    BELL_GRACE_SECONDS,
    FACE_CACHE_SECONDS,
    IDENTITY_TTL_SECONDS,
    REID_MAX_DISTANCE_RATIO,
    REID_MIN_HIST_SIMILARITY,
    REID_MIN_SCORE,
    TRACK_LOST_GRACE_SECONDS,
    ZONE_EXIT_RESET_SECONDS,
)

# ==============================================================================
# ### FILE: tracking.py
# Feature 9 — Robust person tracking (added before Feature 10).
# Sits on top of YOLO's track ids and keeps a stable identity per person, so a
# short dropout or an id switch does not restart the Feature 5 loiter timer.
# Uses Feature 8's in_zone flag, but knows nothing about the camera loop.
# Feature 12 now also caches each identity's recognised face here.
# ==============================================================================

# ---------------------------------------------------------------------------
# Feature 9 helpers: identity signature + re-identification
# ---------------------------------------------------------------------------
def person_signature(frame, x1, y1, x2, y2):
    """
    A simple colour "fingerprint" of a person, used to recognise them again
    after they disappear for a moment.

    We take the TOP 60% of the person box (torso / shirt / jacket) because
    clothing colour there is stable, while legs and floor change a lot.
    Then we build an HSV hue+saturation histogram. Two crops of the same
    person give similar histograms even if the pose changes.

    This is not real face-level ReID — it is a cheap, CPU-friendly stand-in
    that works well at a single quiet entrance.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    torso_bottom = y1 + int((y2 - y1) * 0.6)
    crop = frame[y1:torso_bottom, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def signature_similarity(sig_a, sig_b):
    """0..1, higher means more likely the same person's clothing."""
    if sig_a is None or sig_b is None:
        return None
    score = cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_CORREL)
    return max(0.0, float(score))


def find_lost_identity(identities, now, center, signature, frame_diag):
    """
    Try to match a freshly-appeared box to someone we recently lost.

    Two clues are combined:
      - distance: a person cannot teleport, so they should reappear near
        where they vanished.
      - colour: their clothing histogram should still match.

    Returns the best identity id, or None if nothing is a confident match.
    """
    best_id = None
    best_score = 0.0
    max_distance = frame_diag * REID_MAX_DISTANCE_RATIO

    for ident_id, person in identities.items():
        if person["seen_this_frame"]:
            continue  # already claimed by another box this frame

        gap = now - person["last_seen"]
        if gap > IDENTITY_TTL_SECONDS:
            continue

        distance = math.hypot(
            center[0] - person["last_center"][0],
            center[1] - person["last_center"][1],
        )
        if distance > max_distance:
            continue
        distance_score = 1.0 - (distance / max_distance)

        colour_score = signature_similarity(signature, person["signature"])
        if colour_score is None:
            colour_score = 0.5  # no signature available → stay neutral
        elif colour_score < REID_MIN_HIST_SIMILARITY and gap > 1.0:
            continue  # clothing clearly different and the gap was long

        score = 0.45 * distance_score + 0.55 * colour_score
        if score >= REID_MIN_SCORE and score > best_score:
            best_score = score
            best_id = ident_id

    return best_id


def new_identity(identities, ident_id, now, center, signature):
    identities[ident_id] = {
        "dwell": 0.0,            # seconds accumulated inside the zone
        "zone_active": False,    # currently counted as "at the entrance"
        "left_zone_at": None,    # when they stepped out (grace before reset)
        "last_seen": now,
        "last_center": center,
        "signature": signature,
        "bell_time": None,
        "seen_this_frame": True,
        "recovered": False,      # for the on-screen "REACQ" marker
        "recovered_at": 0.0,
        # --- Feature 12: cached face result for this person ---
        # Recognising the same person on every frame is the slowest thing
        # the program does. Once we are sure, we store the answer here and
        # stop asking until FACE_CACHE_SECONDS has passed.
        "face_name": None,
        "face_status": None,     # "AUTHENTIC" / "ALERT" / None = not decided
        "face_votes": {},        # name → how many times we have seen it
        "face_checked_at": 0.0,
        "face_confirmed_at": 0.0,
        "face_box": None,
    }
    return ident_id


def update_identity(person, now, center, signature, in_zone):
    """
    Advance one person's dwell clock.

    The key line is `person["dwell"] += gap`. Because we ADD elapsed time
    instead of measuring from a fixed start, a 2-second detection dropout
    simply gets counted as 2 seconds of standing there — the clock never
    restarts.
    """
    gap = now - person["last_seen"]

    if in_zone:
        if person["zone_active"] and gap <= TRACK_LOST_GRACE_SECONDS:
            person["dwell"] += gap
        person["zone_active"] = True
        person["left_zone_at"] = None
    else:
        # Out of the zone: freeze the clock, and only wipe it if they stay
        # out for a while (a quick side-step should not clear the timer).
        if person["zone_active"]:
            if person["left_zone_at"] is None:
                person["left_zone_at"] = now
            elif now - person["left_zone_at"] >= ZONE_EXIT_RESET_SECONDS:
                person["dwell"] = 0.0
                person["zone_active"] = False
                person["left_zone_at"] = None

    person["last_seen"] = now
    person["last_center"] = center
    if signature is not None:
        # Blend slowly so lighting changes don't ruin the fingerprint.
        if person["signature"] is None:
            person["signature"] = signature
        else:
            person["signature"] = cv2.addWeighted(
                person["signature"], 0.8, signature, 0.2, 0
            )


def identity_has_rung_bell(person, now):
    """Feature 10 + 11 helper: is this person inside the doorbell grace window?"""
    bell_at = person.get("bell_time")
    return bell_at is not None and (now - bell_at) <= BELL_GRACE_SECONDS
