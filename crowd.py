"""
crowd.py — Feature 11: gathering detection.

Pure logic: it is handed the identities and who is in the zone,
and answers whether that counts as a gathering. No camera, no
drawing, no I/O — which is why it is easy to test.
"""

import time

from config import (
    CROWD_IGNORE_BELL_RUNG,
    CROWD_IGNORE_IF_RESIDENT_PRESENT,
    CROWD_PERSON_MIN,
    CROWD_SUSTAIN_SECONDS,
    CROWD_UNKNOWN_MIN,
)
from tracking import identity_has_rung_bell

# ==============================================================================
# ### FILE: crowd.py
# Feature 7 — crowd / unusual gathering, now upgraded by Feature 11.
#
# The old rule was: 2+ YOLO boxes anywhere in the frame AND 2+ unknown faces
# AND no resident → alert, on a single frame. Three problems with that:
#   - it counted people walking past in the background
#   - a group passing the door for one second was enough to fire
#   - it counted FACES, so two detections of one person could look like two
#     people, and a person facing away counted as nobody
#
# Feature 11 fixes all three by counting DISTINCT TRACKED IDENTITIES whose
# feet are inside the entrance zone, ignoring anyone who rang the bell, and
# requiring the situation to hold for CROWD_SUSTAIN_SECONDS.
# ==============================================================================

class CrowdWatcher:
    """Counts people at the door and decides when that becomes a gathering."""

    def __init__(self):
        self.gathering_since = None
        self.last_count = 0
        self.last_unknown = 0

    def evaluate(self, identities, ids_in_zone, now=None):
        """
        Returns (is_gathering, info dict).

        info carries the numbers so the alert text and the dashboard can
        show exactly what was counted, rather than a bare "crowd alert".
        """
        now = now or time.time()

        people = []
        for ident_id in ids_in_zone:
            person = identities.get(ident_id)
            if person is None:
                continue
            if CROWD_IGNORE_BELL_RUNG and identity_has_rung_bell(person, now):
                continue  # rang the bell → a visitor, not a gathering
            people.append(person)

        total = len(people)
        residents = sum(1 for p in people if p.get("face_status") == "AUTHENTIC")
        # "Unknown" includes people we have not identified yet — someone
        # standing with their back to the camera still counts as a body at
        # the door, which is the point of this feature.
        unknown = total - residents

        self.last_count = total
        self.last_unknown = unknown

        info = {
            "people": total,
            "unknown": unknown,
            "residents": residents,
            "held_for": 0.0,
        }

        big_enough = total >= CROWD_PERSON_MIN and unknown >= CROWD_UNKNOWN_MIN
        resident_blocks = CROWD_IGNORE_IF_RESIDENT_PRESENT and residents > 0

        if not big_enough or resident_blocks:
            self.gathering_since = None
            return False, info

        if self.gathering_since is None:
            self.gathering_since = now
        held = now - self.gathering_since
        info["held_for"] = held

        return held >= CROWD_SUSTAIN_SECONDS, info
