"""
faces.py — Features 1-4 (face auth) with the Feature 12 speed work.

known_face_encodings / known_face_names live HERE, not in config:
they are runtime state that load_known_faces() fills in.
"""

import os

import cv2
import numpy as np

# Heavy, and only needed when faces are actually used. Wrapped so
# the rest of the program can still be imported without them.
try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from config import (
    FACE_ADAPTIVE_SKIP,
    FACE_BOX_TOP_FRACTION,
    FACE_CACHE_SECONDS,
    FACE_DOWNSCALE,
    FACE_MAX_SKIP,
    FACE_MODEL,
    FACE_TARGET_FPS,
    FACE_TOLERANCE,
    FACE_VOTES_REQUIRED,
    photos_folder,
)

# Runtime state for this module. Filled by load_known_faces(),
# read by _match_encoding(). Moved out of config.py in the split,
# because a list that the program mutates is not a setting.
known_face_encodings = []
known_face_names = []

# ==============================================================================
# ### FILE: faces.py
# Features 1-4 — the ORIGINAL face work, now upgraded by Feature 12.
#
# What Feature 12 changed, and why:
#   (a) Faces are searched only inside YOLO person boxes. The old code
#       scanned the whole frame every time, including walls and posters.
#   (b) A recognised name is CACHED on the tracked identity (Feature 9), so
#       a person standing at the door is recognised once, not 300 times.
#       This is the single biggest speed win in the whole program.
#   (c) Several photos per person are supported — photos/name.jpg still
#       works, and photos/name/*.jpg now works too. More angles = fewer
#       "UNKNOWN" mistakes on a resident.
#   (d) A name must win FACE_VOTES_REQUIRED looks before it is trusted, and
#       the frame-skip adapts to the measured FPS.
# The recognition maths itself is unchanged: same encodings, same distance.
# ==============================================================================

def load_known_faces():
    """
    Register residents from photos/. Filename (without extension) = person name.

    Feature 12 addition: a SUBFOLDER is also a person. Either layout works:
        photos/shivam.jpg                  → one photo  (original layout)
        photos/shivam/front.jpg
        photos/shivam/side.jpg             → several photos of one person
    More photos per person means the same name can match from more angles.
    """
    if face_recognition is None or Image is None:
        print("⚠️ face_recognition / Pillow not installed — face auth disabled.")
        return

    if not os.path.isdir(photos_folder):
        print(f"⚠️ Photos folder not found: {photos_folder}")
        return

    def register(image_path, person_name):
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"⚠️ No face detected in {os.path.basename(image_path)}")
            return False
        known_face_encodings.append(encodings[0])
        known_face_names.append(person_name)
        return True

    for entry in sorted(os.listdir(photos_folder)):
        full = os.path.join(photos_folder, entry)

        # Layout 2 (new): a folder per person, several photos inside.
        if os.path.isdir(full):
            person_name = entry
            count = 0
            for filename in sorted(os.listdir(full)):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                print(f"Loading: {person_name}/{filename}")
                if register(os.path.join(full, filename), person_name):
                    count += 1
            if count:
                print(f"✅ Registered: {person_name} ({count} photos)")
            continue

        # Layout 1 (original): one file per person.
        if not entry.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        person_name = os.path.splitext(entry)[0]
        print(f"Loading: {entry}")
        if register(full, person_name):
            print(f"✅ Registered: {person_name}")


def _match_encoding(face_encoding):
    """Original matching logic, pulled out so both code paths share it."""
    name = "UNKNOWN"
    status = "ALERT"
    distance = 1.0
    if known_face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=FACE_TOLERANCE
        )
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = int(np.argmin(face_distances))
        distance = float(face_distances[best_match_index])
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            status = "AUTHENTIC"
    return name, status, distance


def identify_faces(frame):
    """
    Original face-auth logic, extracted into a function so YOLO can reuse the results.

    Shrinks the frame 4x (faster), finds faces, compares to known encodings.
    Returns list of {name, status, box} in full-frame coordinates.

    Feature 12 note: this whole-frame version is KEPT as the fallback for
    when there are no person boxes (or the setting is turned off). The
    faster per-person version is identify_faces_in_boxes() below.
    """
    if face_recognition is None:
        return []
    scale = FACE_DOWNSCALE
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame, model=FACE_MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        name, status, distance = _match_encoding(face_encoding)
        # Locations were computed on the scaled image — scale boxes back up.
        top, right, bottom, left = [int(v / scale) for v in face_location]
        results.append(
            {
                "name": name,
                "status": status,
                "distance": distance,
                "box": (left, top, right, bottom),
            }
        )
    return results


def identify_faces_in_boxes(frame, person_boxes, identities, now):
    """
    Feature 12 — the fast path.

    For each tracked person we:
      1. skip them entirely if we already know who they are and the cache
         is still fresh  (this is where nearly all the speed comes from)
      2. otherwise crop the TOP part of their body box and look for a face
         only in there
      3. feed the answer into a small vote, so one bad frame cannot rename
         a resident to UNKNOWN

    Returns the list of face results found this frame (same shape as
    identify_faces), each one tagged with the identity it belongs to.
    """
    if face_recognition is None:
        return []

    results = []
    height, width = frame.shape[:2]
    scale = FACE_DOWNSCALE

    for ident_id, x1, y1, x2, y2, in_zone in person_boxes:
        person = identities.get(ident_id)
        if person is None:
            continue

        # -- step 1: is the cached answer still good? --
        confirmed = person["face_status"] is not None
        fresh = (now - person["face_confirmed_at"]) < FACE_CACHE_SECONDS
        if confirmed and fresh:
            if person["face_box"]:
                results.append({
                    "name": person["face_name"],
                    "status": person["face_status"],
                    "box": person["face_box"],
                    "identity": ident_id,
                    "cached": True,
                })
            continue

        # -- step 2: crop just the head area of this person --
        box_h = y2 - y1
        crop_bottom = int(y1 + box_h * FACE_BOX_TOP_FRACTION)
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(width, x2)
        cy2 = min(height, max(cy1 + 10, crop_bottom))
        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            continue

        crop = frame[cy1:cy2, cx1:cx2]
        small = cv2.resize(crop, (0, 0), fx=scale, fy=scale) if scale != 1.0 else crop
        if small.shape[0] < 20 or small.shape[1] < 20:
            small = crop  # too small to shrink further; use full size
            local_scale = 1.0
        else:
            local_scale = scale

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model=FACE_MODEL)
        person["face_checked_at"] = now
        if not locations:
            continue

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            continue

        # Biggest face in the crop is the person we are looking at.
        best = max(
            range(len(locations)),
            key=lambda i: (locations[i][2] - locations[i][0]) * (locations[i][1] - locations[i][3]),
        )
        name, status, distance = _match_encoding(encodings[best])

        top, right, bottom, left = locations[best]
        abs_box = (
            cx1 + int(left / local_scale),
            cy1 + int(top / local_scale),
            cx1 + int(right / local_scale),
            cy1 + int(bottom / local_scale),
        )

        # -- step 3: vote before trusting --
        votes = person["face_votes"]
        votes[name] = votes.get(name, 0) + 1
        person["face_box"] = abs_box

        if votes[name] >= FACE_VOTES_REQUIRED:
            person["face_name"] = name
            person["face_status"] = status
            person["face_confirmed_at"] = now
            person["face_votes"] = {}  # start clean for the next check cycle

        results.append({
            "name": name,
            "status": status,
            "distance": distance,
            "box": abs_box,
            "identity": ident_id,
            "cached": False,
        })

    return results


def adaptive_face_skip(measured_fps, current_skip):
    """
    Feature 12 — spend less time on faces when the machine is struggling.

    If the loop is running slower than FACE_TARGET_FPS we check faces on
    fewer frames; when it speeds up again we check more often. Detection and
    tracking keep full speed either way, so nobody stops being tracked.
    """
    if not FACE_ADAPTIVE_SKIP or measured_fps <= 0:
        return current_skip
    if measured_fps < FACE_TARGET_FPS * 0.8:
        return min(FACE_MAX_SKIP, current_skip + 1)
    if measured_fps > FACE_TARGET_FPS * 1.2:
        return max(1, current_skip - 1)
    return current_skip
