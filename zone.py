"""
zone.py — Feature 8: the entrance ROI polygon.

Draw it once, save it to entrance_roi.json, and test whether a
person's feet are inside it.
"""

import json
import os

import cv2
import numpy as np

from config import ROI_FILE, SHOW_WINDOW

# ==============================================================================
# ### FILE: zone.py
# Feature 8 — Entrance ROI / zone (added before Feature 9).
# Everything about drawing, saving, loading and testing the entrance polygon.
# Feature 5 (loitering) depends on this, Feature 9 reuses the same
# in-zone / out-of-zone result, and Feature 11 (crowd) now uses it too.
# ==============================================================================

# ---------------------------------------------------------------------------
# Feature 8: Entrance ROI / zone
# ---------------------------------------------------------------------------
# The zone is a polygon (list of x,y points) saved to this file so you only
# have to draw it once. Delete the file, or press "r" while running, to
# redraw it (useful if the camera ever moves).
# (ROI_FILE itself now sits in the config.py block above, with the other
#  settings — the constant is unchanged, only its position moved.)


def save_roi(points):
    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump(points, f)


def load_roi():
    if not os.path.isfile(ROI_FILE):
        return None
    try:
        with open(ROI_FILE, "r", encoding="utf-8") as f:
            points = json.load(f)
        if isinstance(points, list) and len(points) >= 3:
            return [tuple(p) for p in points]
    except (json.JSONDecodeError, OSError):
        pass
    return None


def select_roi_interactive(frame):
    """
    Let the user click points around the entrance area on a still frame.

    Controls shown on screen:
      Left click  = add a corner point
      U           = undo last point
      ENTER       = confirm (needs at least 3 points)
      ESC         = cancel and use the whole frame as the zone
    """
    points = []
    window = "Draw entrance zone - click corners, ENTER to confirm"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(display, p, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, points[i - 1], p, (0, 255, 0), 2)
        if len(points) > 2:
            cv2.line(display, points[-1], points[0], (0, 255, 0), 1)

        cv2.putText(
            display,
            "Click entrance corners | U=undo | ENTER=confirm | ESC=whole frame",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            if len(points) >= 3:
                break
        elif key == ord("u"):
            if points:
                points.pop()
        elif key == 27:  # ESC → whole frame as fallback zone
            h, w = frame.shape[:2]
            points = [(0, 0), (w, 0), (w, h), (0, h)]
            break

    cv2.destroyWindow(window)
    return points


def get_entrance_roi(video_capture):
    """Load a saved zone, or ask the user to draw one on a fresh frame."""
    roi = load_roi()
    if roi is not None:
        print(f"✅ Loaded entrance zone from {ROI_FILE} ({len(roi)} points)")
        return roi

    # Headless boxes (a Pi with no monitor) cannot show the drawing window,
    # so fall back to the whole frame rather than hanging forever.
    if not SHOW_WINDOW:
        print("⚠️ No saved zone and no display — using the whole frame as the zone.")
        print("   Draw a real zone once with SHOW_WINDOW = true, then deploy.")
        return None

    print("No entrance zone saved yet — draw one on the camera view.")
    ret, frame = video_capture.read()
    if not ret:
        print("⚠️ Could not read a frame to draw the zone; using whole frame.")
        return None

    roi = select_roi_interactive(frame)
    save_roi(roi)
    print(f"✅ Entrance zone saved to {ROI_FILE}")
    return roi


def point_in_roi(point, roi):
    """True if (x, y) is inside the entrance zone polygon. No zone = whole frame counts."""
    if not roi:
        return True
    polygon = np.array(roi, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_roi(frame, roi):
    if not roi:
        return
    overlay = frame.copy()
    polygon = np.array(roi, dtype=np.int32)
    cv2.fillPoly(overlay, [polygon], (0, 255, 0))
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(
        frame, "ENTRANCE ZONE", (polygon[0][0], max(20, polygon[0][1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
