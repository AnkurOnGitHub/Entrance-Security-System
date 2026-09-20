"""
config.py — every tunable setting in one place.

Feature 17: these defaults can all be overridden from config.json, so
tuning on site means editing JSON, not editing Python.

IMPORTANT, read this before changing settings at runtime:
config.json is applied ONCE, when this module is first imported. Other
modules then do `from config import LOITER_SECONDS`, which copies the value
at their import time. That ordering is correct for normal use — config.py
always loads first — but it means calling load_config_overrides() again
while the program is running will NOT reach those modules. Change
config.json and restart. For a camera meant to run for days, a restart is
the honest and predictable way to apply new settings.
"""

import json
import os

# ==============================================================================
# ### FILE: config.py
# All tuning numbers live here.
# Covers: Features 5, 6, 7 (original), 9, 10 (added earlier) and the newest
# batch 11-18. Feature 17 is what makes this block overridable from an
# external config.json, so you can tune on site without editing Python.
# ==============================================================================

# ---------------------------------------------------------------------------
# Thresholds — tweak these for demo vs real use
# ---------------------------------------------------------------------------
# Demo uses 30s so judges can see the alert quickly. Set to 120 for 2 minutes.
LOITER_SECONDS = 30
# "Multiple unknown faces" at a quiet entrance. 2 is easy to demo.
CROWD_UNKNOWN_MIN = 2
CROWD_PERSON_MIN = 2
# Ignore a 1-frame blink; only alert after the lens stays blocked this long.
TAMPER_CONFIRM_SECONDS = 1.5
# Don't spam CSV / console with the same alert every frame.
ALERT_COOLDOWN_SECONDS = 8
# After doorbell (key B), that person is treated as a visitor for this window.
BELL_GRACE_SECONDS = 30
# Face recognition is slower than YOLO, so run it every Nth frame.
FACE_PROCESS_EVERY_N = 2

# ---------------------------------------------------------------------------
# Feature 9: robust tracking settings
# ---------------------------------------------------------------------------
# If a person vanishes and comes back within this time, we treat the gap as
# "still standing there" and keep counting their dwell time.
TRACK_LOST_GRACE_SECONDS = 3.0
# How long we remember a person who has disappeared, so we can match them
# again if they reappear. After this we forget them completely.
IDENTITY_TTL_SECONDS = 15.0
# A reappearing person must be within this fraction of the screen diagonal
# from where the lost person was last seen.
REID_MAX_DISTANCE_RATIO = 0.25
# Clothing colour histograms must be at least this similar (0..1).
REID_MIN_HIST_SIMILARITY = 0.45
# Combined position + colour score needed to accept a re-match (0..1).
REID_MIN_SCORE = 0.50
# Someone must be OUT of the entrance zone this long before their loiter
# timer is wiped. Stops a quick side-step from clearing the clock.
ZONE_EXIT_RESET_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Feature 10: real doorbell input
# ---------------------------------------------------------------------------
# Which doorbell to listen to. Change ONLY this line to move from demo to
# real hardware — nothing in the camera loop has to change.
#   "keyboard" → press B in the video window          (demo default)
#   "gpio"     → real push button wired to a Pi pin   (needs RPi.GPIO)
#   "http"     → ESP32 / smart bell / phone sends a web request
#   "file"     → another program creates a trigger file
DOORBELL_SOURCE = "keyboard"

# A real button bounces: one press can read as many presses in a few
# milliseconds. Ignore repeat rings inside this window.
DOORBELL_DEBOUNCE_SECONDS = 1.0

# --- gpio settings (only used when DOORBELL_SOURCE = "gpio") ---
# BCM pin number the button is wired to. Wire: pin → button → GND.
# We use the Pi's internal pull-up, so the pin reads LOW when pressed.
DOORBELL_GPIO_PIN = 17
# True = pressed reads LOW (button to GND, internal pull-up). This is the
# normal, simplest wiring. Set False if your button pulls the pin HIGH.
DOORBELL_GPIO_ACTIVE_LOW = True

# --- http settings (only used when DOORBELL_SOURCE = "http") ---
# The tiny built-in web server listens here. Any device on the same network
# can ring the bell with:  curl -X POST http://<pi-ip>:8099/ring?token=...
DOORBELL_HTTP_HOST = "0.0.0.0"
DOORBELL_HTTP_PORT = 8099
# Shared secret so a random device on the Wi-Fi cannot ring your bell.
# Set to "" to disable the check (fine on an isolated demo network, NOT fine
# on a real home network).
DOORBELL_HTTP_TOKEN = "change-this-token"

# --- file settings (only used when DOORBELL_SOURCE = "file") ---
# Any other program can ring the bell by creating this file. We delete it
# after reading, so it acts like a one-shot signal.
DOORBELL_TRIGGER_FILE = "doorbell_ring.trigger"

ROI_FILE = "entrance_roi.json"

photos_folder = "photos"
# NOTE: known_face_encodings / known_face_names used to live here. They are
# runtime state, not settings, so they now belong to faces.py.

# ---------------------------------------------------------------------------
# Feature 11: better crowd / unusual gathering detection
# ---------------------------------------------------------------------------
# Count only people standing INSIDE the entrance zone, and require the
# situation to last this long before calling it a gathering. A group walking
# past the door takes about a second; a real gathering stands there.
CROWD_SUSTAIN_SECONDS = 4.0
# If a registered resident is present, a group is almost certainly a normal
# visit (family, guests). Set False to alert on big groups regardless.
CROWD_IGNORE_IF_RESIDENT_PRESENT = True
# People who rang the bell are visitors, not a gathering.
CROWD_IGNORE_BELL_RUNG = True

# ---------------------------------------------------------------------------
# Feature 12: face-recognition performance
# ---------------------------------------------------------------------------
# Search for faces only inside YOLO person boxes instead of the whole frame.
# Much faster, and it removes false faces from posters or reflections.
FACE_SEARCH_IN_PERSON_BOXES = True
# How much of the person box (from the top) to search for a face.
FACE_BOX_TOP_FRACTION = 0.45
# Scale the search image down by this factor. 0.25 was the original.
FACE_DOWNSCALE = 0.25
# "hog" is fast on CPU. "cnn" is far more accurate but needs a GPU.
FACE_MODEL = "hog"
# Matching strictness. Lower = stricter. 0.5 was the original.
FACE_TOLERANCE = 0.5
# How many agreeing looks before we trust a name (cuts flicker/false names).
FACE_VOTES_REQUIRED = 2
# Once an identity is confirmed, skip re-recognising them for this long.
# This is the single biggest speed win: a person standing at the door is
# recognised once, not 300 times.
FACE_CACHE_SECONDS = 20.0
# Adapt the frame skip to keep the loop responsive. If measured FPS drops
# below the target, faces are checked less often.
FACE_ADAPTIVE_SKIP = True
FACE_TARGET_FPS = 12.0
FACE_MAX_SKIP = 8

# ---------------------------------------------------------------------------
# Feature 13: better camera-tamper detection
# ---------------------------------------------------------------------------
# Absolute limits (the original four checks, kept as-is in spirit).
TAMPER_DARK_MEAN = 18
TAMPER_BRIGHT_MEAN = 245
TAMPER_FLAT_STD = 10
TAMPER_BLUR_VARIANCE = 12
# Baseline learning: how fast the "normal scene" picture adapts. Small value
# = slow adaptation, which absorbs sunset but still catches a sudden change.
TAMPER_BASELINE_ALPHA = 0.02
# How different the scene must be from the baseline to count as "camera
# moved / view changed" (0..1, higher = more different required).
TAMPER_SCENE_CHANGE_THRESHOLD = 0.55
# A moved camera must stay changed this long before alerting.
TAMPER_SCENE_CONFIRM_SECONDS = 3.0
# Do not judge scene change until the baseline has had time to settle.
TAMPER_BASELINE_WARMUP_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Feature 14: centralized alert manager
# ---------------------------------------------------------------------------
# Per-alert-type cooldowns. Anything not listed uses ALERT_COOLDOWN_SECONDS.
ALERT_COOLDOWNS = {
    "LOITERING": 20,
    "TAMPER": 30,
    "CROWD": 30,
    "UNKNOWN_FACE": 15,
    "BELL": 1,
}
# Print every fired alert to the console.
ALERT_PRINT = True

# ---------------------------------------------------------------------------
# Feature 15: evidence snapshots and clips
# ---------------------------------------------------------------------------
EVIDENCE_ENABLED = True
EVIDENCE_DIR = "evidence"
# Save a JPEG of the moment the alert fired.
EVIDENCE_SNAPSHOTS = True
EVIDENCE_JPEG_QUALITY = 85
# Save a short video too. Needs a working video writer; turn off on very
# weak hardware.
EVIDENCE_CLIPS = True
# Seconds of video kept BEFORE the alert (this is why a rolling buffer
# exists — otherwise you only ever see the aftermath).
EVIDENCE_PRE_SECONDS = 5.0
EVIDENCE_POST_SECONDS = 8.0
# Assumed FPS for the written clip if the real FPS is not measurable yet.
EVIDENCE_FALLBACK_FPS = 12.0
# Delete evidence older than this many days (0 = keep forever).
EVIDENCE_RETENTION_DAYS = 14

# ---------------------------------------------------------------------------
# Feature 16: SQLite database + CSV export
# ---------------------------------------------------------------------------
DB_ENABLED = True
DB_FILE = "events.db"
# Keep writing the original daily CSV as well, so nothing you already built
# on top of it breaks.
CSV_ENABLED = True

# ---------------------------------------------------------------------------
# Feature 18: real-time dashboard
# ---------------------------------------------------------------------------
DASHBOARD_ENABLED = True
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8080
# Token required as ?token=... on every dashboard request. Empty = no check.
# Leaving this empty puts your live camera feed on the network unprotected.
DASHBOARD_TOKEN = "change-this-token"
# Live stream quality/size. Lower = less network and CPU.
DASHBOARD_STREAM_WIDTH = 640
DASHBOARD_STREAM_QUALITY = 70
DASHBOARD_HISTORY_LIMIT = 100

# ---------------------------------------------------------------------------
# Runtime / deployment
# ---------------------------------------------------------------------------
# Set False on a headless Raspberry Pi with no monitor attached. The camera
# loop then runs without any GUI window (use the dashboard to watch).
SHOW_WINDOW = True
CAMERA_INDEX = 0
CAMERA_WIDTH = 0   # 0 = leave the camera default
CAMERA_HEIGHT = 0
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.45
YOLO_TRACKER = "botsort.yaml"

# ---------------------------------------------------------------------------
# Feature 17: external config.json
# ---------------------------------------------------------------------------
# Every UPPERCASE setting above can be overridden from this file, so tuning
# on site is editing JSON, not editing Python. A template is written on
# first run. Only names that already exist are applied, and the type must
# match, so a typo in the JSON cannot silently create a dead setting.
CONFIG_FILE = "config.json"


def _config_setting_names():
    """Every UPPERCASE module-level name that is a plain setting value."""
    names = []
    for name, value in list(globals().items()):
        if not name.isupper():
            continue
        if name in ("CONFIG_FILE",):
            continue
        if isinstance(value, (int, float, str, bool, dict, list)):
            names.append(name)
    return sorted(names)


def write_config_template(path=None):
    """Write a config.json holding the current defaults, for on-site editing."""
    path = path or CONFIG_FILE
    data = {name: globals()[name] for name in _config_setting_names()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def load_config_overrides(path=None, verbose=True):
    """
    Read config.json and override the settings above.

    Safety rules, on purpose:
      - a key that does not already exist is IGNORED (catches typos)
      - a value of the wrong type is IGNORED (catches "30" instead of 30)
      - a missing or broken file is fine — we just use the defaults
    Returns the list of names actually changed.
    """
    path = path or CONFIG_FILE
    if not os.path.isfile(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as error:
        if verbose:
            print(f"⚠️ Could not read {path}: {error} — using built-in defaults.")
        return []

    if not isinstance(data, dict):
        return []

    changed = []
    for key, value in data.items():
        if key not in globals() or not key.isupper():
            if verbose:
                print(f"⚠️ config.json: unknown setting '{key}' ignored.")
            continue
        current = globals()[key]
        # bool is a subclass of int, so check it first.
        if isinstance(current, bool) and not isinstance(value, bool):
            if verbose:
                print(f"⚠️ config.json: '{key}' must be true/false — ignored.")
            continue
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                if verbose:
                    print(f"⚠️ config.json: '{key}' must be a number — ignored.")
                continue
        elif isinstance(current, str) and not isinstance(value, str):
            if verbose:
                print(f"⚠️ config.json: '{key}' must be text — ignored.")
            continue
        elif isinstance(current, dict) and not isinstance(value, dict):
            continue
        elif isinstance(current, list) and not isinstance(value, list):
            continue

        if value != current:
            globals()[key] = value
            changed.append(key)
    return changed


# Applied at import time, so that even default arguments further down this
# file (which are bound when Python reads the def line) pick up your values.
_CONFIG_CHANGED = load_config_overrides(verbose=False)
