"""
main.py — the camera loop, and the entry point.

Run it exactly as before:

    python main.py

Optional tools:
    python main.py --write-config          write a fresh config.json
    python main.py --export-csv out.csv    export the event history
    python main.py --benchmark 20          measure real FPS on this machine

Each frame, in this order:
    tracking  → resolve YOLO boxes into stable identities   (Feature 9)
    doorbell  → poll whichever source is configured         (Feature 10)
    faces     → recognise, cached per identity              (Features 1-4, 12)
    zone      → draw the entrance area                      (Feature 8)
    loitering → the rule that lives in this file            (Feature 5)
    tamper    → blocked lens or moved camera                (Features 6, 13)
    crowd     → gathering at the door                       (Feature 11)
    alerts    → everything above goes through one manager   (Feature 14),
                which writes the database (16), captures evidence (15)
                and feeds the dashboard (18)
"""

import argparse
import csv
import math
import os
import time
from collections import deque
from datetime import datetime

import cv2

try:
    from ultralytics import YOLO  # YOLOv8: detects + tracks people (class 0)
except ImportError:  # pragma: no cover
    YOLO = None

from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONFIG_FILE,
    CSV_ENABLED,
    EVIDENCE_FALLBACK_FPS,
    FACE_PROCESS_EVERY_N,
    FACE_SEARCH_IN_PERSON_BOXES,
    IDENTITY_TTL_SECONDS,
    LOITER_SECONDS,
    ROI_FILE,
    SHOW_WINDOW,
    TRACK_LOST_GRACE_SECONDS,
    YOLO_CONF,
    YOLO_MODEL,
    YOLO_TRACKER,
    _CONFIG_CHANGED,
)
from config import _CONFIG_CHANGED, write_config_template

# --- the project's own modules --------------------------------------------
from alerts import AlertManager
from crowd import CrowdWatcher
from dashboard import Dashboard
from doorbell import KeyboardDoorbell, create_doorbell_source
from evidence import EvidenceRecorder
from faces import (
    adaptive_face_skip,
    identify_faces,
    identify_faces_in_boxes,
    known_face_names,
    load_known_faces,
)
from storage import EventStore
from tamper import TamperWatcher
from tracking import (
    find_lost_identity,
    identity_has_rung_bell,
    new_identity,
    person_signature,
    update_identity,
)
from zone import draw_roi, get_entrance_roi, point_in_roi, save_roi, select_roi_interactive


# ==============================================================================
# ### FILE: main.py  (part 1 of 2 — helpers)
# Shared drawing helpers used by every feature. These stay with the camera
# loop because they are about the live window, not about one feature's logic.
#
# Note what MOVED OUT in this batch: log_event() used to write the CSV from
# everywhere. That job now belongs to the AlertManager (Feature 14), so the
# rest of the program never touches the CSV writer directly.
# ==============================================================================

def draw_banner(frame, text, y, color):
    """Full-width alert strip at the top of the live window (easy to show in a demo)."""
    cv2.rectangle(frame, (8, y - 28), (frame.shape[1] - 8, y + 8), color, -1)
    cv2.putText(
        frame,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def log_event(writer, name, status, event):
    """
    CSV row. Event is FACE / LOITERING / TAMPER / CROWD / BELL.

    KEPT for compatibility: anything you already wrote against this helper
    still works. New code should call the AlertManager instead, which writes
    this same CSV row plus the database row and the evidence.
    """
    writer.writerow(
        [name, datetime.now().strftime("%H:%M:%S"), status, event]
    )


def handle_doorbell_ring(identities, ids_in_zone, alerts, now):
    """
    Feature 10 — what happens when the bell rings, wherever it came from.

    Everyone standing in the entrance zone right now is marked as a genuine
    visitor, which pauses their Feature 5 loitering timer and takes them out
    of the Feature 11 crowd count.
    """
    for ident_id in ids_in_zone:
        identities[ident_id]["bell_time"] = now
    who = ",".join(f"#{i}" for i in sorted(ids_in_zone)) or "nobody in zone"
    alerts.event(
        "BELL", f"Doorbell rung ({who})", subject="DOORBELL",
        severity="OK", capture=False,
    )


def open_camera():
    """Open the webcam and apply any requested resolution."""
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if CAMERA_WIDTH:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    if CAMERA_HEIGHT:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return capture


class FpsMeter:
    """Rolling average FPS — drives the adaptive face skip and the dashboard."""

    def __init__(self, window=30):
        self.times = deque(maxlen=window)
        self.last = None

    def tick(self, now=None):
        now = now or time.time()
        if self.last is not None:
            self.times.append(now - self.last)
        self.last = now
        if not self.times:
            return 0.0
        average = sum(self.times) / len(self.times)
        return 1.0 / average if average > 0 else 0.0


# ==============================================================================
# ### FILE: main.py  (part 2 of 2 — the camera loop)
# This is where every feature comes together, in this order each frame:
#   Feature 9  → resolve YOLO boxes into stable identities
#   Feature 10 → poll the doorbell, whatever source it is
#   Feature 12 → face auth, cached per identity, searched inside person boxes
#   Feature 8  → draw the entrance zone
#   Feature 5  → loitering, zone-limited and dropout-proof
#   Feature 13 → camera tamper, including "camera moved"
#   Feature 11 → crowd / gathering, zone-limited and time-confirmed
#   Feature 14 → every alert above goes through the AlertManager, which
#                handles cooldowns, the database (16), evidence (15) and
#                the dashboard feed (18)
# ==============================================================================

def main():
    # Feature 17: settings were already loaded at import. Write a template
    # on first run so there is something to edit, and say what changed.
    if not os.path.isfile(CONFIG_FILE):
        write_config_template()
        print(f"📝 Wrote {CONFIG_FILE} — edit it to tune without touching code.")
    elif _CONFIG_CHANGED:
        print(f"📝 config.json overrode: {', '.join(_CONFIG_CHANGED)}")

    load_known_faces()
    print("\nRegistered people:")
    print(sorted(set(known_face_names)))

    # Feature 16: database first, because the alert manager writes into it.
    store = EventStore()
    # Feature 15: evidence recorder (rolling buffer for pre-alert video).
    evidence = EvidenceRecorder()
    removed = evidence.purge_old()
    if removed:
        print(f"🧹 Removed {removed} old evidence files")

    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_file = None
    writer = None
    if CSV_ENABLED:
        csv_file = open(current_date + ".csv", "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(["Name", "Time", "Status", "Event"])

    # Feature 14: one manager for every alert in the program.
    alerts = AlertManager(store=store, evidence=evidence, csv_writer=writer)
    # Feature 18: dashboard reads the same database.
    dashboard = Dashboard(store=store)

    # Feature 10: build the doorbell BEFORE the camera, so a bad setting is
    # reported early rather than after the camera window opens.
    doorbell = create_doorbell_source()
    print(f"Doorbell source: {doorbell.name}")

    if YOLO is None:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return

    # Nano YOLO is small enough for CPU webcam. class 0 = person in COCO.
    # First run downloads yolov8n.pt into the project folder.
    print("Loading YOLOv8 person detector...")
    yolo = YOLO(YOLO_MODEL)

    video_capture = open_camera()
    if not video_capture.isOpened():
        print("❌ Camera could not be accessed")
        doorbell.close()
        dashboard.close()
        store.close()
        return

    # Feature 8: get (or draw) the entrance zone before the main loop starts.
    entrance_roi = get_entrance_roi(video_capture)

    # Feature 9 state.
    # identities[ident_id] = our own stable record of one person.
    # track_to_ident maps YOLO's fragile track id → our stable identity id.
    identities = {}
    track_to_ident = {}
    next_ident_id = 1

    # Feature 13 and 11 keep their own small state objects.
    tamper_watcher = TamperWatcher()
    crowd_watcher = CrowdWatcher()

    fps_meter = FpsMeter()
    face_skip = FACE_PROCESS_EVERY_N
    last_faces = []  # reuse last face results on frames we skip
    frame_index = 0
    fps = 0.0

    alerts.event(
        "SYSTEM", "Entrance security started", subject="SYSTEM",
        severity="INFO", capture=False,
    )

    print("\nControls: B = ring doorbell | R = redraw zone | T = relearn scene | C = quit")
    print(f"Loitering threshold: {LOITER_SECONDS}s inside the entrance zone")
    print(
        f"Tracking: gaps up to {TRACK_LOST_GRACE_SECONDS}s are bridged; "
        f"lost people remembered for {IDENTITY_TTL_SECONDS}s"
    )

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Camera could not be accessed")
                break

            now = time.time()
            frame_index += 1
            fps = fps_meter.tick(now)
            frame_h, frame_w = frame.shape[:2]
            frame_diag = math.hypot(frame_w, frame_h)

            # Feature 15: the rolling buffer must see the RAW frame, before
            # we draw boxes on it, so evidence is not covered in overlays.
            evidence.set_fps(fps if fps > 1 else EVIDENCE_FALLBACK_FPS)
            evidence.add_frame(frame)

            # Feature 14: start a fresh set of banners for this frame.
            alerts.begin_frame(frame.copy())

            # Mark everyone as not-yet-seen for this frame.
            for person in identities.values():
                person["seen_this_frame"] = False

            # BoT-SORT keeps IDs across frames better than plain ByteTrack when
            # people are briefly hidden. Only persons (class 0).
            yolo_results = yolo.track(
                frame,
                persist=True,
                classes=[0],
                verbose=False,
                conf=YOLO_CONF,
                tracker=YOLO_TRACKER,
            )
            result = yolo_results[0]

            person_boxes = []
            ids_in_zone = set()

            if result.boxes is not None and len(result.boxes) > 0:
                ids = result.boxes.id
                xyxy = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    track_id = int(ids[i].item()) if ids is not None else -1

                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    signature = person_signature(frame, x1, y1, x2, y2)

                    # Feature 8: feet point decides if they are in the zone.
                    feet_point = ((x1 + x2) // 2, y2)
                    in_zone = point_in_roi(feet_point, entrance_roi)

                    # --- Feature 9: resolve this box to a stable identity ---
                    ident_id = track_to_ident.get(track_id) if track_id >= 0 else None
                    recovered = False

                    if ident_id is not None and ident_id not in identities:
                        ident_id = None  # identity expired while the id sat unused

                    if ident_id is not None and identities[ident_id]["seen_this_frame"]:
                        ident_id = None  # two boxes claim one identity → split them

                    if ident_id is None:
                        # New or re-appeared YOLO id. Can we match it to someone
                        # we recently lost? If yes, their timer continues.
                        ident_id = find_lost_identity(
                            identities, now, center, signature, frame_diag
                        )
                        if ident_id is not None:
                            recovered = True
                            identities[ident_id]["recovered"] = True
                            identities[ident_id]["recovered_at"] = now
                        else:
                            ident_id = new_identity(
                                identities, next_ident_id, now, center, signature
                            )
                            next_ident_id += 1

                        if track_id >= 0:
                            track_to_ident[track_id] = ident_id

                    person = identities[ident_id]
                    person["seen_this_frame"] = True
                    update_identity(person, now, center, signature, in_zone)

                    if recovered:
                        print(
                            f"🔄 Track recovered: person #{ident_id} re-matched "
                            f"(dwell kept at {int(person['dwell'])}s)"
                        )

                    if in_zone:
                        ids_in_zone.add(ident_id)

                    person_boxes.append((ident_id, x1, y1, x2, y2, in_zone))

            # ----- Feature 10: poll the doorbell -----
            # One line, whatever the source is. Done here (after identities are
            # resolved, before the loiter check) so a ring in this frame pauses
            # the timer in this same frame, not the next one.
            if doorbell.poll():
                handle_doorbell_ring(identities, ids_in_zone, alerts, now)

            # Forget identities nobody has seen for a long time.
            for ident_id in list(identities.keys()):
                if now - identities[ident_id]["last_seen"] > IDENTITY_TTL_SECONDS:
                    identities.pop(ident_id, None)
            # Clean up stale YOLO id → identity links.
            for track_id in list(track_to_ident.keys()):
                if track_to_ident[track_id] not in identities:
                    track_to_ident.pop(track_id, None)

            # ----- Features 1-4 + 12: face auth -----
            # The skip rate adapts to how fast this machine actually is.
            face_skip = adaptive_face_skip(fps, face_skip)
            if frame_index % max(1, face_skip) == 0:
                if FACE_SEARCH_IN_PERSON_BOXES and person_boxes:
                    last_faces = identify_faces_in_boxes(
                        frame, person_boxes, identities, now
                    )
                else:
                    last_faces = identify_faces(frame)

            unknown_count = sum(1 for f in last_faces if f["status"] != "AUTHENTIC")
            authentic_count = sum(1 for f in last_faces if f["status"] == "AUTHENTIC")

            # Draw original AUTHENTIC (green) / UNKNOWN (red) boxes.
            for face in last_faces:
                left, top, right, bottom = face["box"]
                if face["status"] == "AUTHENTIC":
                    color = (0, 255, 0)
                    label = f"{face['name']} - AUTHENTIC"
                else:
                    color = (0, 0, 255)
                    label = "ALERT - UNKNOWN PERSON"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame, label, (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )

                # Feature 14: the manager de-duplicates by subject, so a
                # resident standing at the door logs once, not every frame.
                subject = face.get("identity", face["name"])
                if face["status"] == "AUTHENTIC":
                    alerts.event(
                        "FACE", f"{face['name']} recognised at the door",
                        subject=f"{face['name']}", severity="OK", capture=False,
                    )
                elif not face.get("cached"):
                    alerts.report(
                        "UNKNOWN_FACE", True,
                        "UNKNOWN PERSON at the entrance",
                        subject=f"PERSON#{subject}", severity="WARN",
                        banner=False,  # the red box already says it
                    )

            # ----- Feature 8: draw the entrance zone -----
            draw_roi(frame, entrance_roi)

            # ----- Feature 5: loitering (zone only, stable identities) -----
            for ident_id, x1, y1, x2, y2, in_zone in person_boxes:
                person = identities[ident_id]

                if not in_zone:
                    # Outside the zone — plain box. Timer frozen, not erased.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
                    if person["dwell"] > 0:
                        cv2.putText(
                            frame, f"#{ident_id} paused {int(person['dwell'])}s",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                        )
                    continue

                dwell = person["dwell"]
                bell_ok = identity_has_rung_bell(person, now)

                color = (255, 180, 0)
                label = f"#{ident_id}  {int(dwell)}s/{LOITER_SECONDS}s"

                if bell_ok:
                    color = (0, 200, 255)
                    label = f"#{ident_id}  BELL OK"
                elif dwell >= LOITER_SECONDS:
                    color = (0, 0, 255)
                    label = f"LOITERING  #{ident_id}  {int(dwell)}s"
                    who = person.get("face_name") or "unknown person"
                    alerts.report(
                        "LOITERING", True,
                        f"LOITERING: {who} at entrance {int(dwell)}s, no doorbell",
                        subject=f"PERSON#{ident_id}", severity="ALERT",
                        meta={"dwell": round(dwell, 1), "identity": ident_id},
                    )

                # Show for a few seconds that this person's track was recovered —
                # nice proof during a demo that the timer survived the dropout.
                if person["recovered"]:
                    if now - person["recovered_at"] < 4.0:
                        label += "  [REACQ]"
                    else:
                        person["recovered"] = False

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

            # ----- Feature 13: camera tamper (blocked OR moved) -----
            tampered, reason = tamper_watcher.update(frame, now)
            alerts.report(
                "TAMPER", tampered, f"CAMERA TAMPER: {reason}",
                subject="CAMERA", severity="CRITICAL",
                meta={"reason": reason, "similarity": round(tamper_watcher.last_similarity, 3)},
            )

            # ----- Feature 11: crowd / unusual gathering -----
            gathering, crowd_info = crowd_watcher.evaluate(identities, ids_in_zone, now)
            alerts.report(
                "CROWD", gathering,
                f"CROWD: {crowd_info['people']} people at entrance "
                f"({crowd_info['unknown']} unknown) for "
                f"{int(crowd_info['held_for'])}s",
                subject="ENTRANCE", severity="ALERT", meta=crowd_info,
            )

            # People we are still remembering but cannot currently see.
            tracked_but_hidden = sum(
                1 for p in identities.values() if not p["seen_this_frame"]
            )

            hud = (
                f"FPS:{fps:4.1f}  People:{len(person_boxes)}  "
                f"InZone:{len(ids_in_zone)}  Hidden:{tracked_but_hidden}  "
                f"Unknown:{unknown_count}  Bell:{doorbell.name}"
            )
            cv2.putText(
                frame, hud, (12, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )

            # Feature 14: the manager owns the banner list now.
            y = 36
            for text, color in alerts.banners():
                draw_banner(frame, text, y, color)
                y += 44

            # Feature 18: hand the annotated frame to the dashboard.
            dashboard.update_frame(frame)
            dashboard.update_status(
                in_zone=len(ids_in_zone),
                tracked=len(identities),
                fps=fps,
                doorbell=doorbell.name,
            )

            # ----- window and keys (skipped entirely when headless) -----
            if SHOW_WINDOW:
                cv2.imshow("Entrance Security - YOLO + Face Auth", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    break
                # Feature 10: the keyboard doorbell needs the key we just read.
                if isinstance(doorbell, KeyboardDoorbell):
                    doorbell.feed_key(key)
                # Feature 8: redraw the zone on demand (e.g. camera was moved).
                if key == ord("r"):
                    entrance_roi = select_roi_interactive(frame)
                    save_roi(entrance_roi)
                    identities.clear()
                    track_to_ident.clear()
                    tamper_watcher.relearn()
                    print(f"✅ Entrance zone redrawn and saved to {ROI_FILE}")
                # Feature 13: after moving the camera on purpose, relearn.
                if key == ord("t"):
                    tamper_watcher.relearn()
                    print("✅ Scene baseline cleared — relearning normal view")
            else:
                time.sleep(0.001)  # be polite to the CPU on a headless box

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)")
    finally:
        # Everything below must run even if the loop crashed, or we leak a
        # GPIO pin, an open port, a half-written video and an open database.
        alerts.event(
            "SYSTEM", "Entrance security stopped", subject="SYSTEM",
            severity="INFO", capture=False,
        )
        video_capture.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()
        doorbell.close()
        evidence.close()
        dashboard.close()
        store.close()
        print("Shutdown complete.")


# ==============================================================================
# ### FILE: main.py  (extra tools)
# Feature 19 support — things you run OUTSIDE the camera loop.
#   --export-csv   pull the history out of the database for a report
#   --benchmark    measure what this machine can actually do, before tuning
# These exist because "real-camera testing and tuning" needs numbers, and
# guessing at thresholds without numbers is how false alarms happen.
# ==============================================================================

def export_history(path="alert_history.csv", day=None):
    store = EventStore()
    out = store.export_csv(path, day=day)
    rows = len(store.recent_events(limit=1000000, day=day))
    store.close()
    print(f"✅ Exported {rows} events to {out}")
    return out


def benchmark(seconds=20):
    """
    Measure real FPS on THIS machine with YOLO running, so you can choose
    thresholds from measurements instead of guesses.

    Report it gives you:
      - camera FPS with detection on
      - how long one face-recognition pass costs
      - a suggested FACE_PROCESS_EVERY_N
    """
    if YOLO is None:
        print("❌ ultralytics not installed.")
        return
    print(f"Running a {seconds}s benchmark — stand in front of the camera.")
    yolo = YOLO(YOLO_MODEL)
    capture = open_camera()
    if not capture.isOpened():
        print("❌ Camera could not be accessed")
        return

    frames = 0
    face_time = 0.0
    face_runs = 0
    started = time.time()
    while time.time() - started < seconds:
        ret, frame = capture.read()
        if not ret:
            break
        yolo.track(frame, persist=True, classes=[0], verbose=False,
                   conf=YOLO_CONF, tracker=YOLO_TRACKER)
        frames += 1
        if face_recognition is not None and frames % 5 == 0:
            t0 = time.time()
            identify_faces(frame)
            face_time += time.time() - t0
            face_runs += 1
    capture.release()

    elapsed = time.time() - started
    fps = frames / elapsed if elapsed else 0
    per_face = (face_time / face_runs) if face_runs else 0
    print("\n--- BENCHMARK RESULT ---")
    print(f"Detection+tracking FPS : {fps:.1f}")
    print(f"One face pass          : {per_face * 1000:.0f} ms")
    if per_face and fps:
        suggested = max(1, round(per_face * fps / 0.25))
        print(f"Suggested FACE_PROCESS_EVERY_N : {suggested}")
    print("Put these in config.json, then re-run.")


# ==============================================================================
# ### FILE: main.py  (entry point)
# Still `python entrance_security.py` with no arguments, exactly as before.
# The extra flags are optional tools, not a change to normal use.
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrance security camera")
    parser.add_argument("--export-csv", metavar="PATH", nargs="?", const="alert_history.csv",
                        help="export the event history from the database and exit")
    parser.add_argument("--day", metavar="YYYY-MM-DD",
                        help="limit --export-csv to one day")
    parser.add_argument("--benchmark", type=int, nargs="?", const=20, metavar="SECONDS",
                        help="measure FPS on this machine and exit")
    parser.add_argument("--write-config", action="store_true",
                        help="write a fresh config.json from the built-in defaults and exit")
    args = parser.parse_args()

    if args.write_config:
        print(f"✅ Wrote {write_config_template()}")
    elif args.export_csv:
        export_history(args.export_csv, day=args.day)
    elif args.benchmark:
        benchmark(args.benchmark)
    else:
        main()
