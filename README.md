<div align="center">

# 🚪 Entrance Security System

**A smart door-monitoring system that knows the difference between a visitor and a threat.**

Face recognition · Loitering detection · Camera-tamper detection · Crowd detection · Evidence capture · Live dashboard

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/detector-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/vision-OpenCV-5C3EE8.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-64%20passing-brightgreen.svg)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## The problem

Most DIY security cameras alert on **motion**. That means they alert on a cat, a
passing car, a tree in the wind, and your neighbour walking to their own door.
People switch the alerts off within a week, and then the camera is useless.

This system asks a better question: **is this person behaving like a visitor, or
like someone casing the door?**

A visitor walks up, rings the bell, and leaves. Someone with bad intent stands at
the door without ringing, arrives in a group, or covers the camera first. Those
are the things this detects.

---

## ✨ What it does

| | Feature | What it means in practice |
|---|---|---|
| 👤 | **Face recognition** | Knows your household. Green box for residents, red for strangers. |
| ⏱️ | **Loitering detection** | Alerts when someone stands at the door too long **without ringing the bell**. |
| 🎯 | **Entrance zone** | You draw the door area once. People walking past in the background are ignored. |
| 🔗 | **Occlusion-proof tracking** | Hiding for a few seconds does **not** reset the loitering timer. |
| 📷 | **Tamper detection** | Catches a covered lens **and** a camera that has been turned away. |
| 👥 | **Crowd detection** | Alerts on a group standing at the door — not on a group walking past. |
| 🔔 | **Real doorbell input** | Keyboard, Raspberry Pi button, HTTP webhook, or trigger file. One config line. |
| 📸 | **Evidence capture** | Photo + video for every alert, including the **seconds before** it fired. |
| 🗄️ | **Event database** | SQLite history you can query, plus the original CSV log. |
| 📊 | **Live dashboard** | Watch the camera and browse alert history from your phone. |
| ⚙️ | **JSON config** | Tune every threshold without touching Python. |

---

## 🎬 Demo

> Add your own screenshots or a GIF here. Suggested captures:
> `docs/demo-loitering.gif` · `docs/demo-dashboard.png` · `docs/demo-tamper.png`

**The demo worth recording:** stand at the door for 20 seconds, have someone walk
right in front of the camera blocking you completely for 2 seconds, then step back
into view. The timer **continues from 20s** instead of resetting, and the box shows
`[REACQ]`. Most systems fail this — hiding for one second defeats them entirely.

---

## 🚀 Quick start

```bash
git clone https://github.com/yourusername/entrance-security.git
cd entrance-security

pip install -r requirements.txt

mkdir photos                         # add a photo of each resident
# photos/shivam.jpg                  one photo per person
# photos/shivam/front.jpg            OR a folder per person (more accurate)
# photos/shivam/side.jpg

python main.py
```

**On first run:**

1. A `config.json` is created for you to edit.
2. The camera opens and asks you to **draw the entrance zone** — click the corners
   around your door area, then press `ENTER`. Saved, so you only do this once.
3. The dashboard starts at `http://localhost:8080/`.

**Controls in the window:**

| Key | Action |
|---|---|
| `B` | Ring the doorbell (demo) |
| `R` | Redraw the entrance zone |
| `T` | Relearn the scene baseline (after moving the camera on purpose) |
| `C` | Quit |

---

## 🛠️ Command-line tools

```bash
python main.py                          # normal run
python main.py --benchmark 20           # measure real FPS on your machine
python main.py --write-config           # regenerate config.json from defaults
python main.py --export-csv history.csv # export the event history
python main.py --export-csv d.csv --day 2026-09-20
```

Run `--benchmark` **before** tuning. It tells you your actual FPS, how long one
face-recognition pass costs, and a suggested frame-skip value. Guessed thresholds
are the main cause of false alarms.

---

## ⚙️ Configuration

All 72 settings live in `config.json`. Edit it, then **restart**.

```jsonc
{
  "LOITER_SECONDS": 120,          // how long before "loitering" (30 is a demo value)
  "CROWD_SUSTAIN_SECONDS": 4.0,   // a group must stand this long to count
  "FACE_TOLERANCE": 0.5,          // lower = stricter face matching
  "DOORBELL_SOURCE": "gpio",      // keyboard | gpio | http | file
  "DASHBOARD_TOKEN": "your-secret-here",
  "EVIDENCE_RETENTION_DAYS": 14,
  "SHOW_WINDOW": false            // false for a headless Raspberry Pi
}
```

Unknown keys and wrong types are **ignored with a warning** rather than crashing —
a typo at 2am will not take the camera down.

---

## 🔔 Doorbell options

Change one line in `config.json` to move from demo to real hardware.

<table>
<tr><th>Source</th><th>Use case</th><th>Setup</th></tr>
<tr><td><code>keyboard</code></td><td>Demo</td><td>Press <code>B</code></td></tr>
<tr><td><code>gpio</code></td><td>Real button on a Pi</td><td>Wire <code>GPIO17 → button → GND</code>. Internal pull-up, no resistor needed.</td></tr>
<tr><td><code>http</code></td><td>ESP32, smart bell, phone</td><td><code>curl -X POST "http://pi:8099/ring?token=..."</code></td></tr>
<tr><td><code>file</code></td><td>Glue with another script</td><td><code>touch doorbell_ring.trigger</code></td></tr>
</table>

If a source fails to start (no GPIO library, port in use), it **falls back to
keyboard and keeps running** rather than crashing. A camera that keeps watching
with a degraded doorbell beats a camera that refuses to start.

---

## 🏗️ Architecture

Twelve modules, one flat folder, no packages or install step.

```
                        config.py
                            │   (everything reads settings from here)
    ┌──────────┬────────────┼────────────┬──────────┬──────────┐
    │          │            │            │          │          │
 storage    evidence     zone        tracking    faces      tamper
    │          │                        │
    └────┬─────┘                        │
         │                              │
      alerts                         crowd
         │                              │
         └──────────────┬───────────────┘
                        │
                     main.py  ──── dashboard.py ──── doorbell.py
```

| File | Lines | Responsibility |
|---|---:|---|
| `main.py` | 670 | Camera loop, loitering rule, entry point |
| `config.py` | 328 | 72 settings, JSON override |
| `doorbell.py` | 299 | Four doorbell sources, one interface |
| `faces.py` | 292 | Face registration and recognition, with caching |
| `dashboard.py` | 278 | Live web dashboard |
| `evidence.py` | 194 | Snapshots and clips with pre-roll |
| `tracking.py` | 189 | Stable identities that survive tracking loss |
| `tamper.py` | 182 | Blocked lens and moved-camera detection |
| `storage.py` | 168 | SQLite database and CSV export |
| `alerts.py` | 164 | Centralised alert manager |
| `zone.py` | 153 | Entrance polygon |
| `crowd.py` | 91 | Gathering detection (pure logic, no I/O) |

**Two design rules were followed deliberately:**

- **No circular imports.** `alerts.py` does not import `storage.py` or
  `evidence.py` — it is handed those objects when created. Dependency arrows
  point one way, and `alerts.py` can be tested with fakes.
- **`crowd.py` is pure logic.** It touches no camera, screen or disk. It is given
  the facts and answers a question, which is why it is the easiest module to test.

📖 **Full technical notes:** [`SYSTEM_NOTES.txt`](SYSTEM_NOTES.txt) — 2,500 lines
covering every function, the frame lifecycle, data structures, and why each
decision was made.

---

## 🧠 How the clever bits work

<details>
<summary><b>Why hiding for 2 seconds doesn't reset the loitering timer</b></summary>

YOLO gives each detection a track ID, but those IDs are fragile. If a person is
hidden for half a second, the ID vanishes and returns as a **new number**. Naive
systems delete the old state and start counting from zero — so a loiterer only has
to duck out of view every 20 seconds to stay invisible forever.

Two fixes:

1. **Dwell time is added up, not measured from a start time.**
   `dwell += time_since_last_seen`. A 2-second gap is simply counted as 2 seconds
   of standing there. Adding up cannot be reset by a dropout; measuring from a
   start time can.

2. **Re-identification.** When a new track ID appears, it is compared to recently
   lost people using two clues: how close it is to where they vanished, and an HSV
   colour histogram of their torso (clothing). If both agree, the old timer
   continues and `[REACQ]` shows on screen.

</details>

<details>
<summary><b>How it catches a turned camera without false-alarming at sunset</b></summary>

Covering a lens is easy to detect — the image goes dark, flat or blurry. But the
more likely real attack is simply **turning the camera to face a wall**. The
picture stays bright and sharp, so every brightness check passes.

The fix is a **slowly-learned baseline** of the normal view, compared by **edge
structure** rather than brightness.

Edges say *where things are*. Switching a porch light on changes brightness
everywhere, but the door frame and step stay in the same pixels. Turning the
camera moves every edge.

Measured on test scenes:

| Situation | Similarity | Result |
|---|---:|---|
| Same view, camera noise | 0.94 | ✅ quiet |
| Same view, much darker (dusk) | 0.75 | ✅ quiet |
| Genuinely different view | 0.01 | 🚨 alert |
| *Alert threshold* | *0.45* | |

The baseline is **never updated while an alert is active** — otherwise a covered
lens would slowly become the new "normal" and the alert would switch itself off.

</details>

<details>
<summary><b>Why evidence clips include the seconds before the alert</b></summary>

If recording starts when the alert fires, the clip begins *after* the interesting
moment — you get footage of someone already walking away.

A rolling buffer holds the last few seconds of frames in memory at all times. When
an alert fires, those remembered frames are written into the file **first**. The
clip shows the approach, not just the aftermath.

The buffer is fed the **raw** frame, before any boxes are drawn, so evidence
footage is clean rather than covered in debug overlays.

</details>

<details>
<summary><b>How face recognition keeps up with everything else</b></summary>

Recognising a standing visitor 450 times gives the same answer 450 times. Three
savings, stacked:

1. **Cache the result on the tracked identity.** Recognise once, reuse for 20
   seconds. ~95% of the work disappears.
2. **Search only inside person boxes**, not the whole frame. Faces on posters,
   TVs and reflections are never scanned.
3. **Adaptive frame skip.** If measured FPS drops, face checks happen less often.
   Detection and tracking always run at full speed, so nobody stops being tracked.

A name must also win two agreeing votes before it is trusted, so one blurred frame
cannot rename a resident to UNKNOWN.

</details>

---

## 📊 Dashboard

Open `http://<device-ip>:8080/?token=<your-token>` from any device on the network.

- Live annotated video (MJPEG — works in any browser, no plugin)
- Status: people in zone, tracked count, FPS, doorbell source, uptime
- Alert history from the database, with links to each photo and video

Runs on a background thread and only **reads** shared state, so it cannot slow
down the camera loop.

---

## 🔒 Security checklist

Please do these before running this anywhere real.

- [ ] **Change `DASHBOARD_TOKEN`.** The default lets anyone on your network watch
      your front door live.
- [ ] **Change `DOORBELL_HTTP_TOKEN`** if using the HTTP doorbell. Ringing the
      bell clears a loitering timer — an attacker who can ring it can stay
      invisible to the system.
- [ ] **Never port-forward ports 8080 or 8099 to the internet.** Use a VPN
      (WireGuard or Tailscale) for remote access.
- [ ] **Set `EVIDENCE_RETENTION_DAYS`** so the disk cannot fill. A full disk stops
      recording silently — the worst failure mode.

The `/evidence` endpoint is protected against path traversal (verified by test).
Database queries use parameter placeholders, not string formatting.

---

## ⚖️ Privacy and legal

**Please read this before pointing a camera at anyone.**

Face recognition on identifiable people is regulated in many countries, including
under India's **DPDP Act**. Storing face data of people who have not consented can
be unlawful, even at your own front door.

- Get consent from everyone whose photo goes in `photos/`
- Put up a visible notice that the area is monitored
- Do not point the camera at a public road or a neighbour's property
- Decide and document how long footage is kept and who can access it

---

## ⚠️ Honest limitations

Listed openly, because a system that claims no weaknesses is not being truthful
about how it works.

| Limitation | Detail |
|---|---|
| **Re-ID is colour-based** | A clothing histogram, not deep-learning ReID. Two people in similar clothes who swap places during an occlusion can be confused. Right for one quiet doorway; wrong for a crowd. |
| **No liveness check** | A printed photo held to the camera can pass as AUTHENTIC. Anti-spoofing is a project on its own. |
| **Tamper warm-up** | Scene-change detection is off for the first 10 seconds after start while the baseline settles. |
| **Single camera** | No multi-camera support, no central server, no cloud. |
| **LAN-only dashboard** | One shared token, no user accounts. A deliberate scope choice — do not call it secure remote access. |
| **CPU-bound** | ~5–10 FPS on a Raspberry Pi 4 with `yolov8n`. Fine for a doorway; not for a fast corridor. |
| **No push notifications yet** | Alerts reach the screen, log, database and dashboard — but not your phone. See the roadmap. |

---

## 🗺️ Roadmap

- [ ] **Push notifications** (Telegram bot — ~30 lines, hooks into `AlertManager._fire`)
- [ ] Night mode / IR handling
- [ ] Auto-start on boot via `systemd`
- [ ] Package delivery detection (abandoned object)
- [ ] Multi-camera support

**Not planned: custom YOLO training.** We detect one class, `person`, and COCO
already has 250,000+ labelled examples across every pose and lighting condition.
Training on one doorway risks overfitting and performing *worse* at night. If
detection misses people, try `yolov8s.pt` instead — one config change, no training.

---

## 🧪 Testing

64 automated checks covering config validation, database, evidence (including
verifying the pre-roll footage is really there), alert cooldowns, tamper detection
(including that dusk does **not** false-alarm), crowd logic, face caching,
dashboard security, and regression tests for tracking and zones.

```bash
python test_split.py
```

No camera or GPU required — the logic is tested with synthetic frames.

---

## 📦 Requirements

```
opencv-python >= 4.8
numpy >= 1.24
ultralytics >= 8.0
face_recognition >= 1.3
Pillow >= 10.0
RPi.GPIO >= 0.7        # Raspberry Pi only, for the gpio doorbell
```

Tested on Python 3.9+. `yolov8n.pt` downloads automatically on first run
(~6 MB, needs internet once).

> **Note:** `face_recognition` needs `dlib`, which can be awkward to install on
> Windows. On Windows, installing via Anaconda or WSL is usually easier.

---

## 🤝 Contributing

Issues and pull requests are welcome. If you are adding a feature:

1. Put its settings in `config.py`, not hard-coded numbers.
2. Route its alerts through `AlertManager` rather than printing directly.
3. Add a test.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

> You need to add a `LICENSE` file. If MIT suits you, GitHub can generate one:
> **Add file → Create new file → type `LICENSE` → "Choose a license template"**.
> Pick a different license if you prefer; just update this section to match.

---

## 🙏 Built with

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) ·
[OpenCV](https://opencv.org/) ·
[face_recognition](https://github.com/ageitgey/face_recognition) ·
[dlib](http://dlib.net/)

---

<div align="center">

Built for **Smart India Hackathon**.

⭐ Star this repo if you found it useful.

</div>
