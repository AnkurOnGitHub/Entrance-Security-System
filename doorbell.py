"""
doorbell.py — Feature 10: pluggable doorbell input.

keyboard / gpio / http / file, all behind one poll() call, so the
camera loop never needs to know which one is in use.
"""

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import (
    DOORBELL_DEBOUNCE_SECONDS,
    DOORBELL_GPIO_ACTIVE_LOW,
    DOORBELL_GPIO_PIN,
    DOORBELL_HTTP_HOST,
    DOORBELL_HTTP_PORT,
    DOORBELL_HTTP_TOKEN,
    DOORBELL_SOURCE,
    DOORBELL_TRIGGER_FILE,
)

# ==============================================================================
# ### FILE: doorbell.py
# Feature 10 — Real doorbell input (added before the newest batch).
#
# Why this exists: the camera loop used to check `key == ord("b")` directly.
# That hard-wired the demo keyboard into the middle of the loop, so swapping
# in a real button would have meant editing the loop.
#
# Now every doorbell type follows the same tiny contract:
#     poll()  → True if the bell was rung since the last call
#     close() → release the pin / stop the server
# The loop calls poll() once per frame and does not care which source it is.
# Feature 5 (loitering) and Feature 11 (crowd) both use the result: a ring
# marks that person as a genuine visitor.
# ==============================================================================

class DoorbellSource:
    """
    Base class. A doorbell only has to answer one question:
    "has the bell been rung since I last asked?"

    poll() must be NON-BLOCKING — it is called on every camera frame, so it
    can never wait for input. Each source sets a flag from somewhere else
    (a key, a pin, a web request, a file) and poll() just reads that flag.
    """

    name = "base"

    def __init__(self):
        self._rang = False
        self._last_ring_time = 0.0
        self._lock = threading.Lock()

    def _register_ring(self):
        """
        Called by the real trigger (button, web request, etc).

        Debounce lives here, in ONE place, so every source gets it free.
        A real push button can bounce and report several presses in a few
        milliseconds; we ignore repeats inside DOORBELL_DEBOUNCE_SECONDS.
        """
        now = time.time()
        with self._lock:
            if now - self._last_ring_time < DOORBELL_DEBOUNCE_SECONDS:
                return False
            self._last_ring_time = now
            self._rang = True
            return True

    def poll(self):
        """Read and clear the flag. True = ring happened since last frame."""
        with self._lock:
            rang = self._rang
            self._rang = False
        return rang

    def close(self):
        """Release hardware / stop threads. Safe to call twice."""
        pass


class KeyboardDoorbell(DoorbellSource):
    """
    The demo doorbell: press B in the video window.

    The camera loop already calls cv2.waitKey() once per frame, so instead of
    reading the keyboard again here, the loop hands us the key it got.
    """

    name = "keyboard (press B)"

    def feed_key(self, key):
        """Called by the camera loop with whatever cv2.waitKey returned."""
        if key == ord("b"):
            self._register_ring()


class GpioDoorbell(DoorbellSource):
    """
    Real push button on a Raspberry Pi.

    Wiring (simplest version):
        GPIO pin ──── button ──── GND
    We turn on the Pi's internal pull-up resistor, so the pin sits HIGH and
    drops to LOW when the button is pressed. No extra resistor needed.

    RPi.GPIO is imported INSIDE this class on purpose: a laptop does not have
    it, and we do not want the whole program to crash on import just because
    the GPIO library is missing.
    """

    name = "gpio button"

    def __init__(self, pin=None, active_low=None):
        super().__init__()
        # Read config here, not in the default argument, so that an override
        # loaded from config.json still applies (Feature 17).
        self.pin = DOORBELL_GPIO_PIN if pin is None else pin
        active_low = DOORBELL_GPIO_ACTIVE_LOW if active_low is None else active_low
        self.gpio = None
        try:
            import RPi.GPIO as GPIO  # only exists on a Raspberry Pi
        except ImportError:
            raise RuntimeError(
                "RPi.GPIO not installed — are you running on a Raspberry Pi? "
                "Install with: pip install RPi.GPIO"
            )

        self.gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        pull = GPIO.PUD_UP if active_low else GPIO.PUD_DOWN
        edge = GPIO.FALLING if active_low else GPIO.RISING
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=pull)

        # An interrupt, not a polling loop: the Pi tells US when the button is
        # pressed, so we never miss a quick press between camera frames.
        # bouncetime is the hardware-level debounce; _register_ring adds a
        # second software guard on top.
        GPIO.add_event_detect(
            self.pin,
            edge,
            callback=self._on_edge,
            bouncetime=int(DOORBELL_DEBOUNCE_SECONDS * 1000),
        )

    def _on_edge(self, channel):
        self._register_ring()

    def close(self):
        if self.gpio is not None:
            try:
                self.gpio.remove_event_detect(self.pin)
                self.gpio.cleanup(self.pin)
            except Exception:
                pass  # never let cleanup crash the shutdown
            self.gpio = None


class HttpDoorbell(DoorbellSource):
    """
    Network doorbell: anything that can send a web request can ring it.

    Works with an ESP32/Arduino button, a smart doorbell webhook, a phone
    shortcut, or just curl for testing:

        curl -X POST "http://<ip>:8099/ring?token=change-this-token"

    The server runs in a BACKGROUND THREAD so it never blocks the camera
    loop. It sets the same flag as every other source.

    SECURITY NOTE: without a token, anyone on the same Wi-Fi could ring your
    bell and clear a real intruder's loitering timer. That is why the token
    check below exists. Do not ship this on a real network with the default
    token, and do not expose this port to the open internet — it is meant for
    your local network only.
    """

    name = "http endpoint"

    def __init__(self, host=None, port=None, token=None):
        super().__init__()
        host = DOORBELL_HTTP_HOST if host is None else host
        port = DOORBELL_HTTP_PORT if port is None else port
        self.token = DOORBELL_HTTP_TOKEN if token is None else token
        doorbell = self  # the handler class needs a way back to us

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                # Path looks like /ring?token=xyz — split off the query part.
                path, _, query = self.path.partition("?")
                if path.rstrip("/") not in ("/ring", "/doorbell"):
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"not found")
                    return

                if doorbell.token:
                    supplied = ""
                    for part in query.split("&"):
                        if part.startswith("token="):
                            supplied = part[len("token="):]
                    if supplied != doorbell.token:
                        self.send_response(403)
                        self.end_headers()
                        self.wfile.write(b"bad token")
                        return

                accepted = doorbell._register_ring()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ring" if accepted else b"ignored (debounce)")

            def do_GET(self):
                self._handle()

            def do_POST(self):
                self._handle()

            def log_message(self, fmt, *args):
                pass  # keep the console clean; we print our own messages

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"🔔 Doorbell HTTP server listening on http://{host}:{port}/ring")
        if not self.token:
            print("⚠️ Doorbell HTTP token is empty — anyone on this network can ring it.")

    def close(self):
        if getattr(self, "server", None) is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


class FileDoorbell(DoorbellSource):
    """
    Trigger-file doorbell: any other program rings the bell by creating a file.

        touch doorbell_ring.trigger

    Useful for glue code — a separate MQTT listener, a phone app bridge, or a
    teammate's script can ring the bell without needing to talk to this
    program directly. We delete the file after reading, so one file = one ring.

    This is the only source that does real work inside poll(), but an
    os.path.exists() check per frame is cheap.
    """

    name = "trigger file"

    def __init__(self, path=None):
        super().__init__()
        self.path = DOORBELL_TRIGGER_FILE if path is None else path
        # Clear any leftover file from a previous run, so we don't ring at
        # startup because of an old trigger.
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        print(f"🔔 Doorbell watching for trigger file: {self.path}")

    def poll(self):
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
                self._register_ring()
            except OSError:
                pass
        return super().poll()


def create_doorbell_source(source_name=None):
    """
    Factory: build the doorbell named in config.

    If a real source fails to start (no GPIO library, port already in use),
    we DO NOT crash the whole security camera — we print the reason and fall
    back to the keyboard. A camera that keeps watching with a degraded
    doorbell is far better than a camera that refuses to start.
    """
    source_name = (source_name or DOORBELL_SOURCE or "keyboard").lower()
    try:
        if source_name == "gpio":
            return GpioDoorbell()
        if source_name == "http":
            return HttpDoorbell()
        if source_name == "file":
            return FileDoorbell()
        if source_name != "keyboard":
            print(f"⚠️ Unknown DOORBELL_SOURCE '{source_name}' — using keyboard.")
    except Exception as error:
        print(f"⚠️ Could not start '{source_name}' doorbell: {error}")
        print("   Falling back to keyboard (press B).")
    return KeyboardDoorbell()
