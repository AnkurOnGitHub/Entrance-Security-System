"""
dashboard.py — Feature 18: the live web dashboard.

Runs in a background thread and only READS shared state, so it
cannot slow down or interfere with the camera loop.
"""

import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from config import (
    DASHBOARD_ENABLED,
    DASHBOARD_HISTORY_LIMIT,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DASHBOARD_STREAM_QUALITY,
    DASHBOARD_STREAM_WIDTH,
    DASHBOARD_TOKEN,
    EVIDENCE_DIR,
)

# ==============================================================================
# ### FILE: dashboard.py
# Feature 18 — Real-time dashboard and alert history (NEW).
#
# A small web page served from the device itself. It shows:
#   - the live annotated video (the same frame you would see in the window)
#   - a status line: people in zone, FPS, doorbell source
#   - the alert history read from the SQLite database (Feature 16)
#   - links to the snapshot and clip for each alert (Feature 15)
#
# It runs in a background thread and only ever READS shared state, so it
# cannot slow down or interfere with the camera loop.
#
# SECURITY: this puts a live view of your door on the local network. The
# token check is on by default for that reason. Never port-forward this to
# the internet — if you need remote access, use a VPN.
# ==============================================================================

DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entrance Security</title>
<style>
 :root { color-scheme: dark; }
 body { font-family: system-ui, sans-serif; margin:0; background:#111; color:#eee; }
 header { padding:12px 16px; background:#1b1b1b; border-bottom:1px solid #333; }
 h1 { font-size:18px; margin:0; }
 .wrap { display:flex; flex-wrap:wrap; gap:16px; padding:16px; }
 .col { flex:1 1 380px; min-width:320px; }
 img#live { width:100%; border-radius:8px; background:#000; }
 .status { display:flex; gap:14px; flex-wrap:wrap; margin:10px 0; font-size:14px; }
 .chip { background:#222; padding:6px 10px; border-radius:6px; border:1px solid #333; }
 table { width:100%; border-collapse:collapse; font-size:13px; }
 th,td { text-align:left; padding:6px 8px; border-bottom:1px solid #2a2a2a; vertical-align:top; }
 th { color:#aaa; font-weight:600; }
 .sev-ALERT, .sev-CRITICAL { color:#ff6b6b; font-weight:600; }
 .sev-WARN { color:#ffb84d; }
 .sev-OK, .sev-INFO { color:#8ecf8e; }
 a { color:#7bb6ff; }
</style></head>
<body>
<header><h1>Entrance Security — live</h1></header>
<div class="wrap">
  <div class="col">
    <img id="live" src="STREAM_URL" alt="live view">
    <div class="status" id="status"></div>
  </div>
  <div class="col">
    <h3 style="margin:0 0 8px">Alert history</h3>
    <table id="events"><thead><tr>
      <th>Time</th><th>Type</th><th>Detail</th><th>Evidence</th>
    </tr></thead><tbody></tbody></table>
  </div>
</div>
<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
function q(path){ return path + (TOKEN ? (path.includes('?')?'&':'?') + 'token=' + encodeURIComponent(TOKEN) : ''); }
async function refresh(){
  try {
    const s = await (await fetch(q('/api/status'))).json();
    document.getElementById('status').innerHTML =
      '<span class="chip">People in zone: ' + s.in_zone + '</span>' +
      '<span class="chip">Tracked: ' + s.tracked + '</span>' +
      '<span class="chip">FPS: ' + s.fps.toFixed(1) + '</span>' +
      '<span class="chip">Doorbell: ' + s.doorbell + '</span>' +
      '<span class="chip">Uptime: ' + s.uptime + '</span>';
    const rows = await (await fetch(q('/api/alerts'))).json();
    const body = document.querySelector('#events tbody');
    body.innerHTML = rows.map(r => {
      const links = [];
      if (r.snapshot) links.push('<a href="' + q('/evidence?f=' + encodeURIComponent(r.snapshot)) + '" target="_blank">photo</a>');
      if (r.clip)     links.push('<a href="' + q('/evidence?f=' + encodeURIComponent(r.clip))     + '" target="_blank">video</a>');
      return '<tr><td>' + r.ts_iso.replace('T',' ') + '</td>' +
             '<td class="sev-' + r.severity + '">' + r.kind + '</td>' +
             '<td>' + (r.message || '') + '</td>' +
             '<td>' + (links.join(' · ') || '—') + '</td></tr>';
    }).join('');
  } catch (e) { /* the camera may be restarting; try again next tick */ }
}
refresh(); setInterval(refresh, 2000);
</script>
</body></html>
"""


class Dashboard:
    """
    Background web server showing the live view and the alert history.

    The camera loop calls update_frame() and update_status(); everything
    else happens on the server thread. Shared state is guarded by a lock and
    the server never writes anything the loop depends on.
    """

    def __init__(self, store=None, host=None, port=None, token=None, enabled=None):
        self.enabled = DASHBOARD_ENABLED if enabled is None else enabled
        self.store = store
        self.token = DASHBOARD_TOKEN if token is None else token
        self.host = DASHBOARD_HOST if host is None else host
        self.port = DASHBOARD_PORT if port is None else port
        self.started_at = time.time()
        self._frame_jpeg = None
        self._lock = threading.Lock()
        self._status = {"in_zone": 0, "tracked": 0, "fps": 0.0, "doorbell": "-"}
        self.server = None
        self.thread = None
        if self.enabled:
            self._start()

    # -- fed by the camera loop -------------------------------------------
    def update_frame(self, frame):
        """Store the latest annotated frame, already JPEG-encoded."""
        if not self.enabled or frame is None:
            return
        width = DASHBOARD_STREAM_WIDTH
        if width and frame.shape[1] > width:
            scale = width / frame.shape[1]
            frame = cv2.resize(frame, (width, int(frame.shape[0] * scale)))
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), DASHBOARD_STREAM_QUALITY]
        )
        if ok:
            with self._lock:
                self._frame_jpeg = buf.tobytes()

    def update_status(self, **kwargs):
        if not self.enabled:
            return
        with self._lock:
            self._status.update(kwargs)

    def _snapshot_status(self):
        with self._lock:
            status = dict(self._status)
        seconds = int(time.time() - self.started_at)
        status["uptime"] = f"{seconds // 3600}h {(seconds % 3600) // 60}m"
        return status

    def _latest_jpeg(self):
        with self._lock:
            return self._frame_jpeg

    # -- server ------------------------------------------------------------
    def _start(self):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def _authorised(self, query):
                if not dash.token:
                    return True
                return query.get("token", [""])[0] == dash.token

            def _send(self, code, content_type, body, extra=None):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for key, value in (extra or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)
                path = parsed.path.rstrip("/") or "/"

                if not self._authorised(query):
                    self._send(403, "text/plain", b"forbidden - add ?token=...")
                    return

                if path == "/":
                    stream = "/stream"
                    if dash.token:
                        stream += "?token=" + urllib.parse.quote(dash.token)
                    html = DASHBOARD_HTML.replace("STREAM_URL", stream)
                    self._send(200, "text/html; charset=utf-8", html.encode())
                    return

                if path == "/api/status":
                    body = json.dumps(dash._snapshot_status()).encode()
                    self._send(200, "application/json", body)
                    return

                if path == "/api/alerts":
                    limit = int(query.get("limit", [DASHBOARD_HISTORY_LIMIT])[0])
                    rows = dash.store.recent_events(limit=limit) if dash.store else []
                    self._send(200, "application/json", json.dumps(rows).encode())
                    return

                if path == "/evidence":
                    # Only ever serve files from inside the evidence folder.
                    # Without this check a crafted path could read any file
                    # on the device.
                    wanted = query.get("f", [""])[0]
                    base = os.path.abspath(EVIDENCE_DIR)
                    full = os.path.abspath(wanted)
                    if not full.startswith(base + os.sep) or not os.path.isfile(full):
                        self._send(404, "text/plain", b"not found")
                        return
                    kind = "video/mp4" if full.endswith((".mp4", ".avi")) else "image/jpeg"
                    with open(full, "rb") as f:
                        self._send(200, kind, f.read())
                    return

                if path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.end_headers()
                    try:
                        while True:
                            jpeg = dash._latest_jpeg()
                            if jpeg:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                )
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.08)  # ~12 fps is plenty for a browser
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # viewer closed the tab
                    return

                self._send(404, "text/plain", b"not found")

            def log_message(self, fmt, *args):
                pass

        try:
            self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as error:
            print(f"⚠️ Dashboard could not start on port {self.port}: {error}")
            self.enabled = False
            return

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        suffix = f"/?token={self.token}" if self.token else "/"
        print(f"📊 Dashboard: http://{self.host}:{self.port}{suffix}")
        if not self.token:
            print("⚠️ Dashboard token is empty — anyone on this network can watch the camera.")

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
