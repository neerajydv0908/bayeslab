"""Small web application using Python's standard HTTP server.

Run locally (the default) or, with public=True, on a hosting service: public
mode listens on every interface, drops clients that stall, limits how many
calculations run at once, and gives each calculation a shorter time limit.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import traceback
from typing import Optional
from urllib.parse import urlsplit

from . import __version__
from .decision import DecisionAnalysis
from .examples import EXAMPLES, load_example
from .inference import Inference
from .enumeration import Enumeration
from .limits import time_limit
from .model import ModelError, Network

WEB_DIR = Path(__file__).with_name("web")
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css",
                "/app.js": "app.js", "/builder.js": "builder.js"}
CONTENT_TYPES = {"html": "text/html", "css": "text/css", "js": "text/javascript"}
POST_ROUTES = ("/api/validate", "/api/query", "/api/utility", "/api/worlds", "/api/sensitivity")
MAX_BODY_BYTES = 1_000_000


@dataclass(frozen=True)
class ServerSettings:
    public: bool = False
    # Seconds one calculation may run; None means no limit.
    time_limit: Optional[float] = 60.0
    # Calculations allowed at once, and how long a request waits for a free slot.
    max_concurrent: int = 4
    queue_wait: float = 5.0


LOCAL = ServerSettings()
PUBLIC = ServerSettings(public=True, time_limit=10.0, max_concurrent=2)


class ServerBusy(Exception):
    """Every calculation slot stayed busy for the whole waiting time."""


class Handler(BaseHTTPRequestHandler):
    server_version = f"BayesLab/{__version__}"
    _replied = False

    @property
    def settings(self):
        return getattr(self.server, "bayeslab_settings", LOCAL)

    def setup(self):
        if self.settings.public:
            self.timeout = 30  # seconds: disconnect clients that stall while sending a request
        super().setup()

    def _respond(self, payload, status=200, content_type="application/json; charset=utf-8", headers=None):
        body = json.dumps(payload, allow_nan=False).encode("utf-8") if isinstance(payload, (dict, list)) else payload
        self._replied = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _guarded(self, route):
        """Run a route so that every request gets a JSON reply, even after a bug."""
        self._replied = False
        try:
            route(urlsplit(self.path).path)
        except ConnectionError:
            pass  # The browser disconnected, so nobody is waiting for a reply.
        except Exception:
            self.log_error("Unexpected error while handling %s %s:\n%s", self.command, self.path, traceback.format_exc())
            if not self._replied:
                self._respond({"error": "Unexpected server error. Details were printed in the terminal running BayesLab."}, 500)

    def do_GET(self):
        self._guarded(self._get)

    def do_POST(self):
        self._guarded(self._post)

    def _get(self, path):
        if path == "/healthz":
            self._respond({"status": "ok"})
        elif path == "/api/config":
            settings = self.settings
            self._respond({"version": __version__, "public": settings.public,
                           "time_limit_seconds": settings.time_limit, "max_request_bytes": MAX_BODY_BYTES})
        elif path == "/api/examples":
            self._respond(EXAMPLES)
        elif path.startswith("/api/examples/"):
            try:
                self._respond(load_example(path.rsplit("/", 1)[-1]))
            except ValueError as exc:
                self._respond({"error": str(exc)}, 404)
        elif path in STATIC_FILES:
            filename = STATIC_FILES[path]
            content_type = CONTENT_TYPES[filename.rsplit(".", 1)[-1]]
            self._respond((WEB_DIR / filename).read_bytes(), content_type=content_type + "; charset=utf-8")
        elif path == "/favicon.ico":
            self._respond(b"", 204, "image/x-icon")
        else:
            self._respond({"error": "Not found."}, 404)

    def _read_json(self):
        try:
            size = int(self.headers.get("Content-Length"))
        except (TypeError, ValueError):
            raise ModelError("Send a JSON request body with a valid Content-Length header.") from None
        if not 0 < size <= MAX_BODY_BYTES:
            raise ModelError("Requests must contain a JSON body smaller than 1 MB.")
        raw = self.rfile.read(size)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ModelError("The request body must be UTF-8 encoded JSON.") from None
        try:
            body = json.loads(text)
        except RecursionError:
            raise ModelError("The request JSON is nested too deeply.") from None
        except json.JSONDecodeError as exc:
            raise ModelError(f"The request body is not valid JSON: {exc.msg} "
                             f"(line {exc.lineno}, column {exc.colno}).") from None
        if not isinstance(body, dict):
            raise ModelError("The request must be a JSON object.")
        if "model" not in body:
            raise ModelError('Include the network as "model" in the request body.')
        return body

    @contextmanager
    def _calculation_slot(self):
        slots = getattr(self.server, "bayeslab_slots", None)
        if slots is None:
            yield
            return
        if not slots.acquire(timeout=self.settings.queue_wait):
            raise ServerBusy
        try:
            yield
        finally:
            slots.release()

    def _post(self, path):
        if path not in POST_ROUTES:
            self._respond({"error": "Not found."}, 404)
            return
        try:
            body = self._read_json()
            with self._calculation_slot(), time_limit(self.settings.time_limit):
                result = self._calculate(path, body)
            self._respond(result)
        except ServerBusy:
            self._respond({"error": "BayesLab is busy with other calculations. Try again in a few seconds."},
                          503, headers={"Retry-After": "5"})
        except (ModelError, ValueError, UnicodeError) as exc:
            self._respond({"error": str(exc)}, 400)

    @staticmethod
    def _calculate(path, body):
        network = Network(body.get("model"))
        if path == "/api/validate":
            return network.summary()
        if path == "/api/query":
            method = body.get("method", "elimination")
            if method not in ("enumeration", "elimination"):
                raise ModelError("Choose enumeration or elimination.")
            engine = Enumeration(network) if method == "enumeration" else Inference(network)
            return engine.query(body.get("event", {}), body.get("evidence", {}), body.get("action"))
        if path == "/api/worlds":
            if network.summary()["joint_assignments"] > 256:
                raise ModelError("The interactive joint table supports at most 256 worlds. Use the Python enumeration API for larger tables.")
            return {"worlds": list(Enumeration(network).worlds(body.get("action")))}
        if path == "/api/sensitivity":
            return DecisionAnalysis(network).sensitivity(body.get("target"), body.get("evidence", {}))
        return DecisionAnalysis(network, body.get("method", "elimination")).analyze(body.get("evidence", {}))


def serve(host="127.0.0.1", port=8000, public=False, time_limit=None, max_concurrent=None):
    """Start BayesLab and block until Ctrl+C.

    time_limit and max_concurrent default to 60 s and 4 locally, and to 10 s
    and 2 calculations in public mode. A time_limit of 0 or less means no limit.
    """
    defaults = PUBLIC if public else LOCAL
    if time_limit is not None and not math.isfinite(time_limit):
        raise ValueError("The time limit must be a finite number of seconds (0 means no limit).")
    if time_limit is None:
        time_limit = defaults.time_limit
    elif time_limit <= 0:
        time_limit = None
    settings = ServerSettings(public=public, time_limit=time_limit,
                              max_concurrent=max_concurrent or defaults.max_concurrent)
    server = ThreadingHTTPServer((host, port), Handler)
    server.bayeslab_settings = settings
    server.bayeslab_slots = threading.BoundedSemaphore(settings.max_concurrent)
    limit = f"{settings.time_limit:g} s" if settings.time_limit else "none"
    if public:
        print(f"BayesLab {__version__} is serving publicly on {host}:{server.server_port} "
              f"(time limit per calculation: {limit}; {settings.max_concurrent} at a time).", flush=True)
    else:
        print(f"BayesLab is running at http://{host}:{server.server_port} (time limit per calculation: {limit})",
              flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBayesLab stopped.")
    finally:
        server.server_close()
