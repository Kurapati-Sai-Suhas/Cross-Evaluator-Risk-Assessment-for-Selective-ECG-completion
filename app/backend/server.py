"""Local JSON API for the research demo (standard library only; runs in the project's pinned .venv).

    .\\.venv\\Scripts\\python.exe -m app.backend.server [--port 8765]

GET  /api/health   liveness and asset-integrity status
GET  /api/info     model versions, hashes, threshold, input schema and the 258 lead sets
POST /api/infer    one request in the schema of app/backend/schema.py; returns the decision (and, only when released, the
                   reconstruction). Errors: 400 malformed JSON, 413 too large, 415 not JSON, 422 refused input,
                   500 internal error (nothing released).
It binds to 127.0.0.1 only, refuses to start if any frozen asset's SHA-256 differs, keeps uploads in memory, and never logs
request bodies or waveform values.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import integrity, schema

MAX_BODY = 4_000_000


def make_handler(service, log=None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ECGResearchDemo/1"
        sys_version = ""

        def log_message(self, fmt, *args):                # method, path and status only; never the body
            if log:
                log(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}".strip())

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, err, message):
            self._send(code, {"error": {"code": err, "message": message}, "released": False})

        def _discard_body(self):                          # before an early reply: closing with unread data sends a reset
            try:                                          # that can destroy the reply before the client reads it
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return
            if 0 < n <= MAX_BODY:
                self.rfile.read(n)                        # read and dropped; never parsed, kept or logged

        def do_GET(self):
            if self.path == "/api/health":
                self._send(200, {"status": "ok", "assets_verified": service.integrity.get("ok")})
            elif self.path == "/api/info":
                self._send(200, service.info())
            else:
                self._error(404, "NOT_FOUND", "unknown endpoint")

        def do_POST(self):
            if self.path != "/api/infer":
                self._discard_body()
                return self._error(404, "NOT_FOUND", "unknown endpoint")
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                self._discard_body()
                return self._error(415, "UNSUPPORTED_MEDIA_TYPE", "send the request as application/json")
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._error(400, "INVALID_REQUEST", "bad Content-Length")
            if n <= 0 or n > MAX_BODY:
                return self._error(413, "INPUT_TOO_LARGE", f"the request body must be 1 to {MAX_BODY:,} bytes")
            raw = self.rfile.read(n)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._error(400, "INVALID_JSON", "the request body is not valid JSON")
            finally:
                del raw                                   # the upload is not kept
            try:
                self._send(200, service.infer(payload))
            except schema.InputError as exc:
                self._error(422, exc.code, str(exc))
            except Exception:                             # fail closed; no traceback or data echoed back
                self._error(500, "INTERNAL", "inference failed; nothing was released")

    return Handler


def make_server(service, host="127.0.0.1", port=8765, log=None):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("the research demo binds to the local machine only")
    return ThreadingHTTPServer((host, port), make_handler(service, log))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ECG capstone research-demo backend (local only; not a medical device)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-preload", action="store_true", help="load the models on first use instead of at start-up")
    a = ap.parse_args(argv)
    from .service import InferenceService
    try:
        service = InferenceService(verify_assets=True, preload=not a.no_preload)
    except integrity.IntegrityError as exc:
        print(f"REFUSING TO START: frozen asset check failed: {exc}", file=sys.stderr)
        return 2
    srv = make_server(service, port=a.port, log=lambda m: print(m, flush=True))
    print(f"ECG research-demo backend on http://127.0.0.1:{a.port} (assets verified; Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
