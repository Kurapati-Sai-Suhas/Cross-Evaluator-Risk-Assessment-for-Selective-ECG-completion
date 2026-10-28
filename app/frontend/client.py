"""Minimal HTTP client for the local backend (standard library only). The frontend never imports any model code."""
import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8765"


class BackendError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class BackendClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, method, path, payload=None, timeout=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read().decode("utf-8")).get("error", {})
            except (ValueError, AttributeError):
                err = {}
            raise BackendError(err.get("code", f"HTTP_{exc.code}"), err.get("message", str(exc))) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise BackendError("BACKEND_UNREACHABLE", f"cannot reach the backend at {self.base_url}: {exc}") from None

    def health(self):
        return self._call("GET", "/api/health", timeout=5)

    def info(self):
        return self._call("GET", "/api/info", timeout=10)

    def infer(self, payload):
        return self._call("POST", "/api/infer", payload)
