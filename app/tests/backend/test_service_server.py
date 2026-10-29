"""Service decision flow and the local JSON API, with the gate and the reconstruction replaced by stand-ins (MOCKED).

The real frozen pipeline is exercised separately in test_real_pipeline.py.
    .\\.venv\\Scripts\\python.exe -m pytest app/tests/backend -q
"""
import http.client
import json
import threading

import numpy as np
import pandas as pd
import pytest

from app.backend import schema, server
from app.backend.pipeline import REPO, demo
from app.backend.service import InferenceService
from src.data import ptbxl_io as io
from src.eval import d1 as d1mod
from src.gate import features as gf
from src.gate import m4b1

from .test_schema_integrity import OBS, request


def fake_gate(decision, seen=None):
    def gate(x, observed):
        if seen is not None:
            seen.append((x.copy(), list(observed)))
        return {"features": {"g1_n_observed": len(observed)}, "raw_score": 0.1,
                "calibrated_score": 0.03 if decision == "ACCEPT" else 0.2, "threshold_repr": "0.05927309439311807",
                "decision": decision, "model_sha256": "m", "calibrator_sha256": "c"}
    return gate


def fake_reconstruct(calls, corrupt=None):
    def rec(x, observed):
        calls.append(list(observed))
        x_hat = demo.l2g_input(x).copy()                 # a stand-in that copies the model input exactly
        if corrupt:
            corrupt(x_hat)
        return x_hat, {"device": "cpu", "copy_through_violations": 0, "g_sha256": "g", "batch_rows": 1}
    return rec


@pytest.fixture
def svc():
    return InferenceService(verify_assets=False, preload=False)


def test_accept_triggers_reconstruction_and_release(svc, monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "gate_score", fake_gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", fake_reconstruct(calls))
    res = svc.infer(request())
    assert calls == [OBS] and res["outcome"] == "RELEASED" and res["released"] and res["reconstruction_computed"]
    vals = np.array(res["reconstruction"]["values"])
    assert vals.shape == (12, 2048) and res["release_check"]["passed"]


def test_abstain_never_reconstructs(svc, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("reconstruct called after ABSTAIN")
    monkeypatch.setattr(demo, "gate_score", fake_gate("ABSTAIN"))
    monkeypatch.setattr(demo, "reconstruct", must_not_run)
    res = svc.infer(request())
    assert res["outcome"] == "ABSTAINED" and not res["released"] and res["reconstruction"] is None
    assert res["reconstruction_computed"] is False and res["release_check"] is None


@pytest.mark.parametrize("corrupt,expect", [
    (lambda xh: xh.__setitem__((demo.LEAD_INDEX["I"], 10), xh[demo.LEAD_INDEX["I"], 10] + 1e-3), "copy-through"),
    (lambda xh: xh.__setitem__((demo.LEAD_INDEX["V6"], 5), np.nan), "NaN"),
])
def test_failed_release_check_refuses_output(svc, monkeypatch, corrupt, expect):
    monkeypatch.setattr(demo, "gate_score", fake_gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", fake_reconstruct([], corrupt))
    res = svc.infer(request())
    assert res["outcome"] == "REFUSED_INVALID_RECONSTRUCTION" and res["reconstruction"] is None and not res["released"]
    assert any(expect in p for p in res["release_check"]["problems"])


def test_released_observed_leads_are_copied_exactly(svc, monkeypatch):
    monkeypatch.setattr(demo, "gate_score", fake_gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", fake_reconstruct([]))
    req = request()
    res = svc.infer(req)
    xin = demo.l2g_input(schema.parse_request(req).x)
    idx = [demo.LEAD_INDEX[l] for l in OBS]
    assert np.array_equal(np.array(res["reconstruction"]["values"], dtype=np.float32)[idx], xin[idx])


def test_gate_sees_only_the_observed_leads(svc, monkeypatch):
    seen = []
    monkeypatch.setattr(demo, "gate_score", fake_gate("ABSTAIN", seen))
    svc.infer(request())
    x, observed = seen[0]
    missing = [demo.LEAD_INDEX[l] for l in demo.LEAD_NAMES if l not in OBS]
    assert observed == OBS and x.shape == (12, 5000) and not x[missing].any()


def test_gate_feature_matrix_is_exactly_the_whitelist():
    p2 = io.load_config(REPO / "configs/protocol_v0_2.yaml")
    x = schema.parse_request(request()).x
    feats = {"row_id": "t:0", **gf.row_features(x, OBS)}
    mat = m4b1.method_matrix(pd.DataFrame([feats]), "gate_B", p2).drop(columns="row_id")
    assert list(mat.columns) == list(gf.G1_COLUMNS) + list(gf.G2_COLUMNS) and mat.shape[1] == 30
    assert all(c.startswith(("g1_", "g2_")) for c in mat.columns)
    assert d1mod.check_feature_columns(list(mat.columns), p2) == []
    assert d1mod.check_feature_columns(["D1"], p2)          # the target itself is refused as a feature


def test_invalid_input_stops_before_the_gate(svc, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("gate called on invalid input")
    monkeypatch.setattr(demo, "gate_score", must_not_run)
    for bad in (request(labels=[1]), request(sampling_rate_hz=250), request(observed_leads=["I"])):
        with pytest.raises(schema.InputError):
            svc.infer(bad)


@pytest.fixture
def req():
    return request()                                     # built before the no-writes guard is switched on


def test_service_writes_no_files(svc, monkeypatch, tmp_path, req, no_writes):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(demo, "gate_score", fake_gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", fake_reconstruct([]))
    before = sorted(p.name for p in (REPO / "app").rglob("*") if "__pycache__" not in p.parts)
    assert svc.infer(req)["released"]
    after = sorted(p.name for p in (REPO / "app").rglob("*") if "__pycache__" not in p.parts)
    assert before == after and not list(tmp_path.iterdir())


# ---------- local JSON API ----------
class FakeService:
    integrity = {"ok": True}

    def info(self):
        return {"status": "test"}

    def infer(self, payload):
        if payload.get("mode") == "refuse":
            raise schema.InputError("INVALID_MASK", "bad mask")
        if payload.get("mode") == "boom":
            raise RuntimeError("secret 0.123456")
        return {"outcome": "ABSTAINED", "released": False}


@pytest.fixture
def api():
    logs = []
    srv = server.make_server(FakeService(), port=0, log=logs.append)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1], logs
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, ctype="application/json", length=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    headers = {"Content-Type": ctype}
    if data is not None or length is not None:
        headers["Content-Length"] = str(length if length is not None else len(data))
    c.request(method, path, body=data if length is None else b"", headers=headers)
    r = c.getresponse()
    out = r.status, json.loads(r.read() or b"{}")
    c.close()
    return out


def test_api_startup_health_and_info(api):
    port, _ = api
    assert call(port, "GET", "/api/health") == (200, {"status": "ok", "assets_verified": True})
    assert call(port, "GET", "/api/info")[0] == 200
    assert call(port, "GET", "/nope")[0] == 404


def test_api_infer_and_error_codes(api):
    port, logs = api
    assert call(port, "POST", "/api/infer", {"csv": "I\n0.123456\n"}) == (200, {"outcome": "ABSTAINED", "released": False})
    s, b = call(port, "POST", "/api/infer", {"mode": "refuse"})
    assert s == 422 and b["error"]["code"] == "INVALID_MASK" and b["released"] is False
    s, b = call(port, "POST", "/api/infer", {"mode": "boom"})
    assert s == 500 and "secret" not in json.dumps(b) and b["released"] is False
    assert call(port, "POST", "/api/infer", b"{not json")[0] == 400
    assert call(port, "POST", "/api/infer", b"I\n0.1\n", ctype="text/csv")[0] == 415
    assert call(port, "POST", "/api/infer", None, length=server.MAX_BODY + 1)[0] == 413
    assert logs and all("0.123456" not in line and "secret" not in line for line in logs)   # bodies are never logged


def test_early_error_replies_survive_an_unread_body(api):
    # A 404/415 reply comes before the body is parsed. If the server closed the socket with the body still unread, the OS
    # would send a reset that can destroy the reply (WinError 10053 on Windows), so the reply must arrive every time.
    port, _ = api
    for _ in range(100):
        assert call(port, "POST", "/api/infer", b"I\n0.1\n", ctype="text/csv")[0] == 415
        assert call(port, "POST", "/api/other", {"csv": "I\n0.1\n"})[0] == 404


def test_server_binds_to_the_local_machine_only():
    with pytest.raises(SystemExit):
        server.make_server(FakeService(), host="0.0.0.0", port=0)
