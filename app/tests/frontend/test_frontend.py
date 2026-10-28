"""Frontend tests (system Python with Streamlit). The backend is replaced by a stand-in client (MOCKED); no model is run.

    python -m pytest app/tests/frontend -q
The optional end-to-end test at the bottom runs only when ECG_APP_E2E_URL points to a running real backend.
"""
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.frontend import inputs as ui
from app.frontend.client import BackendClient, BackendError

APP = str(Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py")
WARNING = "Research prototype only — not a medical device. Do not use for diagnosis or treatment."
SIX, V6 = list(ui.FIXTURES)   # public release: synthetic cand01 (6 leads) and cand04 (lead V2 only)
L2G8 = ["I", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SETS = [[l] for l in ui.LEAD_NAMES] + [list(c) for k in range(2, 8) for c in itertools.combinations(L2G8, k)]


class FakeClient:
    def __init__(self, outcome="ABSTAINED", healthy=True, refuse=None):
        self.outcome, self.healthy, self.refuse, self.payloads = outcome, healthy, refuse, []

    def health(self):
        if not self.healthy:
            raise BackendError("BACKEND_UNREACHABLE", "cannot reach the backend")
        return {"status": "ok", "assets_verified": True}

    def info(self):
        return {"assets_verified": True, "lead_sets": LEAD_SETS,
                "models": {"reconstructor_sha256": "d515875a" * 8, "gate_sha256": "7b863410" * 8,
                           "calibrator_sha256": "341ba86a" * 8, "threshold": "0.05927309439311807",
                           "not_used": "L2G v0.2-exp is not part of this app"}}

    def infer(self, payload):
        self.payloads.append(payload)
        if self.refuse:
            raise BackendError(*self.refuse)
        released = self.outcome == "RELEASED"
        return {"outcome": self.outcome, "released": released, "explanation": "test", "warnings": [],
                "observed_leads": payload["observed_leads"], "reconstruction_computed": released,
                "gate": {"decision": "ACCEPT" if released else "ABSTAIN", "calibrated_score": 0.03 if released else 0.2,
                         "raw_score": 0.1, "threshold": "0.05927309439311807", "features": {}},
                "reconstruction": ({"lead_names": ui.LEAD_NAMES, "sampling_rate_hz": 250, "n_samples": 2048,
                                    "values": np.full((12, 2048), 0.5).tolist()} if released else None)}


def app(client):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["backend_client"] = client
    at.run()
    assert not at.exception, at.exception
    return at


def texts(elements):
    return " ".join(e.value for e in elements)


def test_startup_shows_title_warning_and_model_versions():
    at = app(FakeClient())
    assert at.title[0].value == "Risk-Controlled Selective ECG Completion"
    assert WARNING in texts(at.error)
    assert "frozen assets verified" in texts(at.success) and "0.05927309439311807" in texts(at.markdown)


def test_backend_down_is_reported_without_crashing():
    at = app(FakeClient(healthy=False))
    assert "Backend not available" in texts(at.error) and "Start the backend first" in texts(at.info)
    assert not at.button(key="reset").disabled


def test_abstain_shows_no_reconstruction():
    c = FakeClient("ABSTAINED")
    at = app(c)
    at.radio(key="source").set_value(V6).run()
    at.button(key="run").click().run()
    assert "ABSTAIN" in texts(at.warning) and "No reconstruction is shown" in texts(at.info)
    assert at.session_state["shown_reconstruction"] is False
    assert "Reconstruction (released)" not in texts(at.subheader)
    assert c.payloads[0]["observed_leads"] == ["V2"]


def test_release_shows_reconstruction_and_sends_only_observed_leads():
    c = FakeClient("RELEASED")
    at = app(c)
    at.button(key="run").click().run()
    assert "ACCEPT" in texts(at.success) and "Reconstruction (released)" in texts(at.subheader)
    assert at.session_state["shown_reconstruction"] is True
    p = c.payloads[0]
    assert set(p) == {"schema_version", "sampling_rate_hz", "units", "observed_leads", "csv", "source_name"}
    assert p["observed_leads"] == ["I", "III", "V1", "V2", "V4", "V6"]
    assert p["csv"].splitlines()[0] == "I,III,V1,V2,V4,V6"


def test_mask_selection_drops_deselected_leads_and_clears_a_stale_result():
    c = FakeClient("RELEASED")
    at = app(c)
    at.button(key="run").click().run()
    assert at.session_state["shown_reconstruction"] is True
    at.multiselect[0].set_value(["I", "III", "V1"]).run()          # changing the input clears the old result
    assert "result" not in at.session_state and at.session_state["shown_reconstruction"] is False
    at.button(key="run").click().run()
    assert c.payloads[-1]["observed_leads"] == ["I", "III", "V1"]
    assert c.payloads[-1]["csv"].splitlines()[0] == "I,III,V1"


def test_backend_refusal_is_shown_and_nothing_is_released():
    at = app(FakeClient(refuse=("INVALID_MASK", "observed_leads do not match")))
    at.button(key="run").click().run()
    assert "Refused (INVALID_MASK)" in texts(at.error) and at.session_state["shown_reconstruction"] is False


def test_reset_clears_the_result_and_the_input_choice():
    at = app(FakeClient("RELEASED"))
    at.radio(key="source").set_value(V6).run()
    at.button(key="run").click().run()
    assert "result" in at.session_state
    at.button(key="reset").click().run()
    assert "result" not in at.session_state and at.session_state["source"] == SIX
    assert at.session_state["shown_reconstruction"] is False


def test_synthetic_input_is_labelled():
    at = app(FakeClient())
    at.radio(key="source").set_value(ui.SYNTHETIC).run()
    assert "not an ECG recording" in texts(at.warning)


# ---------- input helpers ----------
@pytest.mark.parametrize("name", ["ecg.txt", "ecg.xlsx", "ecg.csv.exe", "", "ecg"])
def test_only_csv_file_names_are_accepted(name):
    with pytest.raises(ui.InputProblem):
        ui.check_upload_name(name)


def test_csv_file_names_in_any_case_are_accepted():
    ui.check_upload_name("ECG.CSV")


def test_binary_oversized_or_non_utf8_uploads_are_refused():
    for raw in (b"\x89PNG\r\n\x1a\n\x00\x00", b"I\n" + b"0.1\n" * 800_000, "I\n0.1\n".encode("utf-16")):
        with pytest.raises(ui.InputProblem):
            ui.read_csv_text(raw)


def test_ragged_unknown_or_duplicate_headers_are_refused():
    for text in ("I,III,V1\n0,0.1,0.2,0.3\n", "I,XYZ\n0.1,0.2\n", "I,I\n0.1,0.2\n", ""):
        with pytest.raises(ui.InputProblem):
            ui.frame_from_text(text)


def test_payload_keeps_full_precision_and_only_selected_leads():
    df = ui.frame_from_text("I,III,V1\n0.123456789012,1.5,2.5\n0.2,1.6,2.6\n")
    p = ui.build_payload(df, ["V1", "I"])
    assert p["observed_leads"] == ["I", "V1"] and "III" not in p["csv"]
    assert ui.frame_from_text(p["csv"])["I"].iloc[0] == 0.123456789012


def test_fixtures_hold_only_their_observed_leads():
    six, _ = ui.load_fixture(SIX)
    v6, _ = ui.load_fixture(V6)
    assert list(six.columns) == ["I", "III", "V1", "V2", "V4", "V6"] and list(v6.columns) == ["V2"]
    assert len(six) == len(v6) == 5000


def test_supported_lead_sets():
    assert ui.is_supported_set(["I", "V2"], LEAD_SETS) and not ui.is_supported_set(["II", "V1"], LEAD_SETS)
    assert len(LEAD_SETS) == 258


# ---------- optional end-to-end test against a running REAL backend ----------
E2E = os.environ.get("ECG_APP_E2E_URL")


@pytest.mark.skipif(not E2E, reason="set ECG_APP_E2E_URL to a running backend to run the real end-to-end test")
def test_e2e_real_backend_release_and_abstain():
    at = app(BackendClient(E2E))
    at.button(key="run").click().run()
    assert "ACCEPT" in texts(at.success) and at.session_state["shown_reconstruction"] is True
    at.radio(key="source").set_value(V6).run()
    at.button(key="run").click().run()
    assert "ABSTAIN" in texts(at.warning) and at.session_state["shown_reconstruction"] is False
