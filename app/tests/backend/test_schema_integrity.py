"""Phase 2 utilities: request-schema validation and the startup asset-integrity check (stage 14).

Synthetic inputs only; no model is run, and no PTB-XL waveform or fold-10 file is read.
    .\\.venv\\Scripts\\python.exe -m pytest app/tests/backend -q
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.backend import integrity, schema
from app.backend.pipeline import REPO, demo

OBS = ["I", "III", "V1"]


def csv_text(leads=OBS, n=5000, scale=1.0, edit=None):
    t = np.arange(n) / 500.0
    cols = {lead: scale * (0.8 * np.sin(2 * np.pi * 1.2 * t + k)) for k, lead in enumerate(leads)}
    df = pd.DataFrame(cols)
    if edit:
        edit(df)
    return df.to_csv(index=False)


def request(**kw):
    r = {"schema_version": schema.SCHEMA_VERSION, "sampling_rate_hz": 500, "units": "mV", "observed_leads": list(OBS),
         "csv": csv_text()}
    r.update(kw)
    return r


def refused(payload, code):
    with pytest.raises(schema.InputError) as e:
        schema.parse_request(payload)
    assert e.value.code == code, (e.value.code, str(e.value))
    return str(e.value)


def test_valid_request_gives_the_demo_input_with_missing_leads_zero():
    p = schema.parse_request(request())
    assert p.x.shape == (12, 5000) and p.observed == OBS and p.samples_in == 5000
    missing = [demo.LEAD_INDEX[l] for l in demo.LEAD_NAMES if l not in OBS]
    assert not p.x[missing].any() and p.x[[demo.LEAD_INDEX[l] for l in OBS]].any()


@pytest.mark.parametrize("field", ["complete_ecg", "labels", "d1", "e1_scores", "reference"])
def test_offline_or_unknown_fields_are_refused(field):
    assert "never be sent to the gate" in refused(request(**{field: [1, 2, 3]}), "FORBIDDEN_OR_UNKNOWN_FIELD")


def test_missing_field_and_wrong_schema_version_are_refused():
    r = request(); del r["units"]
    refused(r, "INVALID_REQUEST")
    refused(request(schema_version="other/1"), "INVALID_REQUEST")
    refused([1, 2], "INVALID_REQUEST")


@pytest.mark.parametrize("fs", [250, 1000, "500", True, None])
def test_only_500_hz_is_accepted(fs):
    refused(request(sampling_rate_hz=fs), "UNSUPPORTED_SAMPLING_RATE")


def test_only_millivolts_are_accepted():
    refused(request(units="uV"), "UNSUPPORTED_UNITS")
    refused(request(csv=csv_text(scale=1000.0)), "INVALID_INPUT")        # microvolt-scale values, caught by the demo check


@pytest.mark.parametrize("obs", [["I", "III"], ["I", "III", "V1", "V2"], ["I", "I", "III"], ["X1"], [], "I,III,V1"])
def test_mask_must_equal_the_csv_header(obs):
    refused(request(observed_leads=obs), "INVALID_MASK")


def test_lead_set_outside_the_258_l2g_sets_is_refused():
    msg = refused(request(observed_leads=["II", "V1"], csv=csv_text(["II", "V1"])), "INVALID_INPUT")
    assert "258" in msg


def test_shapes_nan_inf_and_text_are_refused():
    refused(request(csv=csv_text(n=4000)), "INVALID_INPUT")                                    # too short
    refused(request(csv=csv_text(edit=lambda d: d.__setitem__("I", d["I"].where(d.index != 7)))), "INVALID_INPUT")  # NaN
    refused(request(csv=csv_text(edit=lambda d: d.__setitem__("V1", d["V1"].replace(d["V1"].iloc[3], np.inf)))),
            "INVALID_INPUT")                                                                    # infinite
    refused(request(csv="I,III,V1\n0.1,abc,0.2\n" * 3), "INVALID_CSV")                          # non-numeric
    refused(request(csv="I,III,V1\n0.1,0.2,0.3,0.4\n"), "INVALID_CSV")                          # ragged row
    refused(request(csv=" "), "INVALID_CSV")


def test_extra_column_in_every_row_is_refused_not_shifted():
    """pandas would read a 4th field in every row as an index and shift each lead by one column; the app refuses it."""
    body = "".join(f"{k * 0.001:.3f},{0.5 * np.sin(k / 40):.4f},{0.4 * np.cos(k / 40):.4f},{0.3 * np.sin(k / 30):.4f}\n"
                   for k in range(5000))
    assert "4 fields" in refused(request(csv="I,III,V1\n" + body), "INVALID_CSV")


def test_file_type_and_binary_content_are_refused():
    for name in ("ecg.txt", "ecg.xlsx", "ecg.png", "ecg.dat"):
        refused(request(source_name=name), "UNSUPPORTED_FILE_TYPE")
    refused(request(csv="\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), "UNSUPPORTED_FILE_TYPE")


def test_oversized_input_is_refused():
    refused(request(csv="I\n" + "0.1\n" * (schema.MAX_ROWS + 5)), "INPUT_TOO_LARGE")


def test_fold10_file_name_refusal_is_wired(monkeypatch):
    def fake(path):                          # stand-in: no fold-10 identifier is looked up in this test
        raise SystemExit("this PTB-XL record is in the test fold (fold 10), which is spent; refusing")
    monkeypatch.setattr(demo, "refuse_fold10", fake)
    refused(request(source_name="00001_hr.csv"), "FOLD10_REFUSED")


def test_integrity_passes_on_the_repository_assets():
    rep = integrity.verify()
    assert rep["ok"] and all(f["match"] for f in rep["files"].values()) and len(rep["files"]) == 6


def test_integrity_refuses_a_changed_or_missing_asset(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/protocol_v0_2.yaml").write_text("tampered", encoding="utf-8")
    with pytest.raises(integrity.IntegrityError, match="SHA-256 differs"):
        integrity.verify(tmp_path, {"configs/protocol_v0_2.yaml": integrity.PINNED["configs/protocol_v0_2.yaml"]})
    with pytest.raises(integrity.IntegrityError, match="missing"):
        integrity.verify(tmp_path, {"artifacts/runs/l2g_v0_1_run02/best.pt": integrity.PINNED["artifacts/runs/l2g_v0_1_run02/best.pt"]})


def test_integrity_cross_checks_the_threshold_lock():
    lock = json.loads((REPO / "artifacts/locks/threshold_lock_v0_2.json").read_text(encoding="utf-8"))["methods"]["gate_B"]
    assert lock["t"]["repr"] == integrity.THRESHOLD_REPR
    assert lock["model_sha256"] == integrity.PINNED["artifacts/gate/v0_2/m4b1/01_fit/models/gate_B.joblib"]
