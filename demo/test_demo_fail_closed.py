"""Synthetic regression tests for the demo's fail-closed release and input validation (stage 10, 2026-10-26).

No model, checkpoint, PTB-XL record or fold-10 file is read: the gate and the reconstruction are replaced by monkeypatched
stand-ins, inputs are synthetic CSV files in pytest's temporary folder, and only the pinned config/lock hashes are read.
    .venv/Scripts/python.exe -m pytest demo/test_demo_fail_closed.py -p no:cacheprovider -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_partial_ecg as demo  # noqa: E402

OBS = ["I", "III", "V1"]                                   # one of the 258 L2G lead sets


def _signal(n=5000, scale=1.0, seed=0):
    t = np.arange(n) / 500.0
    rng = np.random.default_rng(seed)
    return {lead: scale * (0.8 * np.sin(2 * np.pi * 1.2 * t + k) + 0.05 * rng.standard_normal(n)) for k, lead in enumerate(OBS)}


def _csv(tmp_path, cols, name="synthetic.csv"):
    p = tmp_path / name
    pd.DataFrame(cols).to_csv(p, index=False)
    return p


def _x(cols):
    x = np.zeros((12, len(next(iter(cols.values())))))
    for lead, v in cols.items():
        x[demo.LEAD_INDEX[lead]] = v
    return x


def _gate(decision):
    return lambda x, observed: {"decision": decision, "calibrated_score": 0.01 if decision == "ACCEPT" else 0.9,
                                "threshold_repr": "0.05927309439311807", "features": {}, "raw_score": 0.0}


def _valid_recon(x, observed):
    xin = demo.l2g_input(x)
    x_hat = xin.copy()
    missing = [i for i in range(12) if demo.LEAD_NAMES[i] not in observed]
    x_hat[missing] = 0.5
    return x_hat, {"device": "stub", "copy_through_violations": 0, "g_sha256": "stub", "batch_rows": 64}


# --- release_check on synthetic arrays ------------------------------------------------------------------------------
def test_release_check_passes_exact_copy_through():
    x = _x(_signal())
    x_hat, _ = _valid_recon(x, OBS)
    assert demo.release_check(demo.l2g_input(x), x_hat, OBS, 0) == []


def test_release_check_detects_changed_observed_sample():
    x = _x(_signal())
    x_hat, _ = _valid_recon(x, OBS)
    x_hat[demo.LEAD_INDEX["III"], 100] += 1e-3
    probs = demo.release_check(demo.l2g_input(x), x_hat, OBS, 0)
    assert any("copy-through" in p for p in probs)


def test_release_check_detects_reported_violations_and_non_finite():
    x = _x(_signal())
    x_hat, _ = _valid_recon(x, OBS)
    x_hat[demo.LEAD_INDEX["V6"], 7] = np.nan
    probs = demo.release_check(demo.l2g_input(x), x_hat, OBS, 2)
    assert any("2 copy-through violations" in p for p in probs) and any("NaN or infinite" in p for p in probs)


# --- main(): accepted, abstained, refused and failure paths ---------------------------------------------------------
def test_accept_with_valid_reconstruction_is_released(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "gate_score", _gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", _valid_recon)
    out = tmp_path / "out"
    assert demo.main(["--csv", str(_csv(tmp_path, _signal())), str(out)]) == 0
    d = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert d["outcome"] == "RELEASED" and d["released"] is True and d["release_check"]["passed"] is True
    assert (out / "reconstruction_250Hz_minmax.csv").exists() and not (out / "reconstruction_250Hz_minmax.csv.partial").exists()


def test_abstain_never_reconstructs_or_writes_a_reconstruction(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("reconstruct must not be called on ABSTAIN")
    monkeypatch.setattr(demo, "gate_score", _gate("ABSTAIN"))
    monkeypatch.setattr(demo, "reconstruct", boom)
    out = tmp_path / "out"
    assert demo.main(["--csv", str(_csv(tmp_path, _signal())), str(out)]) == 0
    d = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert d["outcome"] == "ABSTAINED" and d["released"] is False
    assert not (out / "reconstruction_250Hz_minmax.csv").exists()


@pytest.mark.parametrize("corrupt", ["observed_changed", "non_finite", "reported_violation"])
def test_invalid_reconstruction_is_refused_fail_closed(tmp_path, monkeypatch, corrupt):
    def bad(x, observed):
        x_hat, info = _valid_recon(x, observed)
        if corrupt == "observed_changed":
            x_hat[demo.LEAD_INDEX["I"], 0] += 0.25
        elif corrupt == "non_finite":
            x_hat[demo.LEAD_INDEX["V6"], 10] = np.inf
        else:
            info["copy_through_violations"] = 1
        return x_hat, info
    monkeypatch.setattr(demo, "gate_score", _gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", bad)
    out = tmp_path / "out"
    assert demo.main(["--csv", str(_csv(tmp_path, _signal())), str(out)]) == 3
    d = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert d["outcome"] == "REFUSED_INVALID_RECONSTRUCTION" and d["released"] is False and d["release_check"]["passed"] is False
    assert d["reconstruction"]["file"] is None
    assert sorted(p.name for p in out.iterdir()) == ["decision.json"]


def test_error_during_reconstruction_leaves_no_output(tmp_path, monkeypatch):
    def crash(x, observed):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(demo, "gate_score", _gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", crash)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        demo.main(["--csv", str(_csv(tmp_path, _signal())), str(out)])
    assert not out.exists()


# --- input validation: every refusal happens before any output --------------------------------------------------------
@pytest.mark.parametrize("case", ["duplicate_header", "non_numeric", "unsupported_lead_set", "too_short", "nan", "microvolts",
                                  "unknown_lead", "empty_file"])
def test_input_refusals_write_nothing(tmp_path, monkeypatch, case):
    monkeypatch.setattr(demo, "gate_score", _gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", _valid_recon)
    sig = _signal()
    if case == "duplicate_header":
        p = tmp_path / "dup.csv"
        p.write_text("I,I,V1\n" + "\n".join("0.1,0.2,0.3" for _ in range(5000)) + "\n", encoding="utf-8")
    elif case == "non_numeric":
        sig["V1"] = sig["V1"].astype(object)
        sig["V1"][5] = "abc"
        p = _csv(tmp_path, sig)
    elif case == "unsupported_lead_set":
        p = _csv(tmp_path, {"I": sig["I"], "II": sig["III"]})          # II is a lead set only on its own
    elif case == "too_short":
        p = _csv(tmp_path, {k: v[:1000] for k, v in sig.items()})
    elif case == "nan":
        sig["I"][3] = np.nan
        p = _csv(tmp_path, sig)
    elif case == "microvolts":
        p = _csv(tmp_path, _signal(scale=1000.0))
    elif case == "unknown_lead":
        p = _csv(tmp_path, {"I": sig["I"], "V7": sig["V1"]})
    else:
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        demo.main(["--csv", str(p), str(out)])
    assert e.value.code not in (0, None)
    assert not out.exists()


def test_existing_output_folder_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "gate_score", _gate("ACCEPT"))
    monkeypatch.setattr(demo, "reconstruct", _valid_recon)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        demo.main(["--csv", str(_csv(tmp_path, _signal())), str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
