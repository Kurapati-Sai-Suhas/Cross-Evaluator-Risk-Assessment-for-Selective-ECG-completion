"""REAL frozen pipeline (no mocks of the model) on the distributed synthetic examples: G1/G2, Gate B, Platt, the locked
threshold and L2G G v0.1. This replaces test_real_pipeline.py in the public release, whose PTB-XL record-17 fixtures are not
distributed. The expected gate scores are the values recorded when the synthetic inputs were screened on 27 Oct 2026
(examples/synthetic/screening_log.json). Synthetic signals exercise the software only; they are not ECG recordings.
    .\\.venv\\Scripts\\python.exe -m pytest app/tests/backend/test_synthetic_release.py -q
"""
import json

import numpy as np
import pytest

from app.backend import schema
from app.backend.pipeline import REPO, demo
from app.backend.service import InferenceService

EX = REPO / "examples/synthetic"
LOG = {c["id"]: c for c in json.loads((EX / "screening_log.json").read_text(encoding="utf-8"))["tried"]}


def payload(name):
    text = (EX / name).read_text(encoding="utf-8")
    return {"schema_version": schema.SCHEMA_VERSION, "sampling_rate_hz": 500, "units": "mV",
            "observed_leads": text.splitlines()[0].split(","), "csv": text, "source_name": name}


@pytest.fixture(scope="module")
def real():
    return InferenceService(verify_assets=True, preload=False)          # startup integrity check runs for real


def test_cand01_is_accepted_released_and_copied_through(real, no_writes):
    res = real.infer(payload("synthetic_cand01.csv"))
    assert res["outcome"] == "RELEASED" and res["gate"]["decision"] == "ACCEPT" and res["release_check"]["passed"]
    assert res["gate"]["calibrated_score"] == LOG["cand01"]["calibrated_score"]           # bit-identical gate score
    assert res["gate"]["threshold"] == "0.05927309439311807"
    rec = np.array(res["reconstruction"]["values"], dtype=np.float32)
    assert rec.shape == (12, 2048) and np.isfinite(rec).all()
    x = schema.parse_request(payload("synthetic_cand01.csv")).x
    idx = [demo.LEAD_INDEX[l] for l in res["observed_leads"]]
    assert np.array_equal(rec[idx], demo.l2g_input(x)[idx])                                # exact copy-through


def test_cand04_abstains_and_never_reconstructs(real, monkeypatch, no_writes):
    calls, orig = [], demo.reconstruct
    monkeypatch.setattr(demo, "reconstruct", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    res = real.infer(payload("synthetic_cand04.csv"))
    assert res["outcome"] == "ABSTAINED" and res["reconstruction"] is None and not res["reconstruction_computed"]
    assert not calls
    assert res["gate"]["calibrated_score"] == LOG["cand04"]["calibrated_score"]
