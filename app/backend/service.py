"""In-memory inference service: validate -> gate (G1/G2 -> Gate B -> Platt -> locked t) -> on ACCEPT only, L2G G v0.1 with
copy-through -> release check. Every step is the frozen stage 10 demo function, called unchanged; nothing is written to
disk and no waveform is logged.
"""
import threading
from datetime import datetime, timezone

import numpy as np

from . import integrity, schema
from .pipeline import LEAD_NAMES, LEAD_SETS, demo

WARNING = "Research prototype only — not a medical device. Do not use for diagnosis or treatment."
LIMITATIONS = [
    "Evaluated once on PTB-XL fold 10 with synthetic whole-lead missingness; no external or clinical validation.",
    "Gate B withholds most BBB (especially CLBBB) ECGs; ACCEPT is not evidence of diagnostic fidelity.",
    "D1, the gate's target, is disagreement with one frozen classifier, not clinical harm or truth."]
EXPLAIN = {
    "RELEASED": "The gate's calibrated risk score is at or below the locked threshold, so the L2G reconstruction was computed "
                "and released after the release check passed (observed samples reproduced exactly, all values finite).",
    "ABSTAINED": "The gate's calibrated risk score is above the locked threshold, so the system abstains: no reconstruction "
                 "was computed, and none should be relied on for this input.",
    "REFUSED_INVALID_RECONSTRUCTION": "The gate accepted, but the reconstruction failed the release check, so nothing is "
                                      "released (fail closed)."}


class InferenceService:
    def __init__(self, verify_assets: bool = True, preload: bool = True):
        self.integrity = integrity.verify() if verify_assets else {"ok": None, "skipped": True}
        self._lock = threading.Lock()                       # the cached models are shared; one request at a time
        if preload:
            demo.gate_assets()
            demo.generator()

    def info(self) -> dict:
        return {"status": demo.STATUS, "warning": WARNING, "schema_version": schema.SCHEMA_VERSION,
                "sampling_rate_hz": 500, "units": "mV", "min_samples": int(demo.gf.WINDOW[1]),
                "validated_samples": demo.N_SAMPLES, "lead_names": LEAD_NAMES, "lead_sets": LEAD_SETS,
                "models": {"reconstructor": "L2G-ECG based ECGen, G v0.1 (reconstructor of record)",
                           "reconstructor_sha256": integrity.PINNED["artifacts/runs/l2g_v0_1_run02/best.pt"],
                           "gate": "Gate B: Random Forest on G1 (15) + G2 (15), fitted on B_gate",
                           "gate_sha256": integrity.PINNED["artifacts/gate/v0_2/m4b1/01_fit/models/gate_B.joblib"],
                           "calibrator": "Platt scaling fitted on 9a",
                           "calibrator_sha256": integrity.PINNED["artifacts/gate/v0_2/m4b1/02_calibration/calibrator_gate_B.joblib"],
                           "threshold": integrity.THRESHOLD_REPR, "threshold_locked_on": "9b",
                           "threshold_lock_sha256": integrity.PINNED["artifacts/locks/threshold_lock_v0_2.json"],
                           "not_used": "L2G v0.2-exp (exploratory, development data only) is not part of this app"},
                "assets_verified": self.integrity.get("ok"), "limitations": LIMITATIONS}

    def infer(self, payload) -> dict:
        p = schema.parse_request(payload)                   # raises schema.InputError; nothing computed before this
        with self._lock:
            gate = demo.gate_score(p.x, p.observed)         # the decision never depends on a reconstruction
            res = {"status": demo.STATUS, "warning": WARNING, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "observed_leads": p.observed, "missing_leads": [l for l in LEAD_NAMES if l not in p.observed],
                   "samples_in": p.samples_in, "samples_used": int(p.x.shape[1]), "warnings": p.warnings,
                   "gate": {"decision": gate["decision"], "calibrated_score": gate["calibrated_score"],
                            "raw_score": gate["raw_score"], "threshold": gate["threshold_repr"],
                            "features": gate["features"], "model_sha256": gate["model_sha256"],
                            "calibrator_sha256": gate["calibrator_sha256"]},
                   "reconstruction_computed": False, "reconstruction": None, "release_check": None,
                   "limitations": LIMITATIONS}
            if gate["decision"] == "ACCEPT":
                x_hat, info = demo.reconstruct(p.x, p.observed)
                res["reconstruction_computed"] = True
                problems = demo.release_check(demo.l2g_input(p.x), x_hat, p.observed, info["copy_through_violations"])
                res["release_check"] = {"passed": not problems, "problems": problems}
                if problems:
                    res["outcome"] = "REFUSED_INVALID_RECONSTRUCTION"
                else:
                    res["outcome"] = "RELEASED"
                    res["reconstruction"] = {"lead_names": LEAD_NAMES, "sampling_rate_hz": 250, "n_samples": int(x_hat.shape[1]),
                                             "units": "per-lead min-max scaled to [0, 1] (the model's scale); not mV",
                                             "values": np.asarray(x_hat, dtype=np.float64).tolist(),
                                             "observed_leads_copied_exactly": True,
                                             "g_sha256": info["g_sha256"], "device": info["device"]}
            else:
                res["outcome"] = "ABSTAINED"
        res["released"] = res["outcome"] == "RELEASED"
        res["explanation"] = EXPLAIN[res["outcome"]]
        return res
