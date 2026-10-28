"""Deterministic SYNTHETIC partial-ECG CSVs for the stage 18 software smoke test (28 Oct 2026 deliverable).

Every value is computed here from mathematical waveforms: a sum of Gaussian P, Q, R, S and T bumps per beat with
hand-set, lead-specific gains, a slow sinusoidal baseline and seeded Gaussian noise. Nothing is copied, transformed,
downloaded or extracted from PTB-XL, CPSC 2018, LUDB or any other dataset, and no project data file is read.
The output is NOT a patient ECG and NOT evidence of clinical validity or model accuracy.

The candidate list and its order are fixed here, before any candidate is run (see PROVENANCE.md for the selection rule).
    python make_synthetic_ecg.py OUT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FS, SECONDS = 500, 10.0
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# hand-set gains (mV) for the P, Q, R, S and T bumps of each lead; illustrative values, not fitted to any data
GAINS = {"I": (0.08, -0.05, 0.70, -0.10, 0.20), "II": (0.12, -0.06, 1.00, -0.15, 0.30),
         "III": (0.05, -0.04, 0.40, -0.10, 0.12), "AVR": (-0.10, 0.05, -0.80, 0.10, -0.25),
         "AVL": (0.03, -0.03, 0.30, -0.05, 0.08), "AVF": (0.08, -0.05, 0.70, -0.12, 0.20),
         "V1": (0.05, 0.00, 0.30, -0.90, -0.10), "V2": (0.07, 0.00, 0.60, -1.10, 0.25),
         "V3": (0.08, -0.05, 0.90, -0.70, 0.35), "V4": (0.08, -0.08, 1.30, -0.40, 0.40),
         "V5": (0.08, -0.08, 1.20, -0.20, 0.35), "V6": (0.08, -0.06, 0.90, -0.10, 0.28)}
# bump centre (s, relative to the R peak) and width (s, Gaussian sigma)
SHAPE = [(-0.20, 0.025), (-0.035, 0.010), (0.0, 0.012), (0.035, 0.012), (0.30, 0.060)]
# pre-declared candidates, tried in this order; cand01 is the primary demonstration input
CANDIDATES = [
    {"id": "cand01", "seed": 1, "heart_rate_bpm": 72, "observed": ["I", "III", "V1", "V2", "V4", "V6"]},
    {"id": "cand02", "seed": 2, "heart_rate_bpm": 64, "observed": ["I", "III", "V1", "V3", "V5"]},
    {"id": "cand03", "seed": 3, "heart_rate_bpm": 80, "observed": ["I", "V2", "V5"]},
    {"id": "cand04", "seed": 4, "heart_rate_bpm": 70, "observed": ["V2"]},
    {"id": "cand05", "seed": 5, "heart_rate_bpm": 76, "observed": ["I", "III", "V1", "V2", "V3", "V4", "V6"]},
    {"id": "cand06", "seed": 6, "heart_rate_bpm": 60, "observed": ["III", "V4"]},
]


def synth_12lead(seed: int, heart_rate_bpm: float) -> np.ndarray:
    """(12, 5000) millivolts at 500 Hz."""
    rng = np.random.default_rng(seed)
    n = int(FS * SECONDS)
    t = np.arange(n) / FS
    rr = 60.0 / heart_rate_bpm
    peaks, p = [], 0.4
    while p < SECONDS + 0.5:                      # mild, deterministic respiratory-like RR modulation
        peaks.append(p)
        p += rr * (1.0 + 0.03 * np.sin(2 * np.pi * 0.25 * p))
    x = np.zeros((len(LEADS), n))
    for k, lead in enumerate(LEADS):
        for r in peaks:
            for (c, w), g in zip(SHAPE, GAINS[lead]):
                x[k] += g * np.exp(-0.5 * ((t - r - c) / w) ** 2)
        x[k] += 0.05 * np.sin(2 * np.pi * 0.3 * t + 0.4 * k)        # slow baseline wander
        x[k] += rng.normal(0.0, 0.01, n)                              # small white noise
    return x


def main(out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    index = []
    for c in CANDIDATES:
        x = synth_12lead(c["seed"], c["heart_rate_bpm"])
        cols = [LEADS.index(l) for l in c["observed"]]
        lines = [",".join(c["observed"])] + [",".join(f"{v:.6f}" for v in x[cols, i]) for i in range(x.shape[1])]
        p = out / f"synthetic_{c['id']}.csv"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        index.append({**c, "file": p.name, "rows": x.shape[1], "fs_hz": FS, "units": "mV",
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (out / "candidates.json").write_text(json.dumps({"generator": "make_synthetic_ecg.py", "synthetic": True,
                                                     "candidates": index}, indent=1), encoding="utf-8")
    print(f"wrote {len(index)} synthetic CSVs to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
