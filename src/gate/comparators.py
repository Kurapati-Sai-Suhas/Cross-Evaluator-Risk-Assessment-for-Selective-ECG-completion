"""Comparator scores that are NOT gate features (protocol v0.2): M2 uncertainty-only.

    .\\.venv\\Scripts\\python.exe -m src.gate.comparators --config configs/gate_m4b_v0_2.yaml

M2 score (higher = riskier) = -min_k |p_rec_k - tau_k|, with p_rec = frozen E1 on the reconstruction (already stored by M4a)
and tau from the E1 run summary. This is the only gate-side code allowed to open an offline label file, and it reads
exactly the columns row_id, p_rec_CLBBB, p_rec_CRBBB (never D1, labels or complete-ECG scores). Output column
``e1_recon_margin_score`` is on the protocol's forbidden-input list, so the feature guard rejects it in any gate matrix.
Writes <output_dir>/comparators/<role>_M2_e1_recon_margin.csv for 9a_calibration and 9b_threshold (the roles where M2 is
calibrated and thresholded); refused if that folder exists. E_dir margin scores need E_dir inference and are NOT computed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ptbxl_io as io
from src.eval.d1 import load_thresholds

M2_ROLES = ("9a_calibration", "9b_threshold")
E1_SUMMARY = "artifacts/runs/e1_resnet1d_v0_1_run01/summary.json"


def m2_scores(p_rec: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """-min_k |p_rec_k - tau_k| from float32 probabilities and thresholds (computed in float64)."""
    p = np.asarray(p_rec, dtype=np.float32).astype(np.float64)
    t = np.asarray(tau, dtype=np.float32).astype(np.float64)
    return -np.min(np.abs(p - t[None, :]), axis=1)


def run(repo: Path, cfg_path: Path) -> dict:
    cfg = io.load_config(cfg_path)
    p2 = io.load_config(repo / "configs/protocol_v0_2.yaml")
    outputs = list(p2["targets"]["primary_outputs"])
    tau = load_thresholds(repo / E1_SUMMARY, outputs)
    d1_dir = repo / Path(cfg["d1_keys_dir"]).parent
    manifest = json.loads((d1_dir / "hash_manifest.json").read_text(encoding="utf-8"))
    out = repo / cfg["output_dir"] / "comparators"
    if out.exists():
        raise RuntimeError(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    info = {}
    for role in M2_ROLES:
        src = d1_dir / "offline_labels" / f"{role}_OFFLINE_d1_labels.csv"
        rel = src.relative_to(repo).as_posix()
        if io.sha256_file(src) != manifest["files"][rel]:
            raise RuntimeError(f"{rel} differs from the D1 hash manifest")
        cols = ["row_id"] + [f"p_rec_{y}" for y in outputs]
        df = pd.read_csv(src, usecols=cols)[cols]
        score = m2_scores(df[cols[1:]].to_numpy(), np.array([tau[y] for y in outputs]))
        res = pd.DataFrame({"row_id": df["row_id"], "e1_recon_margin_score": score})
        dest = out / f"{role}_M2_e1_recon_margin.csv"
        res.to_csv(dest, index=False, float_format="%.17g")
        os.chmod(dest, 0o444)
        info[role] = {"rows": len(res), "columns_read": cols, "source_sha256": manifest["files"][rel],
                      "output_sha256": io.sha256_file(dest)}
        print(f"[{role}] M2 scores: {len(res)} rows", flush=True)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "method": "M2_uncertainty_only",
               "formula": "-min_k |p_rec_k - tau_k|", "tau_float32": {y: f"{tau[y]:.9g}" for y in outputs},
               "e1_summary_sha256": io.sha256_file(repo / E1_SUMMARY), "per_role": info,
               "note": "comparator scores; never gate features (forbidden input e1_recon_margin_score)"}
    (out / "comparators_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write M2 comparator scores (no fitting).")
    ap.add_argument("--config", default="configs/gate_m4b_v0_2.yaml")
    args = ap.parse_args(argv)
    repo = Path(".").resolve()
    run(repo, (repo / args.config).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
