"""Milestone 3a development comparison on C_dev (protocol v0.1 Section 9, "reconstruction usefulness"; DEVELOPMENT ONLY).

Run from the repository root (after E1 and E_dir have finished):
    .\\.venv\\Scripts\\python.exe -m src.eval.compare_cdev_evaluators

For each C_dev row (2,620 records x the 8 protocol evaluation masks) it scores, against the PTB-XL labels:
  E1(x)               E1 on the complete ECG (upper reference; identical for the 8 masks of a record)
  E1(x_hat)           E1 on the G v0.1 reconstruction with exact copy-through
  E_dir(x_obs, m)     the direct reduced-lead classifier
and reports, per output, AUROC and AUPRC with patient-cluster bootstrap CIs, three paired AUROC differences (same
replicate weights): E1(x_hat) - E_dir, E1(x) - E1(x_hat), E1(x) - E_dir, and label-based sensitivity / specificity / PPV
at each method's documented threshold: E1's Youden tau (from the E1 run summary) for E1(x) and E1(x_hat), E_dir's own
Youden tau (from the E_dir run summary) for E_dir. All three methods are scored on the SAME rows (same records, same
masks); E1(x) does not use the mask (upper reference with full information).

Limits: C_dev selected both E1 and E_dir (early stopping, attempt choice), so these numbers are optimistic and are
NOT the protocol comparison, which happens once, on fold 10, after the threshold lock. No D1 labels are computed here
(E1 decisions on x_hat are compared with the PTB-XL labels only, never with E1 decisions on x). Nothing is trained.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data import missingness as mz
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.data.ptbxl_waveforms import resolve_waveform_root
from src.data.targets import target_matrix
from src.eval import classification as cl
from src.models import l2g
from src.models.evaluators import ResNet1dWang
from src.train import cache as cachemod
from src.train.preflight import G_V0_1_CHECKPOINT, G_V0_1_SHA256
from src.train.train_l2g import select_role, validation_lead_masks, windows_at
from src.utils.provenance import code_tree_sha256, environment

METHODS = ("e1_complete", "e1_recon", "edir")


def _load_evaluator(repo: Path, run_name: str, device) -> tuple[torch.nn.Module, dict[str, Any]]:
    run = repo / "artifacts" / "runs" / run_name
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    ckpt = repo / summary["selected_checkpoint"]
    if io.sha256_file(ckpt) != summary["selected_checkpoint_sha256"]:
        raise RuntimeError(f"{ckpt} does not match its logged SHA-256")
    in_ch = 24 if summary["input"] == "masked" else 12
    model = ResNet1dWang(in_ch, len(summary["outputs"])).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    return model.eval(), {"run": run_name, "checkpoint": summary["selected_checkpoint"],
                          "sha256": summary["selected_checkpoint_sha256"], "eligible": summary.get("eligible"),
                          "eligibility_applies": summary.get("eligibility_applies", True),
                          "tau": {k: v["youden_tau_probability"] for k, v in summary["per_output"].items()}}


def run(repo: Path, e1_run: str, edir_run: str) -> dict[str, Any]:
    tree = code_tree_sha256(repo)["tree_sha256"]
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    prep_path = repo / "configs" / "preprocessing.yaml"
    prep_cfg = io.load_config(prep_path)
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    seed = int(protocol["seed"])
    k = int(protocol["d1"]["masks_per_record"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g_path = repo / G_V0_1_CHECKPOINT
    if io.sha256_file(g_path) != G_V0_1_SHA256:
        raise RuntimeError("G v0.1 checkpoint changed")
    gen = l2g.ECGen(24).to(device)
    gen.load_state_dict(torch.load(g_path, map_location=device, weights_only=False)["generator"])
    gen.eval()
    e1, e1_info = _load_evaluator(repo, e1_run, device)
    edir, edir_info = _load_evaluator(repo, edir_run, device)

    manifest = load_frozen_manifest(audit_cfg, repo)
    rows = select_role(manifest, "C_dev", set(protocol["targets"]["excluded_ecg_ids"]))
    db = io.load_database(repo / audit_cfg["dataset"]["database_csv"])
    key = cachemod.cache_key(prep_path, repo / audit_cfg["dataset"]["database_csv"],
                             audit_cfg["split"]["frozen_manifest_sha256"], "C_dev", rows["ecg_id"].tolist())
    signals, _, _ = cachemod.ensure_cache("C_dev", rows, resolve_waveform_root(audit_cfg, repo), db, prep_cfg, key,
                                          Path("C:/ecg_cache/v0_1"))
    val_x = windows_at(signals, prep_cfg, 0)
    masks = validation_lead_masks(rows["ecg_id"].tolist(), seed, k)  # (N, k, 12)
    outputs = list(protocol["targets"]["primary_outputs"])
    y_rec = target_matrix(db, rows["ecg_id"].tolist(), protocol)
    y = np.repeat(y_rec, k, axis=0)
    codes = [io.parse_scp_codes(t) for t in db.set_index("ecg_id").loc[rows["ecg_id"], "scp_codes"]]
    label_notes = {
        "rule": "y=1 if the code is listed in scp_codes at any likelihood (protocol targets.positive_min_likelihood 0); "
                "otherwise 0, including ILBBB/IRBBB-only records; there is no missing-label category in PTB-XL",
        "records_with_empty_scp_codes": int(sum(len(c) == 0 for c in codes)),
        **{f"{o}_positive_records_with_likelihood_0": int(sum(c.get(o, -1) == 0 for c in codes)) for o in
           protocol["targets"]["primary_outputs"]},
        **{f"{o}_positive_records": int(y_rec[:, j].sum()) for j, o in enumerate(protocol["targets"]["primary_outputs"])},
        **{f"{o}_positive_patients": int(rows.loc[y_rec[:, j] == 1, "patient_id"].nunique())
           for j, o in enumerate(protocol["targets"]["primary_outputs"])},
        "both_positive_records": int((y_rec.sum(axis=1) == 2).sum()),
    }

    probs = {m: [] for m in METHODS}
    per = 32
    with torch.no_grad():
        for s in range(0, len(val_x), per):
            x = torch.from_numpy(val_x[s:s + per]).to(device)
            probs["e1_complete"].append(torch.sigmoid(e1(x)).repeat_interleave(k, dim=0).cpu().numpy())
            x = x.repeat_interleave(k, dim=0)
            m = torch.from_numpy(masks[s:s + per]).to(device).float().reshape(-1, 12).unsqueeze(-1).expand_as(x)
            x_obs = x * m
            inp = torch.cat([x_obs, m], dim=1)
            x_hat = torch.where(m.bool(), x_obs, l2g.minmax_per_lead(gen(inp)))  # exact copy-through
            probs["e1_recon"].append(torch.sigmoid(e1(x_hat)).cpu().numpy())
            probs["edir"].append(torch.sigmoid(edir(inp)).cpu().numpy())
    probs = {m: np.concatenate(v) for m, v in probs.items()}

    pid = np.repeat(rows["patient_id"].to_numpy(), k)
    lm = masks.reshape(-1, 12)
    table = pd.DataFrame({"ecg_id": np.repeat(rows["ecg_id"].to_numpy(), k), "patient_id": pid,
                          "mask_index": np.tile(np.arange(k), len(rows)), "n_observed": lm.sum(axis=1),
                          "observed_leads": [";".join(np.asarray(mz.LEAD_NAMES)[r.astype(bool)]) for r in lm]})
    reps, level = int(protocol["uncertainty"]["replicates"]), float(protocol["uncertainty"]["two_sided_level"])
    results: dict[str, Any] = {}
    for j, name in enumerate(outputs):
        table[f"y_{name}"] = y[:, j]
        fns = {}
        for m in METHODS:
            table[f"prob_{m}_{name}"] = probs[m][:, j]
            fns[m] = cl.auroc_fn(probs[m][:, j], y[:, j])
        res = {f"auroc_{m}": cl.cluster_bootstrap_metric(fns[m], y[:, j], pid, reps, seed, level) for m in METHODS}
        for m in METHODS:
            res[f"auprc_{m}"] = cl.cluster_bootstrap_metric(cl.average_precision_fn(probs[m][:, j], y[:, j]), y[:, j],
                                                            pid, reps, seed, level)
            tau = (edir_info if m == "edir" else e1_info)["tau"][name]
            pr = probs[m][:, j]
            res[f"operating_point_{m}"] = {"tau": tau, "tau_source": "E_dir summary" if m == "edir" else "E1 summary",
                                           **{q: cl.cluster_bootstrap_metric(
                                               lambda w, q=q, pr=pr, tau=tau: cl.operating_point(pr, y[:, j], tau, w)[q],
                                               y[:, j], pid, reps, seed, level) for q in ("sensitivity", "specificity", "ppv")}}
        for a, b in (("e1_recon", "edir"), ("e1_complete", "e1_recon"), ("e1_complete", "edir")):
            res[f"diff_auroc_{a}_minus_{b}"] = cl.cluster_bootstrap_metric(
                lambda w, a=a, b=b: fns[a](w) - fns[b](w), y[:, j], pid, reps, seed, level)
        by_n = {}
        for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 11)):
            sel = (table.n_observed >= lo) & (table.n_observed <= hi)
            by_n[f"{lo}-{hi}"] = {"rows": int(sel.sum()), "positive_rows": int(y[sel, j].sum()),
                                  **{m: cl.weighted_auroc(probs[m][sel, j], y[sel, j]) for m in METHODS}}
        res["auroc_point_by_n_observed_descriptive"] = by_n
        results[name] = res
    out_rows = repo / "artifacts" / "reports" / "m3a_cdev_evaluator_comparison_rows.csv"
    table.to_csv(out_rows, index=False)
    summary = {
        "milestone": "M3a", "kind": "development comparison on C_dev (selection-biased; not the protocol test)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "d1_computed": False, "fold10_loaded": False, "roles_loaded": ["C_dev"],
        "rows": len(table), "records": len(rows), "patients": int(rows["patient_id"].nunique()),
        "masks_per_record": k, "mask_rule": "evaluation_combinations(ecg_id, seed=20261030, k=8), identical for all methods",
        "labels": label_notes, "G": {"checkpoint": G_V0_1_CHECKPOINT, "sha256": G_V0_1_SHA256},
        "E1": e1_info, "E_dir": edir_info, "results": results,
        "rows_file": str(out_rows.relative_to(repo)), "rows_file_sha256": io.sha256_file(out_rows),
        "uncertainty": f"patient-cluster percentile bootstrap, {reps} replicates, paired (same weights) for differences",
        "code_tree_sha256": tree, "environment": environment(),
    }
    (repo / "artifacts" / "reports" / "m3a_cdev_evaluator_comparison.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="C_dev development comparison: E1(x), E1(x_hat), E_dir.")
    parser.add_argument("--e1-run", default="e1_resnet1d_v0_1_run01")
    parser.add_argument("--edir-run", default="edir_resnet1d_v0_1_run02")  # run01 was interrupted (D-027)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    s = run(Path(args.root).resolve(), args.e1_run, args.edir_run)
    for name, res in s["results"].items():
        print(name, {k_: (round(v["point"], 4), round(v["ci_low"], 4), round(v["ci_high"], 4))
                     for k_, v in res.items() if isinstance(v, dict) and "point" in v})
    return 0


if __name__ == "__main__":
    sys.exit(main())
