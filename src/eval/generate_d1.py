"""Milestone M4a: offline D1 generation for C_dev (sanity only), B_gate, 9a_calibration and 9b_threshold.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.eval.generate_d1 --config configs/d1_generation_v0_1.yaml --preflight-only
    .\\.venv\\Scripts\\python.exe -m src.eval.generate_d1 --config configs/d1_generation_v0_1.yaml

Frozen inputs (hash-checked, never modified): G v0.1 (L2G), E1 v0.1 and its stored thresholds tau, protocol v0.1, the frozen
split manifest. Nothing is trained. test_fold10 and A_fit are refused.

Fixed scoring path (M3b finding: batch configuration moved probabilities by ~1e-7 and flipped a decision at tau):
* one function scores both complete and reconstructed ECGs; every forward pass of G and E1 has the same batch shape
  (``scoring.batch_rows``; the last batch is zero-padded), float32, eval mode, inference mode, cuDNN deterministic;
* E1 on the complete ECG is computed ONCE per record and reused for the record's 8 mask rows;
* x_hat = where(m, x_obs, G(x_obs, m)) (exact copy-through, verified for every batch);
* every probability within ``boundary_eps`` of tau is reported (never relabelled, tau never changed).

Outputs (``output_dir``; refused if it exists):
  keys/<role>_rows.csv                         row_id + identifiers + mask metadata (no reference-derived values)
  offline_labels/<role>_OFFLINE_d1_labels.csv  row_id + p_full, p_rec, tau, decisions, D1, directional D1, PTB-XL labels
  offline_labels/<role>_OFFLINE_e1_complete_by_record.csv   p_full once per record
  generation_summary.json, hash_manifest.json, README.md
and artifacts/reports/m4a_d1_summary.json, m4a_d1_counts_by_role.csv, m4a_threshold_boundary_cases.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from src.data import missingness as mz
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.data.ptbxl_waveforms import resolve_waveform_root
from src.data.targets import target_matrix
from src.eval import d1 as d1mod
from src.eval.stats import patient_cluster_bootstrap
from src.models import l2g
from src.models.evaluators import ResNet1dWang
from src.train import cache as cachemod
from src.train.train_l2g import validation_lead_masks, windows_at
from src.utils.provenance import code_tree_sha256, environment

FOLD10_ROLE = "test_fold10"


# --------------------------------------------------------------------------- #
# Role selection
# --------------------------------------------------------------------------- #
def select_d1_role(manifest: pd.DataFrame, role: str, protocol: dict[str, Any]) -> pd.DataFrame:
    """Rows of one D1 role (sorted by ecg_id, protocol exclusions removed). Refuses fold 10 and non-D1 roles."""
    if role == FOLD10_ROLE:
        raise RuntimeError("test_fold10 is refused: protocol Section 7 allows fold-10 D1 only after the threshold lock")
    if role not in protocol["d1"]["roles"]:
        raise RuntimeError(f"{role} is not a D1 role of protocol v0.1 (d1.roles = {protocol['d1']['roles']})")
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    rows = manifest[(manifest["role"] == role) & ~manifest["ecg_id"].isin(excluded)].sort_values("ecg_id")
    if rows.empty or (rows["strat_fold"] == 10).any():
        raise RuntimeError(f"role {role}: empty or contains fold-10 records")
    return rows.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fixed scoring path
# --------------------------------------------------------------------------- #
def set_deterministic(cfg: dict[str, Any]) -> None:
    torch.backends.cudnn.deterministic = bool(cfg["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(cfg["cudnn_benchmark"])


def _fixed_batches(n: int, b: int) -> Iterable[tuple[int, int]]:
    for s in range(0, n, b):
        yield s, min(s + b, n)


def _pad(t: torch.Tensor, b: int) -> torch.Tensor:
    if t.shape[0] == b:
        return t
    return torch.cat([t, t.new_zeros((b - t.shape[0],) + tuple(t.shape[1:]))], dim=0)


@torch.inference_mode()
def score_e1(e1: torch.nn.Module, x: torch.Tensor, batch_rows: int) -> torch.Tensor:
    """sigmoid(E1(x)) in float32 for x (N, 12, T) already on the device; every forward pass has batch shape batch_rows."""
    if e1.training:
        raise RuntimeError("E1 must be in eval mode")
    out = []
    for s, e in _fixed_batches(x.shape[0], batch_rows):
        out.append(torch.sigmoid(e1(_pad(x[s:e], batch_rows)).float())[: e - s])
    return torch.cat(out)


@torch.inference_mode()
def reconstruct(gen: torch.nn.Module, x: torch.Tensor, lead_mask: torch.Tensor, batch_rows: int) -> tuple[torch.Tensor, int]:
    """x_hat with exact copy-through for x (N, 12, T) and lead_mask (N, 12); also returns the copy-through violations."""
    if gen.training:
        raise RuntimeError("G must be in eval mode")
    out, violations = [], 0
    for s, e in _fixed_batches(x.shape[0], batch_rows):
        xb = x[s:e]
        m = lead_mask[s:e].to(xb.dtype).unsqueeze(-1).expand_as(xb)
        x_obs = xb * m
        x_gen = l2g.minmax_per_lead(gen(_pad(torch.cat([x_obs, m], dim=1), batch_rows)))[: e - s]
        x_hat = torch.where(m.bool(), x_obs, x_gen)
        obs = m.bool()
        violations += int((x_hat[obs] != xb[obs]).sum())
        out.append(x_hat)
    return torch.cat(out), violations


def state_hash(model: torch.nn.Module) -> str:
    """SHA-256 of every parameter and buffer (BatchNorm running statistics included)."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def score_role(gen, e1, x_np: np.ndarray, masks: np.ndarray, device, batch_rows: int,
               record_chunk: int = 256) -> dict[str, Any]:
    """p_full (N, 2) once per record and p_rec (N*k, 2) for all record-mask rows, through the fixed path."""
    n, k = masks.shape[:2]
    p_full, p_rec, violations = [], [], 0
    for s in range(0, n, record_chunk):
        x = torch.from_numpy(x_np[s:s + record_chunk]).to(device)
        p_full.append(score_e1(e1, x, batch_rows).cpu().numpy())
        xr = x.repeat_interleave(k, dim=0)
        lm = torch.from_numpy(masks[s:s + record_chunk].reshape(-1, 12)).to(device)
        x_hat, v = reconstruct(gen, xr, lm, batch_rows)
        violations += v
        p_rec.append(score_e1(e1, x_hat, batch_rows).cpu().numpy())
    return {"p_full": np.concatenate(p_full), "p_rec": np.concatenate(p_rec), "copy_through_violations": violations}


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def role_summary(labels: pd.DataFrame, keys: pd.DataFrame, outputs: list[str], protocol: dict[str, Any]) -> dict[str, Any]:
    df = keys[["row_id", "ecg_id", "patient_id", "n_observed"]].merge(labels, on="row_id", validate="one_to_one")
    reps, level, seed = (int(protocol["uncertainty"]["replicates"]), float(protocol["uncertainty"]["two_sided_level"]),
                         int(protocol["seed"]))
    pid = df["patient_id"].to_numpy()

    def boot(col: str) -> dict[str, Any]:
        return patient_cluster_bootstrap(df[col].to_numpy(float), pid, replicates=reps, seed=seed, level=level)

    rec = df.groupby("ecg_id").agg(patient_id=("patient_id", "first"), D1_any=("D1", "max"))
    out: dict[str, Any] = {
        "rows": int(len(df)), "records": int(df.ecg_id.nunique()), "patients": int(df.patient_id.nunique()),
        "D1_rows": int(df.D1.sum()), "D1_rate_rows": boot("D1"),
        "records_with_any_D1": int(rec.D1_any.sum()), "patients_with_any_D1": int(rec.groupby("patient_id").D1_any.max().sum()),
    }
    for y in outputs:
        pos = df[f"y_{y}"] == 1
        out[y] = {
            "label_positive_records": int(df.loc[pos, "ecg_id"].nunique()),
            "label_positive_patients": int(df.loc[pos, "patient_id"].nunique()),
            "z_full_positive_records": int(df.loc[df[f"z_full_{y}"] == 1, "ecg_id"].nunique()),
            "z_rec_positive_rows": int(df[f"z_rec_{y}"].sum()),
            "D1_miss_rows": int(df[f"D1_miss_{y}"].sum()), "D1_false_rows": int(df[f"D1_false_{y}"].sum()),
            "D1_miss_rate_rows": boot(f"D1_miss_{y}"), "D1_false_rate_rows": boot(f"D1_false_{y}"),
            # protocol Section 2 descriptive split of disagreeing rows against PTB-XL labels (not adjudicated)
            "disagree_label_correct_to_incorrect": int(((df[f"z_full_{y}"] == df[f"y_{y}"]) & (df[f"z_rec_{y}"] != df[f"y_{y}"])).sum()),
            "disagree_label_incorrect_to_correct": int(((df[f"z_full_{y}"] != df[f"y_{y}"]) & (df[f"z_rec_{y}"] == df[f"y_{y}"])).sum()),
        }
    flips = [((df[f"z_full_{y}"] != df[f"z_rec_{y}"])) for y in outputs]
    wrong_rec = [df[f"z_rec_{y}"] != df[f"y_{y}"] for y in outputs]
    wrong_full = [df[f"z_full_{y}"] != df[f"y_{y}"] for y in outputs]
    d1_rows = df.D1 == 1
    out["D1_rows_both_full_and_rec_label_incorrect_any_output"] = int((d1_rows & np.logical_or.reduce(wrong_rec)
                                                                      & np.logical_or.reduce(wrong_full)).sum())
    out["D1_rows_disagreeing_in_both_outputs"] = int(np.logical_and.reduce(flips).sum())
    bins = {}
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 11)):
        sel = (df.n_observed >= lo) & (df.n_observed <= hi)
        bins[f"{lo}-{hi}"] = {"rows": int(sel.sum()), "D1_rows": int(df.D1[sel].sum()),
                              "D1_rate": float(df.D1[sel].mean()) if sel.any() else None}
    out["by_n_observed_descriptive"] = bins
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _ro(p: Path) -> bool:
    return not os.access(p, os.W_OK)


def _rel(p: Path, repo: Path) -> str:
    return p.relative_to(repo).as_posix() if p.is_relative_to(repo) else p.as_posix()


def preflight(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Hash, lock, role and environment checks. Raises on the first failure; returns the verified inputs."""
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    protocol_path = repo / "configs" / "protocol_v0_1.yaml"
    lock = json.loads((repo / "artifacts" / "locks" / "protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    checks = []

    def ok(name: str, cond: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if cond else "FAIL", "detail": detail})
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        if not cond:
            raise RuntimeError(f"preflight failed: {name}: {detail}")

    psha = io.sha256_file(protocol_path)
    ok("protocol_hash_matches_lock_and_config", psha == lock["sha256"] == cfg["frozen"]["protocol_sha256"], psha)
    ok("protocol_and_lock_read_only", _ro(protocol_path) and _ro(repo / "artifacts/locks/protocol_v0_1.lock.json"), "read-only")
    protocol = io.load_config(protocol_path)
    mpath = repo / audit_cfg["split"]["manifest_dir"] / f"{audit_cfg['split']['frozen_manifest_name']}.csv"
    msha = io.sha256_file(mpath)
    ok("frozen_manifest_hash_and_read_only", msha == audit_cfg["split"]["frozen_manifest_sha256"] and _ro(mpath), msha)
    g = repo / cfg["frozen"]["g_checkpoint"]
    gsha = io.sha256_file(g)
    ok("G_v0_1_hash_and_read_only", gsha == cfg["frozen"]["g_sha256"] and _ro(g), gsha)
    e1_summary_path = repo / cfg["frozen"]["e1_run"] / "summary.json"
    e1s = json.loads(e1_summary_path.read_text(encoding="utf-8"))
    e1_ckpt = repo / e1s["selected_checkpoint"]
    esha = io.sha256_file(e1_ckpt)
    ok("E1_v0_1_hash_matches_summary_and_config_and_read_only",
       esha == e1s["selected_checkpoint_sha256"] == cfg["frozen"]["e1_sha256"] and _ro(e1_ckpt), esha)
    ok("E1_eligible", e1s["eligible"] is True, e1s["eligible"])
    outputs = list(protocol["targets"]["primary_outputs"])
    tau = d1mod.load_thresholds(e1_summary_path, outputs)
    ok("thresholds_read_from_E1_summary", True, {k: repr(v) for k, v in tau.items()})
    roles = list(cfg["roles"])
    ok("roles_are_protocol_D1_roles_without_fold10",
       all(r in protocol["d1"]["roles"] for r in roles) and FOLD10_ROLE not in roles and "A_fit" not in roles, roles)
    ok("no_threshold_lock_yet_so_fold10_stays_closed", not (repo / protocol["threshold"]["lock_file"]).exists(),
       protocol["threshold"]["lock_file"])
    manifest = load_frozen_manifest(audit_cfg, repo)
    pats = {r: set(manifest.loc[manifest.role == r, "patient_id"]) for r in manifest.role.unique()}
    overlap = max(len(pats[a] & pats[b]) for a, b in combinations(sorted(pats), 2))
    ok("all_roles_patient_disjoint_by_manifest", overlap == 0, {"max_overlap": overlap})
    out_dir = repo / cfg["output_dir"]
    ok("output_dir_does_not_exist", not out_dir.exists(), _rel(out_dir, repo))
    ok("cuda_available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    return {"checks": checks, "protocol": protocol, "audit_cfg": audit_cfg, "manifest": manifest, "tau": tau,
            "outputs": outputs, "e1_summary": e1s, "e1_ckpt": e1_ckpt, "g": g,
            "hashes": {"protocol": psha, "manifest": msha, "g": gsha, "e1": esha,
                       "e1_summary": io.sha256_file(e1_summary_path)}}


def run(repo: Path, cfg_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = io.load_config(cfg_path)
    tree_start = code_tree_sha256(repo)["tree_sha256"]
    pf = preflight(repo, cfg)
    protocol, manifest, tau, outputs = pf["protocol"], pf["manifest"], pf["tau"], pf["outputs"]
    seed, k = int(protocol["seed"]), int(protocol["d1"]["masks_per_record"])
    sc, eps = cfg["scoring"], float(cfg["boundary_eps"])
    set_deterministic(sc)
    device = torch.device("cuda")
    gen = l2g.ECGen(24).to(device)
    gen.load_state_dict(torch.load(pf["g"], map_location=device, weights_only=False)["generator"])
    gen.eval()
    e1 = ResNet1dWang(12, len(outputs)).to(device)
    e1.load_state_dict(torch.load(pf["e1_ckpt"], map_location=device, weights_only=False)["model"])
    e1.eval()
    state_before = {"G": state_hash(gen), "E1": state_hash(e1)}

    audit_cfg = pf["audit_cfg"]
    prep_path = repo / "configs" / "preprocessing.yaml"
    prep_cfg = io.load_config(prep_path)
    db_path = repo / audit_cfg["dataset"]["database_csv"]
    db = io.load_database(db_path)
    root = resolve_waveform_root(audit_cfg, repo)
    combos = mz.l2g_lead_combinations()
    combo_index = {c: i for i, c in enumerate(combos)}
    out = repo / cfg["output_dir"]
    reports = repo / cfg.get("reports_dir", "artifacts/reports")
    for sub in ("keys", "offline_labels"):
        (out / sub).mkdir(parents=True)

    summaries, boundary_rows, repro, cache_meta, extra = {}, [], {}, {}, {}
    for role in cfg["roles"]:
        rows = select_d1_role(manifest, role, protocol)
        ids = rows["ecg_id"].tolist()
        key = cachemod.cache_key(prep_path, db_path, audit_cfg["split"]["frozen_manifest_sha256"], role, ids)
        signals, index, meta = cachemod.ensure_cache(role, rows, root, db, prep_cfg, key, Path(cfg["data"]["cache_dir"]))
        if list(index["ecg_id"]) != ids:
            raise RuntimeError(f"{role}: cache order differs from the manifest order")
        cache_meta[role] = {"signals_sha256": meta["signals_sha256"], "shape": meta["shape"], "reused": meta.get("reused")}
        x_np = windows_at(signals, prep_cfg, int(cfg["data"]["window_offset"]))
        combos_per_record = [mz.evaluation_combinations(e, seed, k=k) for e in ids]
        masks = np.stack([np.stack([mz.lead_mask(c) for c in cs]) for cs in combos_per_record])
        if not np.array_equal(masks, validation_lead_masks(ids, seed, k)):
            raise RuntimeError("mask generation differs from the M2c/M3 evaluation masks")
        tr = time.perf_counter()
        sc_out = score_role(gen, e1, x_np, masks, device, int(sc["batch_rows"]))
        if sc_out["copy_through_violations"]:
            raise RuntimeError(f"{role}: {sc_out['copy_through_violations']} observed samples changed by copy-through")
        p_full_rec, p_rec = sc_out["p_full"], sc_out["p_rec"]
        p_full = np.repeat(p_full_rec, k, axis=0)  # the SAME stored value for all k rows of a record
        n = len(ids)
        y_rec = target_matrix(db, ids, protocol)
        lm = masks.reshape(-1, 12)
        flat_combos = [c for cs in combos_per_record for c in cs]
        keys = pd.DataFrame({
            "row_id": [f"{role}:{e}:{i}" for e in ids for i in range(k)],
            "ecg_id": np.repeat(ids, k), "patient_id": np.repeat(rows["patient_id"].to_numpy(), k),
            "role": role, "strat_fold": np.repeat(rows["strat_fold"].to_numpy(), k),
            "mask_index": np.tile(np.arange(k), n), "mask_id": [combo_index[c] for c in flat_combos],
            "observed_leads": [";".join(c) for c in flat_combos], "n_observed": lm.sum(axis=1)})
        labels = pd.DataFrame({"row_id": keys["row_id"]})
        for j, y in enumerate(outputs):
            labels[f"y_{y}"] = np.repeat(y_rec[:, j], k)
            labels[f"p_full_{y}"] = p_full[:, j]
            labels[f"p_rec_{y}"] = p_rec[:, j]
            labels[f"tau_{y}"] = np.float32(tau[y])
        for name, col in d1mod.d1_columns(p_full, p_rec, tau, outputs).items():
            labels[name] = col
        for j, y in enumerate(outputs):
            labels[f"near_tau_full_{y}"] = d1mod.near_threshold(p_full[:, j], tau[y], eps).astype(np.uint8)
            labels[f"near_tau_rec_{y}"] = d1mod.near_threshold(p_rec[:, j], tau[y], eps).astype(np.uint8)
        complete = pd.DataFrame({"ecg_id": ids, "patient_id": rows["patient_id"].to_numpy(), "role": role})
        for j, y in enumerate(outputs):
            complete[f"y_{y}"] = y_rec[:, j]
            complete[f"p_full_{y}"] = p_full_rec[:, j]
        keys.to_csv(out / "keys" / f"{role}_rows.csv", index=False)
        labels.to_csv(out / "offline_labels" / f"{role}_OFFLINE_d1_labels.csv", index=False, float_format="%.9g")
        complete.to_csv(out / "offline_labels" / f"{role}_OFFLINE_e1_complete_by_record.csv", index=False, float_format="%.9g")

        # boundary cases (|p - tau| <= eps), records for p_full and rows for p_rec
        for j, y in enumerate(outputs):
            for kind, arr, ident in (("p_full (per record)", p_full_rec[:, j], [(e, "all") for e in ids]),
                                     ("p_rec (per row)", p_rec[:, j], list(zip(keys.ecg_id, keys.mask_index)))):
                for i in np.flatnonzero(d1mod.near_threshold(arr, tau[y], eps)):
                    boundary_rows.append({"role": role, "output": y, "score": kind, "ecg_id": int(ident[i][0]),
                                          "mask_index": ident[i][1], "probability": f"{np.float32(arr[i]):.9g}",
                                          "tau": f"{tau[y]:.9g}", "difference": float(np.float32(arr[i])) - float(tau[y]),
                                          "decision": int(np.float32(arr[i]) >= tau[y])})
        # reproducibility: first rows re-scored in reverse order through the same path
        m = min(int(cfg["reproducibility_check_rows"]), len(keys))
        rec_needed = int(np.ceil(m / k))
        sel = np.arange(rec_needed)[::-1]
        chk = score_role(gen, e1, x_np[sel], masks[sel], device, int(sc["batch_rows"]))
        pf_ref, pr_ref = p_full_rec[sel], p_rec.reshape(n, k, -1)[sel].reshape(-1, len(outputs))
        repro[role] = {"records_rescored": int(rec_needed), "rows_rescored": int(rec_needed * k),
                       "max_abs_diff_p_full": float(np.abs(chk["p_full"] - pf_ref).max()),
                       "max_abs_diff_p_rec": float(np.abs(chk["p_rec"] - pr_ref).max()),
                       "decisions_identical": bool(np.array_equal(chk["p_full"] >= np.array([tau[y] for y in outputs]),
                                                                  pf_ref >= np.array([tau[y] for y in outputs]))
                                                   and np.array_equal(chk["p_rec"] >= np.array([tau[y] for y in outputs]),
                                                                      pr_ref >= np.array([tau[y] for y in outputs])))}
        s = role_summary(labels, keys, outputs, protocol)
        s.update({"sanity_only": role in cfg["sanity_only_roles"], "scoring_seconds": round(time.perf_counter() - tr, 1),
                  "copy_through_violations": 0, "near_tau_full_records": {y: int(d1mod.near_threshold(p_full_rec[:, j], tau[y], eps).sum()) for j, y in enumerate(outputs)},
                  "near_tau_rec_rows": {y: int(d1mod.near_threshold(p_rec[:, j], tau[y], eps).sum()) for j, y in enumerate(outputs)}})
        summaries[role] = s
        if role == "C_dev":  # compare with the E1 run's own C_dev predictions (different batch path)
            ref = pd.read_csv(repo / cfg["frozen"]["e1_run"] / "c_dev_predictions_selected.csv").set_index("ecg_id").loc[ids]
            extra["c_dev_vs_e1_run_predictions"] = {
                y: {"max_abs_diff": float(np.abs(ref[f"prob_{y}"].to_numpy(np.float32) - p_full_rec[:, j]).max()),
                    "decision_flips": [int(e) for e, a, b in zip(ids, ref[f"prob_{y}"].to_numpy(np.float32) >= tau[y],
                                                                p_full_rec[:, j] >= tau[y]) if a != b]}
                for j, y in enumerate(outputs)}
        print(f"[{role}] rows {s['rows']} D1 rows {s['D1_rows']} rate {s['D1_rate_rows']['point']:.4f} "
              f"[{s['D1_rate_rows']['ci_low']:.4f}, {s['D1_rate_rows']['ci_high']:.4f}] | {s['scoring_seconds']}s", flush=True)

    state_after = {"G": state_hash(gen), "E1": state_hash(e1)}
    if state_after != state_before:
        raise RuntimeError("model state changed during scoring")
    pd.DataFrame(boundary_rows, columns=["role", "output", "score", "ecg_id", "mask_index", "probability", "tau", "difference",
                                         "decision"]).to_csv(reports / "m4a_threshold_boundary_cases.csv", index=False)
    (out / "README.md").write_text(
        "# OFFLINE D1 artifacts (M4a)\n\n`offline_labels/` holds reference-derived values (E1 on the complete ECG, D1, PTB-XL labels).\n"
        "They are training TARGETS and evaluation data only. Never use any column of these files as a gate feature; use\n"
        "`src.eval.d1.attach_d1_target`, which copies only `D1` by `row_id`. `keys/` holds identifiers and mask metadata.\n"
        "Fold 10 is not present and must not be generated before the threshold lock (protocol Section 7).\n", encoding="utf-8")
    nine_a = summaries.get("9a_calibration", {})
    summary = {
        "milestone": "M4a", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": _rel(cfg_path, repo), "config_sha256": io.sha256_file(cfg_path),
        "roles": list(cfg["roles"]), "sanity_only_roles": list(cfg["sanity_only_roles"]), "fold10_loaded": False,
        "outputs": outputs, "tau_float32": {y: f"{tau[y]:.9g}" for y in outputs}, "decision_rule": "p >= tau (float32)",
        "masks_per_record": k, "mask_rule": f"evaluation_combinations(ecg_id, seed={seed}, k={k})",
        "scoring": dict(sc), "boundary_eps": eps, "inputs": pf["hashes"], "preflight": pf["checks"],
        "model_state_unchanged": state_after == state_before, "model_state_sha256": state_after,
        "cache": cache_meta, "reproducibility": repro, **extra, "per_role": summaries,
        "calibration_readiness_9a": {"D1_rows": nine_a.get("D1_rows"), "D1_patients": nine_a.get("patients_with_any_D1"),
                                     "rule": f"Platt only if 9a has >= {protocol['calibration']['min_d1_events']} D1 events",
                                     "platt_permitted": (nine_a.get("D1_rows") or 0) >= int(protocol["calibration"]["min_d1_events"])},
        "mask_generator_sha256": io.sha256_file(Path(mz.__file__)),
        "combos_sha256": hashlib.sha256(json.dumps(combos).encode()).hexdigest(),
        "code_tree_at_start": tree_start, "code_tree_at_end": code_tree_sha256(repo)["tree_sha256"],
        "environment": environment(), "total_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "hash_manifest.json")
    files.append(reports / "m4a_threshold_boundary_cases.csv")
    manifest_out = {"created_utc": summary["created_utc"], "inputs": pf["hashes"],
                    "files": {_rel(p, repo): io.sha256_file(p) for p in files}}
    (out / "hash_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    for p in files + [out / "hash_manifest.json"]:
        if p.parent.name in ("keys", "offline_labels"):
            os.chmod(p, 0o444)  # produced labels are frozen read-only
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M4a offline D1 generation (no training).")
    ap.add_argument("--config", default="configs/d1_generation_v0_1.yaml")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    repo = Path(".").resolve()
    cfg_path = (repo / args.config).resolve()
    if args.preflight_only:
        pf = preflight(repo, io.load_config(cfg_path))
        rep = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": pf["checks"],
               "hashes": pf["hashes"], "code_tree_sha256": code_tree_sha256(repo)["tree_sha256"], "status": "PASS"}
        (repo / "artifacts/reports/preflight_m4a_d1.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        print("M4a preflight: PASS")
        return 0
    s = run(repo, cfg_path)
    print(json.dumps({r: {"rows": v["rows"], "D1_rows": v["D1_rows"]} for r, v in s["per_role"].items()}, indent=1))
    print("9a calibration readiness:", s["calibration_readiness_9a"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
