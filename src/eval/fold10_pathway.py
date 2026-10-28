"""Fold-10 pathway and M5 run (protocol v0.2 fold10_pathway; built in M4b-1 stage 04 under the owner's authorisation of
2026-10-24, which does NOT authorise any fold-10 access).

    .\\.venv\\Scripts\\python.exe -m src.eval.fold10_pathway dry_run      # role 9b_threshold only; fold 10 never touched
    .\\.venv\\Scripts\\python.exe -m src.eval.fold10_pathway final        # refused unless every precondition holds

Modes
* ``dry_run`` runs the whole pathway on 9b_threshold (a permitted development role): D1 regenerated through the fixed scoring
  path, G1/G2 features recomputed from the raw waveforms, M2 scores, calibrated gate scores, then the complete M5 analysis.
  Because 9b already has frozen outputs, every regenerated file is compared with its frozen counterpart. The M5 numbers of a
  dry run are NOT results: 9b chose the thresholds, so they are in-sample checks of the code only.
* ``final`` is the single fold-10 run. It refuses unless: role test_fold10; protocol v0.2 and threshold locks exist and
  verify; the locked 9b calibrated scores are reproduced bit for bit in this environment; the owner's separate final-run
  authorisation file exists; no fold-10 artifact or cache exists yet; the output folder does not exist.
Every existing refusal elsewhere (generate_d1, extract_features, m4b1) stays in place; this is the only code path that may
select fold-10 rows, and only in ``final`` mode.
"""
from __future__ import annotations

import argparse
import hashlib
import io as pyio
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import missingness as mz
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.eval import d1 as d1mod
from src.eval import descriptive_endpoints as de
from src.eval import m5
from src.eval import primary_decision as pdec
from src.gate import features as gf
from src.gate import m4b1
from src.gate.comparators import m2_scores
from src.utils.check_frozen_artifacts import FROZEN
from src.utils.provenance import code_tree_sha256, environment

REPO = Path(__file__).resolve().parents[2]
FOLD10 = "test_fold10"
DRY_RUN_ROLES = ("9b_threshold",)
FINAL_AUTHORISATION = "artifacts/authorisations/fold10_final_run.json"   # written only after the owner's explicit words
OUT = {"dry_run": "artifacts/m5/dry_run_9b_threshold", "final": "artifacts/m5/final_test_fold10"}
CACHE_DIR = Path("C:/ecg_cache/v0_1")
METHODS = list(m5.GATES)


# ------------------------------------------------------------------------------------------------------------------
# Guards (pure; synthetic tests)
# ------------------------------------------------------------------------------------------------------------------
def check_mode(mode: str, role: str, final_authorised: bool) -> None:
    if mode == "dry_run":
        if role == FOLD10 or role not in DRY_RUN_ROLES:
            raise RuntimeError(f"dry_run runs on {DRY_RUN_ROLES} only, never {role}")
    elif mode == "final":
        if role != FOLD10:
            raise RuntimeError("final mode evaluates test_fold10 only")
        if not final_authorised:
            raise RuntimeError("final mode refused: no explicit owner authorisation for the final fold-10 run")
    else:
        raise RuntimeError(f"unknown mode {mode}")


def final_authorisation_ok(repo: Path) -> tuple[bool, str]:
    """The owner's final-run authorisation, recorded in FINAL_AUTHORISATION, is valid only if the owner's own words contain
    exactly "AUTHORISE FINAL FOLD-10 RUN <SHA-256 of the threshold lock>" and the recorded hashes equal the live lock files."""
    p = repo / FINAL_AUTHORISATION
    if not p.exists():
        return False, "no authorisation file"
    a = json.loads(p.read_text(encoding="utf-8"))
    tl_sha = io.sha256_file(repo / m4b1.THRESHOLD_LOCK) if (repo / m4b1.THRESHOLD_LOCK).exists() else None
    phrase = f"AUTHORISE FINAL FOLD-10 RUN {tl_sha}"
    if tl_sha is None or phrase not in a.get("owner_statement", ""):
        return False, "the owner statement does not contain the exact phrase with the current threshold-lock hash"
    if a.get("threshold_lock_sha256") != tl_sha or a.get("protocol_v0_2_sha256") != io.sha256_file(repo / "configs/protocol_v0_2.yaml"):
        return False, "recorded hashes differ from the live lock files"
    return True, phrase


def check_rows(rows: pd.DataFrame, role: str) -> None:
    """Rows of the requested role only; fold-10 records if and only if the role is test_fold10."""
    if rows.empty or set(rows["role"]) != {role}:
        raise RuntimeError(f"rows are empty or not all of role {role}")
    is10 = rows["strat_fold"] == 10
    if (role == FOLD10 and not is10.all()) or (role != FOLD10 and is10.any()):
        raise RuntimeError(f"role {role}: fold-10 records where they must not be (or the reverse)")


def fold10_paths(repo: Path, cache_dir: Path) -> list[str]:
    """Every file or folder under artifacts/ or the cache whose name mentions test_fold10 (must be empty before the final run)."""
    found = []
    for base in (repo / "artifacts", cache_dir):
        if base.exists():
            found += [p.as_posix() for p in base.rglob("*") if FOLD10 in p.name]
    return sorted(found)


def csv_sha(df: pd.DataFrame, float_format: str | None) -> str:
    buf = pyio.StringIO()
    df.to_csv(buf, index=False, float_format=float_format, lineterminator="\r\n" if os.name == "nt" else "\n")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------------------
# Locks and the bitwise 9b check
# ------------------------------------------------------------------------------------------------------------------
class Check:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    def __call__(self, name: str, cond: bool, detail: Any) -> None:
        self.items.append({"check": name, "status": "PASS" if cond else "FAIL", "detail": detail})
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)
        if not cond:
            raise RuntimeError(f"pathway check failed: {name}")


def verify_locks(repo: Path, ok: Check) -> dict[str, Any]:
    plock = json.loads((repo / "artifacts/locks/protocol_v0_2.lock.json").read_text(encoding="utf-8"))
    ok("protocol_v0_2_lock", plock["status"] == "APPROVED" and plock["sha256"] == io.sha256_file(repo / "configs/protocol_v0_2.yaml"), plock["sha256"])
    tpath = repo / m4b1.THRESHOLD_LOCK
    ok("threshold_lock_exists_read_only", tpath.exists() and not os.access(tpath, os.W_OK), m4b1.THRESHOLD_LOCK)
    tl = json.loads(tpath.read_text(encoding="utf-8"))
    ok("threshold_lock_names_pinned_protocol", tl["protocol_v0_2"]["sha256"] == plock["sha256"]
       and tl["protocol_v0_2"]["lock_sha256"] == io.sha256_file(repo / "artifacts/locks/protocol_v0_2.lock.json"), tl["protocol_v0_2"]["sha256"])
    for rel, exp in FROZEN.items():
        ok(f"frozen_{Path(rel).name}", io.sha256_file(repo / rel) == exp, exp[:16])
    fit = json.loads((repo / m4b1.OUT / "01_fit/summary.json").read_text(encoding="utf-8"))
    cal = json.loads((repo / m4b1.OUT / "02_calibration/summary.json").read_text(encoding="utf-8"))
    ok("stage_records_match_lock", io.sha256_file(repo / m4b1.OUT / "01_fit/summary.json") == tl["stage_records"]["fit_summary_sha256"]
       and io.sha256_file(repo / m4b1.OUT / "02_calibration/summary.json") == tl["stage_records"]["calibration_summary_sha256"], "fit + calibration")
    for m in METHODS:
        ok(f"locked_method_{m}", m in tl["methods"], m)
        ok(f"calibrator_{m}", io.sha256_file(repo / cal["methods"][m]["calibrator"]["path"]) == tl["methods"][m]["calibrator_sha256"], m)
        if m in fit["methods"]:
            ok(f"model_{m}", io.sha256_file(repo / fit["methods"][m]["model"]["path"]) == tl["methods"][m]["model_sha256"], m)
        pdec.threshold_from_record(tl["methods"][m]["t"])
    env = m4b1.env_record()
    now = {"os": env["os"], "python": env["python"], **{k: env["packages"][k] for k in ("numpy", "scipy", "scikit-learn", "pandas", "joblib")}}
    ok("pinned_environment", now == tl["environment"], now)
    return tl


def reproduce_locked_9b(repo: Path, tl: dict[str, Any], ok: Check) -> None:
    inp = m4b1.Inputs(repo)
    cs, _ = m4b1.calibrated_scores(repo, inp, "9b_threshold")
    sha = csv_sha(cs, "%.17g")
    ok("locked_9b_calibrated_scores_bitwise", sha == tl["calibrated_9b_scores"]["sha256"] == io.sha256_file(repo / tl["calibrated_9b_scores"]["path"]), sha)


# ------------------------------------------------------------------------------------------------------------------
# D1, features and M2 for one role (same code paths and formats as M4a, M4b-0 and the comparators)
# ------------------------------------------------------------------------------------------------------------------
def d1_for_role(repo: Path, role: str, ok: Check) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import torch
    from src.data.ptbxl_waveforms import resolve_waveform_root
    from src.data.targets import target_matrix
    from src.eval import generate_d1 as gd
    from src.models import l2g
    from src.models.evaluators import ResNet1dWang
    from src.train import cache as cachemod
    from src.train.train_l2g import validation_lead_masks, windows_at

    cfg = io.load_config(repo / "configs/d1_generation_v0_1.yaml")
    p1 = io.load_config(repo / "configs/protocol_v0_1.yaml")
    audit = io.load_config(repo / "configs/data_audit.yaml")
    manifest = load_frozen_manifest(audit, repo)
    excluded = set(p1["targets"]["excluded_ecg_ids"])
    rows = manifest[(manifest["role"] == role) & ~manifest["ecg_id"].isin(excluded)].sort_values("ecg_id").reset_index(drop=True)
    check_rows(rows, role)
    others = set(manifest.loc[manifest["role"] != role, "patient_id"])
    ok("role_patients_disjoint_from_other_roles", not (set(rows["patient_id"]) & others), int(rows["patient_id"].nunique()))
    outputs = list(p1["targets"]["primary_outputs"])
    e1s = json.loads((repo / cfg["frozen"]["e1_run"] / "summary.json").read_text(encoding="utf-8"))
    tau = d1mod.load_thresholds(repo / cfg["frozen"]["e1_run"] / "summary.json", outputs)
    seed, k, sc = int(p1["seed"]), int(p1["d1"]["masks_per_record"]), cfg["scoring"]
    gd.set_deterministic(sc)
    device = torch.device("cuda")
    gen = l2g.ECGen(24).to(device)
    gen.load_state_dict(torch.load(repo / cfg["frozen"]["g_checkpoint"], map_location=device, weights_only=False)["generator"])
    gen.eval()
    e1 = ResNet1dWang(12, len(outputs)).to(device)
    e1.load_state_dict(torch.load(repo / e1s["selected_checkpoint"], map_location=device, weights_only=False)["model"])
    e1.eval()
    prep_path, db_path = repo / "configs/preprocessing.yaml", repo / audit["dataset"]["database_csv"]
    prep_cfg, db = io.load_config(prep_path), io.load_database(db_path)
    root = resolve_waveform_root(audit, repo)
    ids = rows["ecg_id"].tolist()
    key = cachemod.cache_key(prep_path, db_path, audit["split"]["frozen_manifest_sha256"], role, ids)
    signals, index, meta = cachemod.ensure_cache(role, rows, root, db, prep_cfg, key, Path(cfg["data"]["cache_dir"]))
    ok("cache_order", list(index["ecg_id"]) == ids, meta["signals_sha256"][:16])
    x_np = windows_at(signals, prep_cfg, int(cfg["data"]["window_offset"]))
    combos = [mz.evaluation_combinations(e, seed, k=k) for e in ids]
    masks = np.stack([np.stack([mz.lead_mask(c) for c in cs]) for cs in combos])
    ok("protocol_masks", np.array_equal(masks, validation_lead_masks(ids, seed, k)), f"{len(ids)} records x {k}")
    t0 = time.perf_counter()
    out = gd.score_role(gen, e1, x_np, masks, device, int(sc["batch_rows"]))
    ok("copy_through", out["copy_through_violations"] == 0, 0)
    p_full = np.repeat(out["p_full"], k, axis=0)
    p_rec = out["p_rec"]
    combo_index = {c: i for i, c in enumerate(mz.l2g_lead_combinations())}
    flat = [c for cs in combos for c in cs]
    lm = masks.reshape(-1, 12)
    n = len(ids)
    keys = pd.DataFrame({"row_id": [f"{role}:{e}:{i}" for e in ids for i in range(k)], "ecg_id": np.repeat(ids, k),
                         "patient_id": np.repeat(rows["patient_id"].to_numpy(), k), "role": role,
                         "strat_fold": np.repeat(rows["strat_fold"].to_numpy(), k), "mask_index": np.tile(np.arange(k), n),
                         "mask_id": [combo_index[c] for c in flat], "observed_leads": [";".join(c) for c in flat],
                         "n_observed": lm.sum(axis=1)})
    y_rec = target_matrix(db, ids, p1)
    labels = pd.DataFrame({"row_id": keys["row_id"]})
    for j, y in enumerate(outputs):
        labels[f"y_{y}"] = np.repeat(y_rec[:, j], k)
        labels[f"p_full_{y}"] = p_full[:, j]
        labels[f"p_rec_{y}"] = p_rec[:, j]
        labels[f"tau_{y}"] = np.float32(tau[y])
    for name, col in d1mod.d1_columns(p_full, p_rec, tau, outputs).items():
        labels[name] = col
    eps = float(cfg["boundary_eps"])
    for j, y in enumerate(outputs):
        labels[f"near_tau_full_{y}"] = d1mod.near_threshold(p_full[:, j], tau[y], eps).astype(np.uint8)
        labels[f"near_tau_rec_{y}"] = d1mod.near_threshold(p_rec[:, j], tau[y], eps).astype(np.uint8)
    info = {"records": n, "rows": len(keys), "d1_rows": int(labels["D1"].sum()), "scoring_seconds": round(time.perf_counter() - t0, 1),
            "cache_signals_sha256": meta["signals_sha256"], "tau": {y: f"{tau[y]:.9g}" for y in outputs}}
    return keys, labels, info


def features_for_role(repo: Path, keys: pd.DataFrame, seed: int, k: int, ok: Check) -> pd.DataFrame:
    from src.data.ptbxl_waveforms import resolve_waveform_root
    from src.gate.extract_features import _per_lead_table

    audit = io.load_config(repo / "configs/data_audit.yaml")
    db_files = pd.read_csv(repo / audit["dataset"]["database_csv"], usecols=["ecg_id", "filename_hr"]).set_index("ecg_id")["filename_hr"]
    root = resolve_waveform_root(audit, repo)
    observed = [s.split(";") for s in keys["observed_leads"]]
    ok("feature_masks_are_protocol_masks", all(list(mz.evaluation_combinations(int(e), seed, k=k)[int(i)]) == o
                                               for e, i, o in zip(keys["ecg_id"], keys["mask_index"], observed)), len(keys))
    ids = keys["ecg_id"].drop_duplicates().tolist()
    with ProcessPoolExecutor(max_workers=8) as pool:
        tables = list(pool.map(_per_lead_table, [(str(root), db_files.loc[e]) for e in ids], chunksize=16))
    per_record = dict(zip(ids, tables))
    rows = [{**gf.g1_features(obs), **gf.aggregate([dict(zip(gf.SQIS, per_record[e][gf.LEAD_INDEX[lead]])) for lead in obs])}
            for e, obs in zip(keys["ecg_id"], observed)]
    feats = pd.DataFrame(rows, columns=gf.G1_COLUMNS + gf.G2_COLUMNS)
    feats.insert(0, "row_id", keys["row_id"].to_numpy())
    return feats


# ------------------------------------------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------------------------------------------
def run(repo: Path, mode: str, role: str, replicates: int | None = None, label: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    ok = Check()
    final_auth, why = final_authorisation_ok(repo)
    check_mode(mode, role, final_auth)
    ok("mode_and_role", True, {"mode": mode, "role": role, "final_authorisation": why})
    before = fold10_paths(repo, CACHE_DIR)
    ok("no_fold10_artifact_or_cache_before", before == [], before)
    out_rel = OUT[mode] + (f"_{label}" if label and mode == "dry_run" else "")
    out = repo / out_rel
    ok("output_folder_absent", not out.exists(), out_rel)
    code = {rel: io.sha256_file(repo / rel) for rel in ("src/eval/fold10_pathway.py", "src/eval/m5.py", "src/gate/m4b1.py",
                                                        "src/eval/selective.py", "src/eval/descriptive_endpoints.py",
                                                        "src/eval/primary_decision.py", "src/eval/generate_d1.py",
                                                        "src/gate/features.py", "src/gate/extract_features.py")}
    tl = verify_locks(repo, ok)
    reproduce_locked_9b(repo, tl, ok)
    p2 = io.load_config(repo / "configs/protocol_v0_2.yaml")
    seed, k = int(p2["seed"]), int(p2["d1"]["masks_per_record"])
    tree = code_tree_sha256(repo)["tree_sha256"]

    keys, labels, d1_info = d1_for_role(repo, role, ok)
    feats = features_for_role(repo, keys, seed, k, ok)
    outputs = [c.removeprefix("tau_") for c in labels.columns if c.startswith("tau_")]
    tau = np.array([labels[f"tau_{y}"].iloc[0] for y in outputs])
    m2 = pd.DataFrame({"row_id": labels["row_id"], "e1_recon_margin_score": m2_scores(labels[[f"p_rec_{y}" for y in outputs]].to_numpy(), tau)})

    # calibrated scores with the locked models and calibrators (feature whitelist per method; D1 never joined)
    _, models = m4b1.load_models(repo)
    _, cals = m4b1.load_calibrators(repo)
    scores = {"row_id": feats["row_id"]}
    for m in METHODS:
        if m == m4b1.M2:
            raw = m2["e1_recon_margin_score"].to_numpy(np.float64)
        else:
            mat = m4b1.method_matrix(feats, m, p2)
            raw = m4b1.score(models[m], mat.drop(columns="row_id"))
        scores[m] = m4b1.calibrated(cals[m], raw, m)
    scores = pd.DataFrame(scores)
    thresholds = {m: pdec.threshold_from_record(tl["methods"][m]["t"]) for m in METHODS}

    table = de.align_rows(keys[["row_id", "patient_id", "mask_id", "n_observed"]], labels[de.required_offline_columns()], scores)
    ok("rows_aligned", len(table) == len(keys) == len(feats) == len(scores), len(table))
    results = m5.evaluate(table, thresholds, p2, replicates=replicates)

    comparisons = {}
    if mode == "dry_run":                                   # every regenerated 9b file must equal its frozen counterpart
        frozen = {"keys": (keys, None, f"artifacts/d1/v0_1/keys/{role}_rows.csv"),
                  "offline_labels": (labels, "%.9g", f"artifacts/d1/v0_1/offline_labels/{role}_OFFLINE_d1_labels.csv"),
                  "features": (feats, "%.17g", f"artifacts/gate/v0_2/features/{role}_features.csv"),
                  "M2": (m2, "%.17g", f"artifacts/gate/v0_2/comparators/{role}_M2_e1_recon_margin.csv"),
                  "calibrated_scores": (scores, "%.17g", tl["calibrated_9b_scores"]["path"])}
        for name, (df, ff, rel) in frozen.items():
            comparisons[name] = {"frozen_file": rel, "frozen_sha256": io.sha256_file(repo / rel), "regenerated_sha256": csv_sha(df, ff)}
            comparisons[name]["identical"] = comparisons[name]["frozen_sha256"] == comparisons[name]["regenerated_sha256"]
        cov = {m: results["point"]["locked"][m]["coverage"] for m in METHODS}
        comparisons["locked_coverage_equals_lock"] = {m: cov[m] == tl["methods"][m]["coverage_9b"] for m in METHODS}

    after = fold10_paths(repo, CACHE_DIR)
    if mode == "dry_run":
        ok("no_fold10_artifact_or_cache_after", after == [], after)
    out.mkdir(parents=True)
    keys.to_csv(out / "keys.csv", index=False)
    labels.to_csv(out / "offline_labels.csv", index=False, float_format="%.9g")
    feats.to_csv(out / "features.csv", index=False, float_format="%.17g")
    scores.to_csv(out / "calibrated_scores.csv", index=False, float_format="%.17g")
    reps = results.pop("replicate_arrays")
    pd.DataFrame(reps).to_csv(out / "h1_h2_replicates.csv", index=False, float_format="%.17g")
    summary = {"mode": mode, "role": role, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "not_a_result": mode == "dry_run", "note": ("DRY RUN on 9b: code check only; 9b chose the thresholds, so these "
                                                           "numbers are in-sample and are not study results") if mode == "dry_run" else
               "single authorised fold-10 run; outputs frozen", "checks": ok.items, "d1": d1_info, "comparisons_with_frozen": comparisons,
               "fold10_paths_before": before, "fold10_paths_after": after, "threshold_lock_sha256": io.sha256_file(repo / m4b1.THRESHOLD_LOCK),
               "code_tree_sha256": tree, "code_files_sha256": code, "replicates": results["replicates"],
               "environment": environment(), "seconds": round(time.perf_counter() - t0, 1)}
    (out / "m5_results.json").write_text(json.dumps(results, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    files = {p.relative_to(repo).as_posix(): io.sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "hash_manifest.json").write_text(json.dumps(files, indent=1) + "\n", encoding="utf-8")
    for p in out.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o444)
    print(json.dumps({"comparisons": comparisons, "H1": {k: results["primary"]["H1"].get(k) for k in ("point", "reject", "upper_bound")},
                      "seconds": summary["seconds"]}, indent=1, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fold-10 pathway: dry run on 9b, or the single authorised final run.")
    ap.add_argument("mode", choices=["dry_run", "final"])
    ap.add_argument("--label", help="dry_run only: suffix of the output folder, so a repeated dry run never overwrites an earlier one")
    args = ap.parse_args(argv)
    run(REPO, args.mode, "9b_threshold" if args.mode == "dry_run" else FOLD10, label=args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
