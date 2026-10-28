"""Milestone 1: build a reproducible, patient-disjoint split manifest for PTB-XL.

Run from the repository root (after the audit):
    .\\.venv\\Scripts\\python.exe -m src.data.make_split_manifest --config configs/data_audit.yaml

Design (see docs/decision_log.md, D-005 to D-008):
  * Official folds are respected: a patient's role is chosen only from roles
    allowed for that patient's official fold.
  * Fold 10 -> 'test_fold10' for every record. No shuffling; its labels are not used.
  * Fold 9  -> patient-level split into '9a_calibration' / '9b_threshold'.
  * Folds 1-8 -> patient-level split into 'A_fit' / 'B_gate' / 'C_dev'.
  * Splits are made on patient_id (never on individual records) and are
    stratified by the patient's BBB code pattern so rare codes are spread
    across roles. A fixed seed makes the result identical on every run.

The manifest is a DRAFT. It may only be frozen after (1) the audit passes,
(2) every split check below passes, and (3) the BBB target definition is approved.

Exit codes: 0 = checks passed (draft written), 1 = a check failed (no manifest
written), 2 = missing input file.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import ptbxl_io as io
from src.data.audit_ptbxl import environment_info

ROLE_GROUP_DEV, ROLE_GROUP_FOLD9, ROLE_GROUP_TEST = "development", "fold9", "test"


# --------------------------------------------------------------------------- #
# Input validation (refuse to split broken data)
# --------------------------------------------------------------------------- #
def validate_split_inputs(db: pd.DataFrame) -> list[str]:
    """Return a list of problems that make a patient-level split unsafe (empty = OK)."""
    problems = []
    if db["ecg_id"].duplicated().any():
        problems.append("duplicate ecg_id values")
    pid = pd.to_numeric(db["patient_id"], errors="coerce")
    if pid.isna().any() or (pid <= 0).any() or (pid != np.floor(pid)).any():
        problems.append("missing, non-integer or non-positive patient_id values")
    fold = pd.to_numeric(db["strat_fold"], errors="coerce")
    if not fold.isin(range(1, 11)).all():
        problems.append("strat_fold values outside 1..10")
    if (db.groupby("patient_id")["strat_fold"].nunique() > 1).any():
        problems.append("at least one patient appears in more than one official fold")
    return problems


# --------------------------------------------------------------------------- #
# Core allocation logic
# --------------------------------------------------------------------------- #
def allocate_counts(n: int, proportions: dict[str, float], rng: np.random.Generator) -> dict[str, int]:
    """Split n items into roles. Each role gets floor(n*p) items; the leftover
    items go to roles chosen at random in proportion to their fractional parts.
    So the EXPECTED count of every role is exactly n*p, even when n is tiny
    (e.g. a stratum with a single rare-label patient)."""
    names = list(proportions)
    p = np.array([proportions[k] for k in names], dtype=float)
    if (p < 0).any() or not np.isclose(p.sum(), 1.0):
        raise ValueError(f"role proportions must be >= 0 and sum to 1, got {proportions}")
    exact = n * p
    counts = np.floor(exact + 1e-9).astype(int)
    leftover = n - int(counts.sum())
    if leftover < 0:
        raise RuntimeError(f"allocation overshoot for n={n}, proportions={proportions}")
    if leftover > 0:
        frac = np.clip(exact - counts, 0, None)
        chosen = rng.choice(len(names), size=leftover, replace=False, p=frac / frac.sum())
        counts[chosen] += 1
    return dict(zip(names, counts.tolist()))


def patient_table(
    db: pd.DataFrame, codes: pd.Series, target_codes: list[str], min_likelihood: float
) -> pd.DataFrame:
    """One row per patient: fold, number of records, and BBB pattern string."""
    labels = io.label_matrix(codes, target_codes, min_likelihood)
    flags = labels.groupby(db["patient_id"]).any()
    pattern = pd.Series(
        ["+".join(c for c in target_codes if row[c]) or "none" for row in flags.to_dict("records")],
        index=flags.index, dtype=object,
    )
    table = db.groupby("patient_id").agg(
        strat_fold=("strat_fold", "first"), n_records=("ecg_id", "size")
    )
    table["bbb_pattern"] = pattern
    return table.reset_index()


def assign_roles(
    patients: pd.DataFrame,
    proportions: dict[str, float],
    strata_columns: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign each patient one role, stratum by stratum, in a deterministic order."""
    out = []
    ordered = patients.sort_values(strata_columns + ["patient_id"])
    for key, group in ordered.groupby(strata_columns, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ids = group["patient_id"].to_numpy()
        shuffled = ids[rng.permutation(len(ids))]
        counts = allocate_counts(len(ids), proportions, rng)
        roles = np.repeat(list(counts), list(counts.values()))
        stratum = "|".join(f"{c}={v}" for c, v in zip(strata_columns, key))
        out.append(pd.DataFrame({"patient_id": shuffled, "role": roles, "stratum": stratum}))
    return pd.concat(out, ignore_index=True)


def make_manifest(db: pd.DataFrame, codes: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    """Build the record-level manifest (one row per ecg_id)."""
    sp, target_codes = config["split"], list(config["targets"]["codes"])
    rng = np.random.default_rng(int(sp["seed"]))
    test_fold, fold9 = int(sp["test_fold"]), int(sp["fold9"])
    dev_folds = [int(f) for f in sp["development_folds"]]

    # Fold 10 is excluded BEFORE any label is read: its patients are never stratified.
    not_test = db["strat_fold"] != test_fold
    patients = patient_table(
        db[not_test], codes[not_test], target_codes, float(sp["stratify_min_likelihood"])
    )
    dev = assign_roles(
        patients[patients["strat_fold"].isin(dev_folds)],
        dict(sp["development_roles"]), ["strat_fold", "bbb_pattern"], rng,
    )
    dev["role_group"] = ROLE_GROUP_DEV
    f9 = assign_roles(
        patients[patients["strat_fold"] == fold9],
        dict(sp["fold9_roles"]), ["bbb_pattern"], rng,
    )
    f9["role_group"] = ROLE_GROUP_FOLD9
    test_patients = db.loc[~not_test, "patient_id"].drop_duplicates()
    test = pd.DataFrame({
        "patient_id": test_patients.to_numpy(), "role": sp["test_role"],
        "stratum": "fold10_not_stratified", "role_group": ROLE_GROUP_TEST,
    })
    roles = pd.concat([dev, f9, test], ignore_index=True)

    manifest = db[["ecg_id", "patient_id", "strat_fold"]].merge(
        roles, on="patient_id", how="left", validate="many_to_one"
    )
    manifest["ecg_id"] = manifest["ecg_id"].astype(int)
    manifest["patient_id"] = manifest["patient_id"].astype(int)
    manifest["strat_fold"] = manifest["strat_fold"].astype(int)
    return manifest.sort_values("ecg_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Checks that must ALL pass before the manifest is written
# --------------------------------------------------------------------------- #
def check_manifest(manifest: pd.DataFrame, db: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    sp = config["split"]
    test_fold, fold9, test_role = int(sp["test_fold"]), int(sp["fold9"]), sp["test_role"]
    dev_roles, f9_roles = set(sp["development_roles"]), set(sp["fold9_roles"])
    dev_folds = {int(f) for f in sp["development_folds"]}
    results = []

    def add(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("every_record_assigned_once",
        len(manifest) == len(db) and manifest["ecg_id"].is_unique
        and set(manifest["ecg_id"]) == set(db["ecg_id"].astype(int)),
        f"{len(manifest)} manifest rows vs {len(db)} database rows")
    add("no_missing_roles", manifest["role"].notna().all(),
        f"{int(manifest['role'].isna().sum())} records without a role")
    roles_per_patient = manifest.groupby("patient_id")["role"].nunique()
    add("each_patient_has_one_role", bool((roles_per_patient == 1).all()),
        f"{int((roles_per_patient > 1).sum())} patients with more than one role")
    patient_sets = {r: set(g["patient_id"]) for r, g in manifest.groupby("role")}
    overlaps = [(a, b) for a in patient_sets for b in patient_sets
                if a < b and patient_sets[a] & patient_sets[b]]
    add("no_patient_overlap_between_roles", not overlaps, f"overlapping role pairs: {overlaps}")
    in_test_fold = manifest["strat_fold"] == test_fold
    add("fold10_records_all_in_test_role", bool((manifest.loc[in_test_fold, "role"] == test_role).all()),
        "every fold-10 record must have the test role")
    add("test_role_contains_only_fold10", bool((manifest.loc[manifest["role"] == test_role, "strat_fold"] == test_fold).all()),
        "no record outside fold 10 may have the test role")
    add("fold10_never_stratified", bool((manifest.loc[in_test_fold, "stratum"] == "fold10_not_stratified").all()),
        "fold-10 patients were assigned without reading their labels")
    add("fold9_only_in_9a_9b", bool(manifest.loc[manifest["strat_fold"] == fold9, "role"].isin(f9_roles).all()),
        f"fold {fold9} records may only have roles {sorted(f9_roles)}")
    add("folds_1_8_only_in_A_B_C", bool(manifest.loc[manifest["strat_fold"].isin(dev_folds), "role"].isin(dev_roles).all()),
        f"development folds may only have roles {sorted(dev_roles)}")
    return results


def role_summary(manifest: pd.DataFrame, db: pd.DataFrame, codes: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    """Per role: patients, records, folds and BBB positives. Label counts are NOT
    computed for the test role (fold 10 stays untouched)."""
    sp, target_codes = config["split"], list(config["targets"]["codes"])
    keep = (db["strat_fold"] != int(sp["test_fold"])).to_numpy()
    labels = io.label_matrix(codes[keep], target_codes, float(sp["stratify_min_likelihood"]))
    labels.index = db.loc[keep, "ecg_id"].astype(int).to_numpy()
    rows = []
    for role, group in manifest.groupby("role", sort=False):
        row: dict[str, Any] = {
            "role": role,
            "role_group": group["role_group"].iloc[0],
            "n_patients": int(group["patient_id"].nunique()),
            "n_records": len(group),
            "folds": ";".join(str(f) for f in sorted(group["strat_fold"].unique())),
        }
        for code in target_codes:
            if role == sp["test_role"]:  # fold 10: labels deliberately not summarised
                row[f"{code}_records"] = row[f"{code}_patients"] = pd.NA
                continue
            pos = labels.loc[group["ecg_id"], code].to_numpy()
            row[f"{code}_records"] = int(pos.sum())
            row[f"{code}_patients"] = int(group.loc[pos, "patient_id"].nunique())
        rows.append(row)
    order = list(sp["development_roles"]) + list(sp["fold9_roles"]) + [sp["test_role"]]
    return pd.DataFrame(rows).set_index("role").loc[order].reset_index()


def audit_status(root: Path, config: dict[str, Any], db_sha256: str) -> str:
    """'PASS' only if the audit ran on this exact database file and passed."""
    path = root / config["audit"]["report_dir"] / "audit_summary.json"
    if not path.is_file():
        return "NOT_RUN"
    summary = json.loads(path.read_text(encoding="utf-8"))
    audited = {i["file"]: i["sha256"] for i in summary.get("inputs", [])}
    if audited.get("ptbxl_database.csv") != db_sha256:
        return "STALE (audit ran on a different ptbxl_database.csv)"
    return summary.get("audit_status", "UNKNOWN")


def run(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], pd.DataFrame | None]:
    ds, sp = config["dataset"], config["split"]
    db_path = root / ds["database_csv"]
    db = io.load_database(db_path)
    out_dir = root / sp["manifest_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = sp["manifest_name"]

    problems = validate_split_inputs(db)
    if problems:
        meta = {"status": "REFUSED", "problems": problems}
        (out_dir / f"{name}_FAILED.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta, None

    codes, failures = io.parse_all_scp_codes(db)
    if len(failures):
        meta = {"status": "REFUSED", "problems": [f"{len(failures)} unparseable scp_codes cells"]}
        (out_dir / f"{name}_FAILED.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta, None

    manifest = make_manifest(db, codes, config)
    checks = check_manifest(manifest, db, config)
    checks_ok = all(c["status"] == "PASS" for c in checks)
    db_sha = io.sha256_file(db_path)
    audit = audit_status(root, config, db_sha)
    if checks_ok and audit == "PASS":
        status = "DRAFT_CHECKS_PASSED - not final until the BBB target definition is approved"
    elif checks_ok:
        status = f"DRAFT_BLOCKED - split checks passed but audit status is {audit}"
    else:
        status = "FAILED_SPLIT_CHECKS"

    src_dir = Path(__file__).resolve().parent
    meta: dict[str, Any] = {
        "manifest": f"{name}.csv",
        "status": status,
        "audit_status": audit,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(sp["seed"]),
        "development_roles": dict(sp["development_roles"]),
        "fold9_roles": dict(sp["fold9_roles"]),
        "test_role": sp["test_role"],
        "stratification": {
            "folds_1_8": ["strat_fold", "bbb_pattern"],
            "fold_9": ["bbb_pattern"],
            "fold_10": "none (assigned by fold only; labels not read)",
            "bbb_pattern_codes": list(config["targets"]["codes"]),
            "bbb_pattern_min_likelihood": float(sp["stratify_min_likelihood"]),
        },
        "inputs": {"ptbxl_database.csv": db_sha},
        "code_sha256": {p.name: io.sha256_file(p) for p in sorted(src_dir.glob("*.py"))},
        "environment": environment_info(),
        "checks": checks,
        "assumptions": [
            "patient_id identifies the same person across records (PhysioNet-provided pseudonymous ID).",
            "Official strat_fold assignment is kept: roles are only chosen within a patient's own fold group.",
            "Stratification uses a patient-level BBB pattern (any record listing the code); "
            "it balances rare codes but does not define the diagnostic target.",
            "Fold 10 is assigned wholesale to the test role; its labels are not used for "
            "stratification or summaries (scp_codes are only parsed to detect malformed cells).",
        ],
        "model_training_started": False,
    }
    if not checks_ok:
        (out_dir / f"{name}_FAILED.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta, None

    summary = role_summary(manifest, db, codes, config)
    manifest.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    summary.to_csv(out_dir / f"{name}_role_summary.csv", index=False, encoding="utf-8")
    meta["manifest_sha256"] = io.sha256_file(out_dir / f"{name}.csv")
    meta["role_counts"] = summary[["role", "n_patients", "n_records"]].to_dict(orient="records")
    (out_dir / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the patient-disjoint PTB-XL split manifest.")
    parser.add_argument("--config", default="configs/data_audit.yaml")
    parser.add_argument("--root", default=".", help="Repository root (default: current folder)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        config = io.load_config(root / args.config)
        meta, summary = run(config, root)
    except io.MissingInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Split manifest status: {meta['status']}")
    for problem in meta.get("problems", []):
        print(f"  [FAIL] {problem}")
    for chk in meta.get("checks", []):
        print(f"  [{chk['status']}] {chk['check']}")
    if summary is not None:
        print(summary[["role", "n_patients", "n_records", "folds"]].to_string(index=False))
        print(f"Written to: {root / config['split']['manifest_dir']}")
    if meta["status"].startswith("DRAFT_CHECKS_PASSED"):
        return 0
    if meta["status"].startswith("DRAFT_BLOCKED"):
        print("Next action: run the audit first (python -m src.data.audit_ptbxl) and fix any FAIL.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
