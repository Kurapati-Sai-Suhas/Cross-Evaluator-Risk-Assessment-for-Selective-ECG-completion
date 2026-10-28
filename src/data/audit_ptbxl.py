"""Milestone 1: audit PTB-XL metadata for the four bundle-branch-block (BBB) codes.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.data.audit_ptbxl --config configs/data_audit.yaml

What it does (metadata only; no waveform is loaded and no model is trained):
  1. Verifies the input CSVs exist and match PhysioNet's official SHA-256 checksums.
  2. Runs data-quality checks (duplicate IDs, invalid patient IDs, fold integrity, ...).
  3. Reports, for CLBBB / ILBBB / CRBBB / IRBBB kept SEPARATE:
     definitions, likelihood values, record and unique-patient counts,
     fold distribution, record- and patient-level co-occurrence.
  4. Writes CSV/JSON/Markdown reports to artifacts/reports/.

Exit codes: 0 = all checks passed, 1 = at least one FAIL check, 2 = missing input file.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import ptbxl_io as io

EXPECTED_FOLDS = list(range(1, 11))
DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _check(name: str, status: str, value: Any, detail: str) -> dict[str, Any]:
    return {"check": name, "status": status, "value": value, "detail": detail}


def _is_integer_valued(series: pd.Series) -> pd.Series:
    """True where a numeric value is finite and has no fractional part."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & (numeric == np.floor(numeric))


def _pattern(row: pd.Series) -> str:
    present = [code for code, flag in row.items() if flag]
    return "+".join(present) if present else "none"


# --------------------------------------------------------------------------- #
# 1. Target-code definitions
# --------------------------------------------------------------------------- #
def check_target_codes(statements: pd.DataFrame, target_codes: list[str]) -> pd.DataFrame:
    """One row per target code with its official metadata (or present=False)."""
    optional = {
        "statement_category": "Statement Category",
        "scp_ecg_description": "SCP-ECG Statement Description",
        "form": "form",
        "rhythm": "rhythm",
    }
    rows = []
    for code in target_codes:
        row: dict[str, Any] = {"code": code, "present_in_scp_statements": code in statements.index}
        source = statements.loc[code] if row["present_in_scp_statements"] else None
        for col in ["description", "diagnostic", "diagnostic_class", "diagnostic_subclass"]:
            row[col] = None if source is None else source[col]
        for key, col in optional.items():
            row[key] = None if source is None or col not in statements.columns else source[col]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Data quality
# --------------------------------------------------------------------------- #
def data_quality_checks(
    db: pd.DataFrame,
    statements: pd.DataFrame,
    codes: pd.Series,
    parse_failures: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return a list of PASS/WARN/FAIL/INFO checks. FAIL means 'stop and fix first'."""
    checks: list[dict[str, Any]] = []
    n = len(db)
    checks.append(_check("row_count", INFO, n, "Number of records (rows) in ptbxl_database.csv"))

    # ecg_id
    ecg_ok = _is_integer_valued(db["ecg_id"])
    checks.append(_check(
        "ecg_id_valid_integer", PASS if ecg_ok.all() else FAIL, int((~ecg_ok).sum()),
        "Records whose ecg_id is missing or not a whole number",
    ))
    dup_ecg = int(db["ecg_id"].duplicated(keep=False).sum())
    checks.append(_check(
        "ecg_id_unique", PASS if dup_ecg == 0 else FAIL, dup_ecg,
        "Rows sharing an ecg_id with another row",
    ))
    dup_rows = int(db.duplicated(keep=False).sum())
    checks.append(_check(
        "fully_duplicated_rows", PASS if dup_rows == 0 else WARN, dup_rows,
        "Rows identical to another row in every column",
    ))

    # patient_id
    pid = pd.to_numeric(db["patient_id"], errors="coerce")
    pid_missing = int(db["patient_id"].isna().sum())
    pid_non_int = int((db["patient_id"].notna() & ~_is_integer_valued(db["patient_id"])).sum())
    pid_non_pos = int((pid <= 0).sum())
    bad_pid = pid_missing + pid_non_int + pid_non_pos
    checks.append(_check(
        "patient_id_valid", PASS if bad_pid == 0 else FAIL,
        {"missing": pid_missing, "not_whole_number": pid_non_int, "zero_or_negative": pid_non_pos},
        "patient_id must be present, a whole number, and > 0 (stored as e.g. 15709.0 in the CSV)",
    ))
    checks.append(_check(
        "unique_patients", INFO, int(pid.nunique()), "Distinct patient_id values",
    ))
    per_patient = db.groupby("patient_id").size()
    checks.append(_check(
        "records_per_patient", INFO,
        {"min": int(per_patient.min()), "median": float(per_patient.median()),
         "max": int(per_patient.max()), "patients_with_2plus_records": int((per_patient >= 2).sum())},
        "Distribution of records per patient (why patient-level splitting is needed)",
    ))

    # strat_fold
    fold = pd.to_numeric(db["strat_fold"], errors="coerce")
    bad_fold = int((~fold.isin(EXPECTED_FOLDS)).sum())
    checks.append(_check(
        "strat_fold_in_1_to_10", PASS if bad_fold == 0 else FAIL, bad_fold,
        "Records whose strat_fold is missing or not one of 1..10",
    ))
    folds_present = sorted(int(f) for f in fold.dropna().unique())
    checks.append(_check(
        "all_10_folds_present", PASS if folds_present == EXPECTED_FOLDS else FAIL, folds_present,
        "Official folds found in the file",
    ))
    folds_per_patient = db.groupby("patient_id")["strat_fold"].nunique()
    multi_fold = int((folds_per_patient > 1).sum())
    checks.append(_check(
        "each_patient_in_one_fold", PASS if multi_fold == 0 else FAIL, multi_fold,
        "Patients whose records are spread over more than one official fold",
    ))

    # scp_codes
    n_fail = len(parse_failures)
    checks.append(_check(
        "scp_codes_parseable", PASS if n_fail == 0 else FAIL, n_fail,
        "scp_codes cells that are not a valid {code: likelihood} dictionary",
    ))
    empty = int(sum(1 for d in codes if len(d) == 0)) - n_fail
    checks.append(_check(
        "scp_codes_non_empty", PASS if empty == 0 else WARN, empty,
        "Parsed records with an empty scp_codes dictionary",
    ))
    all_codes = {c for d in codes for c in d}
    unknown = sorted(all_codes - set(statements.index))
    checks.append(_check(
        "scp_codes_known_in_statements", PASS if not unknown else FAIL, unknown,
        "Codes used in ptbxl_database.csv but missing from scp_statements.csv",
    ))
    unused = sorted(set(statements.index) - all_codes)
    checks.append(_check(
        "statements_never_used", INFO, unused, "Codes in scp_statements.csv never used by any record",
    ))
    bad_lik = int(sum(1 for d in codes for v in d.values() if v < 0 or v > 100))
    checks.append(_check(
        "likelihood_in_0_to_100", PASS if bad_lik == 0 else FAIL, bad_lik,
        "Code likelihoods outside 0..100 (PhysioNet: 0 means 'unknown')",
    ))

    # filenames
    for col in ["filename_lr", "filename_hr"]:
        missing = int(db[col].isna().sum())
        dup = int(db[col].dropna().duplicated(keep=False).sum())
        checks.append(_check(
            f"{col}_present_and_unique", PASS if missing == 0 and dup == 0 else FAIL,
            {"missing": missing, "duplicated": dup}, f"Waveform path column {col}",
        ))
    stem_id = pd.to_numeric(
        db["filename_hr"].astype(str).str.extract(r"(\d+)_hr$")[0], errors="coerce"
    )
    mismatch = int((stem_id != pd.to_numeric(db["ecg_id"], errors="coerce")).sum())
    checks.append(_check(
        "filename_matches_ecg_id", PASS if mismatch == 0 else WARN, mismatch,
        "Records whose filename_hr number differs from ecg_id",
    ))

    # demographics (informational; PTB-XL encodes ages > 89 as ~300 per HIPAA)
    if "age" in db.columns:
        age = pd.to_numeric(db["age"], errors="coerce")
        checks.append(_check(
            "age_values", WARN if (age < 0).any() else INFO,
            {"missing": int(age.isna().sum()), "negative": int((age < 0).sum()),
             "above_89": int((age > 89).sum()),
             "above_89_distinct_values": sorted(float(a) for a in age[age > 89].unique())},
            "PhysioNet: ages > 89 'appear in the range of 300 years' (HIPAA). Not an error.",
        ))
    if "sex" in db.columns:
        sex_values = sorted(str(v) for v in db["sex"].dropna().unique())
        checks.append(_check(
            "sex_values", PASS if set(sex_values) <= {"0", "1"} else WARN,
            {"values": sex_values, "missing": int(db["sex"].isna().sum())},
            "Expected 0/1 coding",
        ))
    if "validated_by_human" in db.columns:
        vbh = db["validated_by_human"].astype(str).str.lower() == "true"
        by_fold = vbh.groupby(fold).mean().round(4)
        not_validated_9_10 = int((~vbh[fold.isin([9, 10])]).sum())
        checks.append(_check(
            "folds_9_10_human_validated", PASS if not_validated_9_10 == 0 else WARN,
            {"records_not_validated_in_folds_9_10": not_validated_9_10,
             "share_validated_by_fold": {str(int(k)): float(v) for k, v in by_fold.items()}},
            "PhysioNet: records in folds 9 and 10 'underwent at least one human evaluation'",
        ))
    return checks


def missing_value_table(db: pd.DataFrame) -> pd.DataFrame:
    """Missing (empty) cells per column, as count and percent."""
    missing = db.isna().sum()
    return pd.DataFrame({
        "column": missing.index,
        "n_missing": missing.values.astype(int),
        "pct_missing": (100 * missing.values / max(len(db), 1)).round(2),
    })


# --------------------------------------------------------------------------- #
# 3. BBB label statistics (codes always kept separate)
# --------------------------------------------------------------------------- #
def likelihood_values(codes: pd.Series, target_codes: list[str]) -> pd.DataFrame:
    """How often each likelihood value occurs for each target code."""
    rows = []
    for code in target_codes:
        values = pd.Series([d[code] for d in codes if code in d], dtype=float)
        for value, count in values.value_counts().sort_index().items():
            rows.append({"code": code, "likelihood": float(value), "n_records": int(count)})
    return pd.DataFrame(rows, columns=["code", "likelihood", "n_records"])


def label_counts(
    db: pd.DataFrame,
    codes: pd.Series,
    target_codes: list[str],
    thresholds: list[float],
    decision_folds: list[int],
) -> pd.DataFrame:
    """Record-level and unique-patient counts per code and likelihood threshold."""
    in_decision = db["strat_fold"].isin(decision_folds)
    rows = []
    for threshold in thresholds:
        labels = io.label_matrix(codes, target_codes, threshold)
        for code in target_codes:
            pos = labels[code]
            rows.append({
                "code": code,
                "min_likelihood": threshold,
                "n_records": int(pos.sum()),
                "n_unique_patients": int(db.loc[pos, "patient_id"].nunique()),
                "pct_of_all_records": round(100 * pos.mean(), 3),
                "n_records_decision_folds": int((pos & in_decision).sum()),
                "n_patients_decision_folds": int(db.loc[pos & in_decision, "patient_id"].nunique()),
            })
    return pd.DataFrame(rows)


def fold_distribution(db: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Long table: per fold, total records/patients and positives per code."""
    rows = []
    for fold_id in EXPECTED_FOLDS:
        in_fold = db["strat_fold"] == fold_id
        base = {
            "strat_fold": fold_id,
            "fold_records": int(in_fold.sum()),
            "fold_patients": int(db.loc[in_fold, "patient_id"].nunique()),
        }
        for code in labels.columns:
            pos = labels[code] & in_fold
            rows.append({**base, "code": code, "n_records": int(pos.sum()),
                         "n_unique_patients": int(db.loc[pos, "patient_id"].nunique())})
    return pd.DataFrame(rows)


def cooccurrence(labels: pd.DataFrame) -> pd.DataFrame:
    """Square matrix: cell (a, b) = number of rows where both a and b are present.
    The diagonal is the total count for each code."""
    as_int = labels.astype(int)
    matrix = as_int.T.dot(as_int)
    matrix.index.name = "code"
    return matrix


def patient_labels(db: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """A patient is positive for a code if ANY of their records is positive."""
    return labels.groupby(db["patient_id"]).any()


def label_patterns(labels: pd.DataFrame) -> pd.DataFrame:
    """Count of every exact combination of BBB codes (e.g. 'CRBBB', 'IRBBB+ILBBB', 'none')."""
    patterns = labels.apply(_pattern, axis=1)
    counts = patterns.value_counts()
    return pd.DataFrame({"pattern": counts.index, "n": counts.values.astype(int)})


def within_patient_changes(db: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """For patients with 2+ records: how many have a code in some records but not all."""
    multi = db.groupby("patient_id")["ecg_id"].transform("size") >= 2
    sub = labels[multi]
    grouped = sub.groupby(db.loc[multi, "patient_id"])
    any_pos, all_pos = grouped.any(), grouped.all()
    rows = []
    for code in labels.columns:
        rows.append({
            "code": code,
            "patients_with_2plus_records_and_code_in_any": int(any_pos[code].sum()),
            "code_in_all_their_records": int(all_pos[code].sum()),
            "code_in_some_but_not_all_records": int((any_pos[code] & ~all_pos[code]).sum()),
        })
    return pd.DataFrame(rows)


def superclass_cooccurrence(
    codes: pd.Series, labels: pd.DataFrame, statements: pd.DataFrame
) -> pd.DataFrame:
    """Among records positive for each BBB code, how many ALSO carry another code
    from each diagnostic superclass (NORM, MI, STTC, HYP, CD)."""
    code_class = statements["diagnostic_class"].dropna().to_dict()
    rows = []
    for code in labels.columns:
        positive = codes[labels[code]]
        row: dict[str, Any] = {"code": code, "n_positive_records": len(positive)}
        for superclass in DIAGNOSTIC_SUPERCLASSES:
            row[f"also_{superclass}"] = int(sum(
                any(code_class.get(other) == superclass for other in d if other != code)
                for d in positive
            ))
        rows.append(row)
    return pd.DataFrame(rows)


def top_cooccurring_codes(
    codes: pd.Series, labels: pd.DataFrame, statements: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    """For each BBB code, the most frequent OTHER codes listed on the same records."""
    rows = []
    for code in labels.columns:
        positive = codes[labels[code]]
        counts = pd.Series([o for d in positive for o in d if o != code], dtype=object).value_counts()
        for other, count in counts.head(top_k).items():
            meta = statements.loc[other] if other in statements.index else None
            rows.append({
                "code": code,
                "other_code": other,
                "n_records": int(count),
                "pct_of_code_records": round(100 * count / max(len(positive), 1), 2),
                "other_description": None if meta is None else meta["description"],
                "other_diagnostic_class": None if meta is None else meta["diagnostic_class"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Orchestration + report writing
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal Markdown table writer (avoids an extra 'tabulate' dependency)."""
    frame = df.reset_index() if index else df
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def run_audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run every audit step and write reports. Returns the summary dictionary."""
    ds, tg, au = config["dataset"], config["targets"], config["audit"]
    db_path, st_path = root / ds["database_csv"], root / ds["statements_csv"]
    checksums_path = root / ds["checksums_file"]
    report_dir = root / au["report_dir"]

    target_codes: list[str] = list(tg["codes"])
    thresholds: list[float] = [float(t) for t in tg["likelihood_thresholds"]]
    decision_folds: list[int] = [int(f) for f in au["decision_folds"]]

    db = io.load_database(db_path)
    statements = io.load_statements(st_path)
    codes, parse_failures = io.parse_all_scp_codes(db)
    report_dir.mkdir(parents=True, exist_ok=True)

    checksum_results = [io.verify_checksum(p, checksums_path) for p in (db_path, st_path)]
    checks = []
    for res in checksum_results:
        status = {"MATCH": PASS, "MISMATCH": FAIL}.get(res["status"], WARN)
        checks.append(_check(f"official_checksum_{res['file']}", status, res["status"],
                             "SHA-256 compared with PhysioNet SHA256SUMS.txt"))
    checks += data_quality_checks(db, statements, codes, parse_failures)

    definitions = check_target_codes(statements, target_codes)
    missing_codes = definitions.loc[~definitions["present_in_scp_statements"], "code"].tolist()
    checks.append(_check(
        "target_codes_in_scp_statements", PASS if not missing_codes else FAIL, missing_codes,
        "Target codes missing from scp_statements.csv",
    ))

    labels_any = io.label_matrix(codes, target_codes, min(thresholds))
    counts = label_counts(db, codes, target_codes, thresholds, decision_folds)
    lik = likelihood_values(codes, target_codes)
    folds = fold_distribution(db, labels_any)
    rec_co = cooccurrence(labels_any)
    pat_lab = patient_labels(db, labels_any)
    pat_co = cooccurrence(pat_lab)
    patterns_rec = label_patterns(labels_any)
    patterns_pat = label_patterns(pat_lab)
    changes = within_patient_changes(db, labels_any)
    supers = superclass_cooccurrence(codes, labels_any, statements)
    top_other = top_cooccurring_codes(codes, labels_any, statements, int(au["top_k_cooccurring_codes"]))
    missing_tbl = missing_value_table(db)
    checks_df = pd.DataFrame(checks)

    outputs = {
        "data_quality_checks.csv": checks_df,
        "missing_values.csv": missing_tbl,
        "scp_codes_parse_failures.csv": parse_failures,
        "bbb_code_definitions.csv": definitions,
        "bbb_label_counts.csv": counts,
        "bbb_likelihood_values.csv": lik,
        "bbb_fold_distribution.csv": folds,
        "bbb_cooccurrence_records.csv": rec_co.reset_index(),
        "bbb_cooccurrence_patients.csv": pat_co.reset_index(),
        "bbb_label_patterns_records.csv": patterns_rec,
        "bbb_label_patterns_patients.csv": patterns_pat,
        "bbb_within_patient_changes.csv": changes,
        "bbb_superclass_cooccurrence.csv": supers,
        "bbb_top_cooccurring_codes.csv": top_other,
    }
    for name, frame in outputs.items():
        frame.to_csv(report_dir / name, index=False, encoding="utf-8")

    n_fail = int((checks_df["status"] == FAIL).sum())
    summary = {
        "audit_status": "PASS" if n_fail == 0 else "FAIL",
        "n_fail_checks": n_fail,
        "n_warn_checks": int((checks_df["status"] == WARN).sum()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": ds["name"], "version": ds["version"]},
        "inputs": checksum_results,
        "environment": environment_info(),
        "n_records": int(len(db)),
        "n_unique_patients": int(db["patient_id"].nunique()),
        "target_codes": target_codes,
        "likelihood_thresholds": thresholds,
        "decision_folds": decision_folds,
        "checks": checks,
        "label_counts": counts.to_dict(orient="records"),
        "model_training_started": False,
    }
    (report_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _write_markdown_report(report_dir / "audit_report.md", summary, outputs, folds)
    return summary


def _write_markdown_report(
    path: Path, summary: dict[str, Any], outputs: dict[str, pd.DataFrame], folds: pd.DataFrame
) -> None:
    rec_wide = folds.pivot(index="strat_fold", columns="code", values="n_records")
    pat_wide = folds.pivot(index="strat_fold", columns="code", values="n_unique_patients")
    totals = folds.drop_duplicates("strat_fold").set_index("strat_fold")[["fold_records", "fold_patients"]]
    lines = [
        "# PTB-XL metadata audit (Milestone 1)",
        "",
        f"Generated by `src/data/audit_ptbxl.py` at {summary['created_utc']} (UTC). "
        "Every number below is computed from the input files; nothing is typed in by hand.",
        "",
        f"**Audit status: {summary['audit_status']}** "
        f"({summary['n_fail_checks']} FAIL, {summary['n_warn_checks']} WARN checks). "
        "No model was trained.",
        "",
        "## Inputs",
        _md_table(pd.DataFrame(summary["inputs"])),
        "",
        f"Records: **{summary['n_records']}**. Unique patients: **{summary['n_unique_patients']}**.",
        "",
        "## Data-quality checks",
        _md_table(pd.DataFrame(summary["checks"]).astype(str)),
        "",
        "## Columns with missing values",
        _md_table(outputs["missing_values.csv"].query("n_missing > 0")),
        "",
        "## BBB code definitions (from scp_statements.csv)",
        _md_table(outputs["bbb_code_definitions.csv"]),
        "",
        "## Likelihood values per code",
        "PhysioNet: likelihood 0 means *unknown*, not absent.",
        "",
        _md_table(outputs["bbb_likelihood_values.csv"]),
        "",
        "## Record and unique-patient counts",
        f"`*_decision_folds` columns use folds {summary['decision_folds']} only. "
        "Fold 10 is reported for completeness but must not drive design decisions.",
        "",
        _md_table(outputs["bbb_label_counts.csv"]),
        "",
        "## Fold distribution - records (code listed at any likelihood)",
        _md_table(rec_wide.join(totals), index=True),
        "",
        "## Fold distribution - unique patients (code listed at any likelihood)",
        _md_table(pat_wide.join(totals), index=True),
        "",
        "## Co-occurrence within records (diagonal = total)",
        _md_table(outputs["bbb_cooccurrence_records.csv"]),
        "",
        "## Co-occurrence across a patient's records (diagonal = total patients)",
        _md_table(outputs["bbb_cooccurrence_patients.csv"]),
        "",
        "## Exact BBB label combinations - records",
        _md_table(outputs["bbb_label_patterns_records.csv"]),
        "",
        "## Exact BBB label combinations - patients",
        _md_table(outputs["bbb_label_patterns_patients.csv"]),
        "",
        "## Label consistency across a patient's records",
        _md_table(outputs["bbb_within_patient_changes.csv"]),
        "",
        "## Diagnostic superclasses on the same records",
        _md_table(outputs["bbb_superclass_cooccurrence.csv"]),
        "",
        "## Most frequent other codes on the same records",
        _md_table(outputs["bbb_top_cooccurring_codes.csv"]),
        "",
        "## Not decided by this script",
        "The diagnostic target definition is a human decision recorded in "
        "`docs/decision_log.md` after reviewing these tables.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_console_summary(summary: dict[str, Any], report_dir: Path) -> None:
    print(f"PTB-XL audit status: {summary['audit_status']} "
          f"({summary['n_fail_checks']} FAIL, {summary['n_warn_checks']} WARN)")
    print(f"Records: {summary['n_records']}  Unique patients: {summary['n_unique_patients']}")
    for res in summary["inputs"]:
        print(f"  {res['file']}: checksum {res['status']}")
    print("BBB counts (records / unique patients) by minimum likelihood:")
    for row in summary["label_counts"]:
        print(f"  {row['code']:<6} >= {row['min_likelihood']:>5.0f}: "
              f"{row['n_records']:>5} records / {row['n_unique_patients']:>5} patients")
    for chk in summary["checks"]:
        if chk["status"] in (FAIL, WARN):
            print(f"  [{chk['status']}] {chk['check']}: {chk['value']}")
    print(f"Reports written to: {report_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit PTB-XL metadata (no model training).")
    parser.add_argument("--config", default="configs/data_audit.yaml")
    parser.add_argument("--root", default=".", help="Repository root (default: current folder)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    root = Path(args.root).resolve()
    try:
        config = io.load_config(root / args.config)
        summary = run_audit(config, root)
    except io.MissingInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _print_console_summary(summary, root / config["audit"]["report_dir"])
    return 0 if summary["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
