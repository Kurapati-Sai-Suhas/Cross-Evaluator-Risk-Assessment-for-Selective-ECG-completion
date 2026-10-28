"""Pre-pin check that the M4b-0 G1/G2 feature files may be reused under the current gate pre-registration.

Proof by recomputation, not by assertion: for B_gate, 9a_calibration and 9b_threshold the features are recomputed in memory
with the CURRENT code (``src.gate.extract_features.extract_role``) and the CURRENT config (``configs/gate_m4b_v0_2.yaml``),
serialised exactly as the extractor does, and their SHA-256 is compared with the stored files. Nothing is written under
``artifacts/gate``; offline labels are never opened; fold 10 is refused by ``extract_role``; nothing is fitted.

Also checked: archived config hash (3705aea7...) and the extraction record; parsed and textual identity of the
feature-defining blocks; the formula check of ``features.check_config``; the D1 key files against the D1 hash manifest; key
row order = feature row order; roles, folds and patients of every key row against the frozen split manifest; role
disjointness; the feature hash manifest.

    .\\.venv\\Scripts\\python.exe -m src.gate.verify_feature_reuse
"""
from __future__ import annotations

import hashlib
import io as _io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import ptbxl_io as io
from src.data.ptbxl_waveforms import resolve_waveform_root
from src.gate import extract_features as ef
from src.gate import features as gf

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "artifacts/protocol_history/gate_m4b_v0_2_3705aea7.yaml"
CURRENT = REPO / "configs/gate_m4b_v0_2.yaml"
ARCHIVE_SHA = "3705aea7bebd22e2b36dc389dcb16437cf259409407bbbe858ab71dac7f7b171"
MANIFEST_SHA = "898d0ac707cf86ec9201eb4d3a9d97a49e8b84767374313b01d1a5996f42b83b"
FEATURE_BLOCKS = ("output_dir", "d1_keys_dir", "d1_hash_manifest_sha256", "roles", "g1", "g2")
OUT = REPO / "artifacts/reports/m4b0d_feature_reuse_verification.json"


def block_text(path: Path, start: str, stop: str) -> str:
    """Raw YAML text from the line starting with ``start`` up to (not including) the first later line starting with ``stop``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith(start))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith(stop))
    return "\n".join(lines[i:j])


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    checks = []

    def check(name: str, cond: bool, detail) -> None:
        checks.append({"check": name, "status": "PASS" if cond else "FAIL", "detail": detail})
        print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)

    old, new = io.load_config(ARCHIVE), io.load_config(CURRENT)
    summary = json.loads((REPO / "artifacts/gate/v0_2/extraction_summary.json").read_text(encoding="utf-8"))
    check("archived_config_sha256", io.sha256_file(ARCHIVE) == ARCHIVE_SHA, io.sha256_file(ARCHIVE))
    check("extraction_record_config_sha256", summary["config_sha256"] == ARCHIVE_SHA, summary["config_sha256"])
    check("current_config_sha256_recorded", True, io.sha256_file(CURRENT))
    for b in FEATURE_BLOCKS:
        check(f"parsed_block_identical:{b}", old[b] == new[b], b)
    for start, stop in (("g1:", "# ----"), ("g2:", "# ----"), ("roles:", "# ----")):
        ta, tb = block_text(ARCHIVE, start, stop), block_text(CURRENT, start, stop)
        check(f"text_block_identical:{start}", ta == tb, hashlib.sha256(ta.encode()).hexdigest())
    for line in ("output_dir:", "d1_keys_dir:", "d1_hash_manifest_sha256:"):
        get = lambda p: next(s for s in p.read_text(encoding="utf-8").splitlines() if s.startswith(line))  # noqa: E731
        check(f"text_line_identical:{line}", get(ARCHIVE) == get(CURRENT), get(CURRENT))
    gf.check_config(new)
    check("formulas_match_current_config(features.check_config)", True, "no exception")
    check("feature_columns_unchanged", summary["feature_columns"] == gf.G1_COLUMNS + gf.G2_COLUMNS, len(summary["feature_columns"]))

    d1_manifest_path = REPO / "artifacts/d1/v0_1/hash_manifest.json"
    check("d1_hash_manifest_sha256", io.sha256_file(d1_manifest_path) == new["d1_hash_manifest_sha256"], io.sha256_file(d1_manifest_path))
    d1_manifest = json.loads(d1_manifest_path.read_text(encoding="utf-8"))
    feat_manifest = json.loads((REPO / "artifacts/gate/v0_2/hash_manifest.json").read_text(encoding="utf-8"))
    for rel, sha in feat_manifest["files"].items():
        check(f"feature_manifest_entry:{rel}", io.sha256_file(REPO / rel) == sha, sha)

    split_path = REPO / "artifacts/manifests/split_manifest_v1_frozen.csv"
    check("split_manifest_sha256", io.sha256_file(split_path) == MANIFEST_SHA, io.sha256_file(split_path))
    split = pd.read_csv(split_path).set_index("ecg_id")
    audit_cfg = io.load_config(REPO / "configs/data_audit.yaml")
    db_files = pd.read_csv(REPO / audit_cfg["dataset"]["database_csv"], usecols=["ecg_id", "filename_hr"]).set_index("ecg_id")["filename_hr"]
    root = resolve_waveform_root(audit_cfg, REPO)
    p2 = io.load_config(REPO / "configs/protocol_v0_2.yaml")
    seed, k = int(p2["seed"]), int(p2["d1"]["masks_per_record"])
    patients, recomputed = {}, {}
    for role in new["roles"]["feature_extraction"]:
        keys = pd.read_csv(REPO / new["d1_keys_dir"] / f"{role}_rows.csv")
        stored_path = REPO / new["output_dir"] / "features" / f"{role}_features.csv"
        stored = pd.read_csv(stored_path, usecols=["row_id"])
        m = split.loc[keys["ecg_id"]]
        check(f"{role}:row_id_unique", keys["row_id"].is_unique, len(keys))
        check(f"{role}:key_rows_equal_feature_rows_in_order", keys["row_id"].tolist() == stored["row_id"].tolist(), len(stored))
        check(f"{role}:roles_folds_patients_match_split_manifest",
              (m["role"].to_numpy() == role).all() and (m["strat_fold"].to_numpy() == keys["strat_fold"].to_numpy()).all()
              and (m["patient_id"].to_numpy() == keys["patient_id"].to_numpy()).all() and (keys["strat_fold"] != 10).all(),
              sorted(keys["strat_fold"].unique().tolist()))
        check(f"{role}:8_masks_per_record", bool((keys.groupby("ecg_id").size() == k).all()), int(keys["ecg_id"].nunique()))
        patients[role] = set(keys["patient_id"])
        feats, info = ef.extract_role(REPO, new, role, d1_manifest, db_files, root, seed, k)   # masks re-checked inside
        buf = _io.StringIO()
        feats.to_csv(buf, index=False, float_format="%.17g", lineterminator="\n")
        recomputed_bytes = buf.getvalue().encode("utf-8")
        stored_bytes = stored_path.read_bytes().replace(b"\r\n", b"\n")
        sha_new, sha_stored = hashlib.sha256(recomputed_bytes).hexdigest(), io.sha256_file(stored_path)
        recomputed[role] = {"recomputed_sha256_lf": sha_new, "stored_sha256": sha_stored,
                            "stored_sha256_lf": hashlib.sha256(stored_bytes).hexdigest(), "rows": info["rows"],
                            "seconds": info["seconds"]}
        check(f"{role}:recomputed_features_byte_identical", recomputed_bytes == stored_bytes, recomputed[role])
    names = list(patients)
    overlap = {f"{a}&{b}": len(patients[a] & patients[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    check("roles_patient_disjoint", not any(overlap.values()), overlap)
    ok = all(c["status"] == "PASS" for c in checks)
    OUT.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "milestone": "M4b-0d pre-pin",
                               "verdict": "SAFE TO REUSE" if ok else "NOT SAFE", "command": " ".join(sys.argv),
                               "current_config_sha256": io.sha256_file(CURRENT), "archived_config_sha256": io.sha256_file(ARCHIVE),
                               "recomputed": recomputed, "checks": checks,
                               "not_done": "no write under artifacts/gate; offline labels not read; fold 10 not opened; nothing fitted"},
                              indent=2, default=str), encoding="utf-8")
    print("verdict:", "SAFE TO REUSE" if ok else "NOT SAFE", "->", OUT)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
