"""M4b-1 (authorised by the owner on 2026-10-24): fit the gates on B_gate, calibrate on 9a, select and lock thresholds on 9b.

Every rule comes from the pinned protocol v0.2 (configs/protocol_v0_2.yaml, locked in artifacts/locks/protocol_v0_2.lock.json)
and its gate pre-registration (configs/gate_m4b_v0_2.yaml, blocks methods / rf / calibration / threshold):

    .\\.venv\\Scripts\\python.exe -m src.gate.m4b1 fit          # stage 01: patient-grouped 5-fold CV + refit on B_gate
    .\\.venv\\Scripts\\python.exe -m src.gate.m4b1 calibrate    # stage 02: Platt on 9a_calibration
    .\\.venv\\Scripts\\python.exe -m src.gate.m4b1 threshold    # stage 03: thresholds on 9b_threshold + threshold lock

Each stage writes its own folder under artifacts/gate/v0_2/m4b1/ and refuses to run if that folder exists. Inputs are the
extracted features (G1/G2 only, exact column whitelist per method), the D1 keys (row_id, patient_id) and D1 itself, which
enters only through ``src.eval.d1.attach_d1_target`` (columns row_id and D1 of the offline label file, nothing else).
test_fold10 is refused everywhere in this module; the fold-10 pathway is a separate code path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data import ptbxl_io as io
from src.eval import primary_decision as pd_
from src.eval import selective as sel
from src.eval.d1 import attach_d1_target, check_feature_columns
from src.gate import features as gf
from src.utils.provenance import code_tree_sha256, environment

REPO = Path(__file__).resolve().parents[2]
OUT = "artifacts/gate/v0_2/m4b1"
STAGE_DIRS = {"fit": "01_fit", "calibrate": "02_calibration", "threshold": "03_threshold"}
STAGE_ROLE = {"fit": "B_gate", "calibrate": "9a_calibration", "threshold": "9b_threshold"}
FIT_METHODS = {"gate_A": list(gf.G1_COLUMNS), "gate_B": list(gf.G1_COLUMNS) + list(gf.G2_COLUMNS),
               "M3_signal_quality_only": list(gf.G2_COLUMNS)}
M2 = "M2_uncertainty_only"
CLIP = 1e-6
FEATURE_CONFIG_SHA = "3705aea7bebd22e2b36dc389dcb16437cf259409407bbbe858ab71dac7f7b171"   # features were extracted under this
FEATURE_BLOCKS = ("output_dir", "d1_keys_dir", "d1_hash_manifest_sha256", "roles", "g1", "g2")
THRESHOLD_LOCK = "artifacts/locks/threshold_lock_v0_2.json"
AUTHORISATION = ("owner, 2026-10-24: \"I authorise M4b-1 stages 01-05: Gate A/B patient-grouped CV on B_gate (M3 optional), "
                 "Platt on 9a only, threshold selection and lock on 9b only, fold-10/M5 build and dry-run on synthetic/9b data "
                 "only. Fold 10 stays sealed until I separately authorise the final run.\"")


# ------------------------------------------------------------------------------------------------------------------
# Pure helpers (synthetic tests)
# ------------------------------------------------------------------------------------------------------------------
def check_role(role: str, stage: str) -> None:
    if role == "test_fold10":
        raise RuntimeError("test_fold10 is refused in M4b-1; only the separately authorised fold-10 pathway may read it")
    if STAGE_ROLE.get(stage) != role:
        raise RuntimeError(f"stage {stage} runs on {STAGE_ROLE.get(stage)} only, not {role}")


def method_matrix(features: pd.DataFrame, method: str, protocol: dict[str, Any]) -> pd.DataFrame:
    """row_id + exactly the whitelisted columns of the method, in the pre-registered order; refuses anything else."""
    cols = FIT_METHODS[method]
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"{method}: missing feature columns {missing}")
    bad = check_feature_columns(cols, protocol)
    if bad:
        raise ValueError(f"{method}: forbidden columns {bad}")
    return features[["row_id", *cols]]


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Fold index per row from StratifiedGroupKFold (shuffle=True); every patient is in exactly one fold."""
    folds = np.full(len(y), -1, dtype=int)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        folds[test] = k
    if (folds < 0).any():
        raise RuntimeError("a row was assigned to no fold")
    per_group = pd.Series(folds).groupby(np.asarray(groups)).nunique()
    if (per_group > 1).any():
        raise RuntimeError("a patient appears in more than one fold")
    return folds


def grid(rf_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"min_samples_leaf": m, "max_features": f} for m in rf_cfg["grid"]["min_samples_leaf"] for f in rf_cfg["grid"]["max_features"]]


def select_params(results: list[dict[str, Any]], tol: float = 1e-9) -> dict[str, Any]:
    """Lowest log loss; within ``tol`` of the best: larger min_samples_leaf first, then max_features 'sqrt' before 0.5."""
    best = min(r["log_loss"] for r in results)
    near = [r for r in results if r["log_loss"] - best <= tol]
    return sorted(near, key=lambda r: (-r["min_samples_leaf"], 0 if r["max_features"] == "sqrt" else 1))[0]


def rf_model(rf_cfg: dict[str, Any], params: dict[str, Any]) -> RandomForestClassifier:
    fx = rf_cfg["fixed"]
    return RandomForestClassifier(n_estimators=fx["n_estimators"], criterion=fx["criterion"], max_depth=fx["max_depth"],
                                  bootstrap=fx["bootstrap"], class_weight=fx["class_weight"], random_state=fx["random_state"],
                                  n_jobs=fx["n_jobs"], min_samples_leaf=params["min_samples_leaf"],
                                  max_features=params["max_features"])


def score(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """predict_proba[:, 1] with n_jobs = 1 (gate pre-registration rf.scoring), float64."""
    model.n_jobs = 1
    return model.predict_proba(X)[:, 1].astype(np.float64)


def platt_input(raw: np.ndarray, method: str) -> np.ndarray:
    """z = logit(clip(score, 1e-6, 1 - 1e-6)) for RF scores; the raw score for M2 (gate pre-registration calibration.input)."""
    s = np.asarray(raw, dtype=np.float64)
    if method == M2:
        return s.reshape(-1, 1)
    c = np.clip(s, CLIP, 1 - CLIP)
    return np.log(c / (1 - c)).reshape(-1, 1)


def calibrated(cal: LogisticRegression | None, raw: np.ndarray, method: str) -> np.ndarray:
    """Platt-calibrated score (higher = riskier); the raw score when the rule chose no Platt model."""
    if cal is None:
        return np.asarray(raw, dtype=np.float64)
    return cal.predict_proba(platt_input(raw, method))[:, 1].astype(np.float64)


def write_readonly(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as f:     # "x": never overwrite
        f.write(text)
    os.chmod(path, 0o444)


def env_record() -> dict[str, Any]:
    e = environment()
    e["packages"]["scikit-learn"], e["packages"]["joblib"] = sklearn.__version__, joblib.__version__
    e["packages"]["scipy"], e["packages"]["numpy"] = scipy.__version__, np.__version__
    return e


# ------------------------------------------------------------------------------------------------------------------
# Inputs, with every hash verified before use
# ------------------------------------------------------------------------------------------------------------------
class Inputs:
    def __init__(self, repo: Path):
        self.repo = repo
        self.p2 = io.load_config(repo / "configs/protocol_v0_2.yaml")
        self.gate = io.load_config(repo / "configs/gate_m4b_v0_2.yaml")
        lock = json.loads((repo / "artifacts/locks/protocol_v0_2.lock.json").read_text(encoding="utf-8"))
        self.checks: list[dict[str, Any]] = []
        self.ok("protocol_v0_2_pinned", lock["status"] == "APPROVED" and lock["sha256"] == io.sha256_file(repo / "configs/protocol_v0_2.yaml"), lock["sha256"])
        self.ok("gate_preregistration_locked", lock["gate_preregistration"]["sha256"] == io.sha256_file(repo / "configs/gate_m4b_v0_2.yaml")
                == self.p2["gate_preregistration"]["sha256"], lock["gate_preregistration"]["sha256"])
        summ = json.loads((repo / "artifacts/gate/v0_2/extraction_summary.json").read_text(encoding="utf-8"))
        archive = repo / "artifacts/protocol_history/gate_m4b_v0_2_3705aea7.yaml"
        old = yaml.safe_load(archive.read_text(encoding="utf-8"))
        self.ok("features_config_and_archive", summ["config_sha256"] == FEATURE_CONFIG_SHA == io.sha256_file(archive), FEATURE_CONFIG_SHA)
        self.ok("feature_blocks_identical", all(old[b] == self.gate[b] for b in FEATURE_BLOCKS), list(FEATURE_BLOCKS))
        self.ok("feature_columns_as_extracted", summ["feature_columns"] == list(gf.G1_COLUMNS) + list(gf.G2_COLUMNS), "30 columns")
        self.gman = json.loads((repo / "artifacts/gate/v0_2/hash_manifest.json").read_text(encoding="utf-8"))["files"]
        self.dman = json.loads((repo / "artifacts/d1/v0_1/hash_manifest.json").read_text(encoding="utf-8"))["files"]

    def ok(self, name: str, cond: bool, detail: Any) -> None:
        self.checks.append({"check": name, "status": "PASS" if cond else "FAIL", "detail": detail})
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)
        if not cond:
            raise RuntimeError(f"preflight failed: {name}")

    def _verified(self, rel: str, manifest: dict[str, str]) -> Path:
        p = self.repo / rel
        self.ok(f"hash_{Path(rel).name}", io.sha256_file(p) == manifest[rel], manifest[rel])
        return p

    def features(self, role: str) -> pd.DataFrame:
        p = self._verified(f"artifacts/gate/v0_2/features/{role}_features.csv", self.gman)
        return pd.read_csv(p, float_precision="round_trip")

    def keys(self, role: str) -> pd.DataFrame:
        p = self._verified(f"artifacts/d1/v0_1/keys/{role}_rows.csv", self.dman)
        return pd.read_csv(p, usecols=["row_id", "patient_id", "mask_id"])

    def d1(self, role: str) -> pd.DataFrame:
        p = self._verified(f"artifacts/d1/v0_1/offline_labels/{role}_OFFLINE_d1_labels.csv", self.dman)
        return pd.read_csv(p, usecols=["row_id", "D1"])              # nothing else from the offline label file

    def m2(self, role: str) -> pd.DataFrame:
        rel = f"artifacts/gate/v0_2/comparators/{role}_M2_e1_recon_margin.csv"
        summ = json.loads((self.repo / "artifacts/gate/v0_2/comparators/comparators_summary.json").read_text(encoding="utf-8"))
        self.ok(f"hash_{Path(rel).name}", io.sha256_file(self.repo / rel) == summ["per_role"][role]["output_sha256"], summ["per_role"][role]["output_sha256"])
        return pd.read_csv(self.repo / rel, float_precision="round_trip")


def stage_dir(repo: Path, stage: str) -> Path:
    d = repo / OUT / STAGE_DIRS[stage]
    if d.exists():
        raise RuntimeError(f"{d} exists; refusing to overwrite")
    if (repo / THRESHOLD_LOCK).exists():
        raise RuntimeError("the threshold lock already exists; M4b-1 stages are closed")
    d.mkdir(parents=True)
    return d


def xy(inp: Inputs, feats: pd.DataFrame, method: str, role: str) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    mat = method_matrix(feats, method, inp.p2)
    X, y = attach_d1_target(mat, inp.d1(role), inp.p2)
    return X, y.to_numpy(), mat["row_id"]


# ------------------------------------------------------------------------------------------------------------------
# Stage 01: patient-grouped CV and refit on B_gate
# ------------------------------------------------------------------------------------------------------------------
def run_fit(repo: Path, methods: list[str]) -> dict[str, Any]:
    stage, role = "fit", "B_gate"
    check_role(role, stage)
    inp = Inputs(repo)
    out = stage_dir(repo, stage)
    (out / "models").mkdir()
    rf_cfg, seed = inp.gate["rf"], inp.gate["rf"]["fixed"]["random_state"]
    feats, keys = inp.features(role), inp.keys(role)
    inp.ok("rows_aligned", feats["row_id"].tolist() == keys["row_id"].tolist(), len(feats))
    groups = keys["patient_id"].to_numpy()
    _, y0, _ = xy(inp, feats, "gate_A", role)
    folds = make_folds(y0, groups, 5, seed)
    fold_tab = pd.DataFrame({"row_id": keys["row_id"], "patient_id": groups, "fold": folds, "D1": y0})
    per_fold = fold_tab.groupby("fold").agg(rows=("row_id", "size"), patients=("patient_id", "nunique"), d1=("D1", "sum")).reset_index()
    inp.ok("patients_in_one_fold_only", fold_tab.groupby("patient_id")["fold"].nunique().max() == 1, int(fold_tab["patient_id"].nunique()))
    fold_tab.drop(columns="D1").to_csv(out / "fold_assignment.csv", index=False)
    summary: dict[str, Any] = {"stage": "01 fit", "role": role, "rows": len(fold_tab), "patients": int(fold_tab["patient_id"].nunique()),
                               "d1_rows": int(y0.sum()), "per_fold": per_fold.to_dict(orient="records"), "methods": {}}
    for method in methods:
        t0 = time.time()
        X, y, rid = xy(inp, feats, method, role)
        assert (y == y0).all() and (rid.to_numpy() == keys["row_id"].to_numpy()).all()
        res, oofs = [], {}
        for params in grid(rf_cfg):
            oof = np.full(len(y), np.nan)
            for k in range(5):
                tr, te = folds != k, folds == k
                m = rf_model(rf_cfg, params).fit(X[tr], y[tr])
                oof[te] = score(m, X[te])
            ll = log_loss(y, np.clip(oof, CLIP, 1 - CLIP), labels=[0, 1])
            res.append({**params, "log_loss": float(ll), "oof_auroc_descriptive": float(roc_auc_score(y, oof))})
            oofs[(params["min_samples_leaf"], params["max_features"])] = oof
            print(f"  {method} {params}: log loss {ll:.6f}", flush=True)
        best = select_params(res)
        key = (best["min_samples_leaf"], best["max_features"])
        pd.DataFrame(res).to_csv(out / f"cv_grid_{method}.csv", index=False, float_format="%.17g")
        pd.DataFrame({"row_id": rid, "fold": folds, "oof_score": oofs[key]}).to_csv(out / f"oof_{method}.csv", index=False, float_format="%.17g")
        model = rf_model(rf_cfg, best).fit(X, y)
        model.n_jobs = 1
        mpath = out / "models" / f"{method}.joblib"
        joblib.dump(model, mpath)
        os.chmod(mpath, 0o444)
        summary["methods"][method] = {"features": FIT_METHODS[method], "grid": res, "selected": best,
                                      "model": {"path": mpath.relative_to(repo).as_posix(), "sha256": io.sha256_file(mpath)},
                                      "seconds": round(time.time() - t0, 1)}
    return finish(repo, out, inp, summary)


# ------------------------------------------------------------------------------------------------------------------
# Stage 02: Platt calibration on 9a
# ------------------------------------------------------------------------------------------------------------------
def load_models(repo: Path) -> tuple[dict[str, Any], dict[str, RandomForestClassifier]]:
    fit = json.loads((repo / OUT / "01_fit" / "summary.json").read_text(encoding="utf-8"))
    models = {}
    for method, info in fit["methods"].items():
        p = repo / info["model"]["path"]
        if io.sha256_file(p) != info["model"]["sha256"]:
            raise RuntimeError(f"{p} differs from the stage-01 record")
        models[method] = joblib.load(p)
    return fit, models


def raw_scores(inp: Inputs, models: dict[str, Any], role: str) -> tuple[pd.DataFrame, np.ndarray]:
    feats = inp.features(role)
    cols: dict[str, Any] = {"row_id": feats["row_id"]}
    y = None
    for method, model in models.items():
        X, yy, rid = xy(inp, feats, method, role)
        y = yy if y is None else y
        assert (yy == y).all()
        cols[method] = score(model, X)
    m2 = inp.m2(role)
    if m2["row_id"].tolist() != feats["row_id"].tolist():
        raise RuntimeError("M2 rows are not aligned with the feature rows")
    cols[M2] = m2["e1_recon_margin_score"].to_numpy(dtype=np.float64)
    return pd.DataFrame(cols), y


def run_calibrate(repo: Path) -> dict[str, Any]:
    stage, role = "calibrate", "9a_calibration"
    check_role(role, stage)
    inp = Inputs(repo)
    out = stage_dir(repo, stage)
    fit, models = load_models(repo)
    raw, y = raw_scores(inp, models, role)
    n_events = int(y.sum())
    platt = n_events >= inp.p2["calibration"]["min_d1_events"]
    inp.ok("platt_rule", platt, f"{n_events} D1 rows on 9a >= {inp.p2['calibration']['min_d1_events']}")
    summary: dict[str, Any] = {"stage": "02 calibrate", "role": role, "rows": len(raw), "d1_rows": n_events, "rule": "Platt", "methods": {}}
    cal_tab = {"row_id": raw["row_id"]}
    for method in [c for c in raw.columns if c != "row_id"]:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(platt_input(raw[method], method), y)
        p = out / f"calibrator_{method}.joblib"
        joblib.dump(lr, p)
        os.chmod(p, 0o444)
        cs = calibrated(lr, raw[method].to_numpy(), method)
        cal_tab[f"{method}_raw"], cal_tab[f"{method}_calibrated"] = raw[method], cs
        summary["methods"][method] = {"input": "raw score" if method == M2 else "logit(clip(score, 1e-6, 1 - 1e-6))",
                                      "coef": float(lr.coef_[0, 0]).hex(), "intercept": float(lr.intercept_[0]).hex(),
                                      "coef_repr": repr(float(lr.coef_[0, 0])), "intercept_repr": repr(float(lr.intercept_[0])),
                                      "n_iter": int(lr.n_iter_[0]), "calibrator": {"path": p.relative_to(repo).as_posix(), "sha256": io.sha256_file(p)},
                                      "brier_9a_descriptive": float(np.mean((cs - y) ** 2)), "mean_calibrated_9a": float(cs.mean()),
                                      "d1_rate_9a": float(y.mean())}
    pd.DataFrame(cal_tab).to_csv(out / "9a_scores.csv", index=False, float_format="%.17g")
    summary["fit_summary_sha256"] = io.sha256_file(repo / OUT / "01_fit" / "summary.json")
    return finish(repo, out, inp, summary)


# ------------------------------------------------------------------------------------------------------------------
# Stage 03: thresholds on 9b and the threshold lock
# ------------------------------------------------------------------------------------------------------------------
def load_calibrators(repo: Path) -> tuple[dict[str, Any], dict[str, LogisticRegression]]:
    cal = json.loads((repo / OUT / "02_calibration" / "summary.json").read_text(encoding="utf-8"))
    out = {}
    for method, info in cal["methods"].items():
        p = repo / info["calibrator"]["path"]
        if io.sha256_file(p) != info["calibrator"]["sha256"]:
            raise RuntimeError(f"{p} differs from the stage-02 record")
        out[method] = joblib.load(p)
    return cal, out


def calibrated_scores(repo: Path, inp: Inputs, role: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw -> calibrated scores of every method on ``role`` with the recorded models and calibrators (fresh loads)."""
    _, models = load_models(repo)
    _, cals = load_calibrators(repo)
    raw, y = raw_scores(inp, models, role)
    tab = {"row_id": raw["row_id"]}
    for method in [c for c in raw.columns if c != "row_id"]:
        tab[method] = calibrated(cals[method], raw[method].to_numpy(), method)
    return pd.DataFrame(tab), y


def run_threshold(repo: Path) -> dict[str, Any]:
    stage, role = "threshold", "9b_threshold"
    check_role(role, stage)
    inp = Inputs(repo)
    out = stage_dir(repo, stage)
    cs, y = calibrated_scores(repo, inp, role)
    again, _ = calibrated_scores(repo, inp, role)
    same = all(np.array_equal(cs[c].to_numpy(), again[c].to_numpy()) for c in cs.columns if c != "row_id")
    inp.ok("9b_scores_bitwise_reproducible", same, "two independent scoring passes with fresh model loads")
    spath = out / "9b_calibrated_scores.csv"
    cs.to_csv(spath, index=False, float_format="%.17g")
    os.chmod(spath, 0o444)
    back = pd.read_csv(spath, float_precision="round_trip")
    inp.ok("score_file_round_trip", all(np.array_equal(back[c].to_numpy(), cs[c].to_numpy()) for c in cs.columns if c != "row_id"), "%.17g")
    c_min = float(inp.p2["threshold"]["min_coverage"])
    methods = {}
    for method in [c for c in cs.columns if c != "row_id"]:
        r = sel.select_threshold(cs[method].to_numpy(), y, c_min=c_min)
        rec = pd_.threshold_record(r["t"])
        inp.ok(f"threshold_round_trip_{method}", pd_.threshold_from_record(rec) == r["t"], rec["hex"])
        methods[method] = {"t": rec, "coverage_9b": r["coverage"], "risk_9b": r["risk"], "n_candidates": r["n_candidates"],
                           "n_feasible": r["n_feasible"]}
    fit = json.loads((repo / OUT / "01_fit" / "summary.json").read_text(encoding="utf-8"))
    cal = json.loads((repo / OUT / "02_calibration" / "summary.json").read_text(encoding="utf-8"))
    for method in methods:
        methods[method]["calibrator_sha256"] = cal["methods"][method]["calibrator"]["sha256"]
        if method in fit["methods"]:
            methods[method]["model_sha256"] = fit["methods"][method]["model"]["sha256"]
    env = env_record()
    lock = {"file": THRESHOLD_LOCK, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "authorisation": AUTHORISATION,
            "protocol_v0_2": {"file": "configs/protocol_v0_2.yaml", "sha256": io.sha256_file(repo / "configs/protocol_v0_2.yaml"),
                              "lock_sha256": io.sha256_file(repo / "artifacts/locks/protocol_v0_2.lock.json")},
            "gate_preregistration_sha256": io.sha256_file(repo / "configs/gate_m4b_v0_2.yaml"),
            "rule": "distinct calibrated 9b scores; accept if score <= t; minimise pooled accepted D1 rate s.t. coverage >= "
                    f"{c_min}; tie-break lowest risk, highest coverage, smallest t",
            "c_min": c_min, "rows_9b": len(cs), "d1_rows_9b": int(y.sum()), "methods": methods,
            "M0_accept_all": "no threshold (constant score; coverage 1.0)",
            "not_locked": {"E_dir_margin": "NOT IMPLEMENTED: E_dir margin scores were never computed on 9a/9b (secondary, label-based only)"},
            "calibrated_9b_scores": {"path": spath.relative_to(repo).as_posix(), "sha256": io.sha256_file(spath)},
            "stage_records": {"fit_summary_sha256": io.sha256_file(repo / OUT / "01_fit" / "summary.json"),
                              "calibration_summary_sha256": io.sha256_file(repo / OUT / "02_calibration" / "summary.json")},
            "environment": {"os": env["os"], "python": env["python"],
                            **{k: env["packages"][k] for k in ("numpy", "scipy", "scikit-learn", "pandas", "joblib")}},
            "code_tree_sha256": code_tree_sha256(repo)["tree_sha256"],
            "fold10_rule": "this lock does not authorise any fold-10 access; the final run needs a separate explicit owner authorisation"}
    write_readonly(repo / THRESHOLD_LOCK, json.dumps(lock, indent=2) + "\n")
    summary = {"stage": "03 threshold", "role": role, "rows": len(cs), "d1_rows": int(y.sum()), "methods": methods,
               "threshold_lock": {"path": THRESHOLD_LOCK, "sha256": io.sha256_file(repo / THRESHOLD_LOCK)}}
    return finish(repo, out, inp, summary)


def finish(repo: Path, out: Path, inp: Inputs, summary: dict[str, Any]) -> dict[str, Any]:
    summary.update({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "authorisation": AUTHORISATION,
                    "preflight_checks": inp.checks, "code_tree_sha256": code_tree_sha256(repo)["tree_sha256"],
                    "environment": env_record(), "fold10_loaded": False,
                    "files": {p.relative_to(repo).as_posix(): io.sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    write_readonly(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    for p in out.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o444)
    print(f"wrote {out.relative_to(repo).as_posix()}/summary.json", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M4b-1 stages 01-03 (gate fit, calibration, threshold lock).")
    ap.add_argument("stage", choices=["fit", "calibrate", "threshold"])
    ap.add_argument("--skip-m3", action="store_true", help="drop the optional M3 ablation (deadline contingency)")
    args = ap.parse_args(argv)
    if args.stage == "fit":
        run_fit(REPO, [m for m in FIT_METHODS if not (args.skip_m3 and m == "M3_signal_quality_only")])
    elif args.stage == "calibrate":
        run_calibrate(REPO)
    else:
        run_threshold(REPO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
