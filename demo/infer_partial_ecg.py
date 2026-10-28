"""EXPLORATORY DEMO, NOT VALIDATED: accept-or-abstain for ONE user-supplied partial 12-lead ECG.

It runs the frozen, hash-checked pipeline of the capstone project unchanged:
    observed leads -> G1/G2 gate features -> Gate B Random Forest -> Platt (9a) -> locked 9b threshold -> ACCEPT / ABSTAIN
    and, only on ACCEPT, L2G reconstruction of the missing leads with exact copy-through of the observed leads.
Nothing is trained, calibrated or re-thresholded. It never reads the complete reference ECG, labels, D1, offline files or
fold-10 data, and it refuses PTB-XL fold-10 (test) records. It is not externally or clinically validated: the gate was
evaluated once, on PTB-XL fold 10, with synthetic whole-lead missingness (final report, Sections 8-10).

Input, one of:
  --csv FILE        one column per OBSERVED lead, header names from I II III AVR AVL AVF V1 V2 V3 V4 V5 V6 (any order);
                    500 Hz; millivolts; at least 4,096 rows (5,000 rows = 10 s, as in PTB-XL, is the validated length;
                    longer input is cut to its first 5,000 rows). Missing leads are simply absent.
  --wfdb PATH --observed I,III,V1
                    a 500 Hz 12-lead WFDB record in PTB-XL lead order (e.g. a PTB-XL records500 file); only the
                    listed leads are kept, the others are discarded before anything else is computed.
The observed-lead set must be one of the 258 L2G lead sets (any single lead; or 2-7 leads from I, III, V1-V6).

    .venv/Scripts/python.exe demo/infer_partial_ecg.py --csv my_ecg.csv OUT_DIR
    .venv/Scripts/python.exe demo/infer_partial_ecg.py --wfdb C:/PTB-XL/records500/00000/00017_hr --observed I,V2 OUT_DIR

Output (OUT_DIR must not exist): decision.json; on ACCEPT also reconstruction_250Hz_minmax.csv (2,048 rows x 12 leads,
250 Hz, each lead min-max scaled to [0, 1] as the model sees it; observed leads are the preprocessed input, unchanged).

Fail-closed release (version 2, 2026-10-26): an accepted reconstruction is released only if the copy-through invariant
holds exactly and every value is finite; otherwise no reconstruction file is written and decision.json records outcome
REFUSED_INVALID_RECONSTRUCTION. OUT_DIR is created only after every computation has succeeded, and decision.json is
written last. Exit codes: 0 = RELEASED or ABSTAINED; 1 = input, configuration or asset refusal (nothing written);
3 = REFUSED_INVALID_RECONSTRUCTION (decision.json only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data import missingness as mz  # noqa: E402
from src.data import preprocess as pp  # noqa: E402
from src.data import ptbxl_io as io  # noqa: E402
from src.data.ptbxl_waveforms import LEAD_INDEX, LEAD_NAMES  # noqa: E402
from src.eval import primary_decision as pdec  # noqa: E402
from src.gate import features as gf  # noqa: E402
from src.gate import m4b1  # noqa: E402

STATUS = "EXPLORATORY DEMO - NOT VALIDATED (no external or clinical validation; research prototype only)"
EXPECTED = {"configs/protocol_v0_2.yaml": "38ad0a8669015b442bf80dac35438d7336c3da9bfca4f7804442b95fe5260773",
            "artifacts/locks/threshold_lock_v0_2.json": "bc69c3011739a342b43a8b42b55840a3a8c25d9a5a7bc3ba9c9c3de52568d8de",
            "configs/preprocessing.yaml": None,           # hashed and reported, not pinned here
            "configs/d1_generation_v0_1.yaml": None}
N_SAMPLES = 5000                                       # 10 s at 500 Hz, as in PTB-XL (validated length)
MAX_ABS_MV = 50.0                                      # PTB-XL millivolt signals stay far below this (16-bit ADC range)
MIN_PEAK_MV = 0.01                                     # below this every observed lead is near-flat (or not in mV)
METHOD = "gate_B"
LEAD_SETS = {frozenset(c) for c in mz.l2g_lead_combinations()}


def _sha(rel: str) -> str:
    return io.sha256_file(REPO / rel)


def check_frozen() -> dict[str, str]:
    """SHA-256 of the configs and locks the demo depends on; refuses to run if a pinned one differs."""
    out = {}
    for rel, exp in EXPECTED.items():
        out[rel] = _sha(rel)
        if exp is not None and out[rel] != exp:
            raise SystemExit(f"{rel} differs from the pinned SHA-256; refusing to run")
    return out


def refuse_fold10(path: Path) -> None:
    """Refuse a PTB-XL test-fold record, looked up by its file name (e.g. 00017_hr -> ecg_id 17) in the frozen, hash-checked
    split manifest. Only metadata is read. A CSV made from a fold-10 record cannot be detected: do not use fold 10."""
    from src.data.freeze_split_manifest import load_frozen_manifest
    stem = path.name.split("_")[0].split(".")[0]
    if not stem.isdigit():
        return
    man = load_frozen_manifest(io.load_config(REPO / "configs/data_audit.yaml"), REPO)
    hit = man.loc[man["ecg_id"] == int(stem), "role"]
    if len(hit) and hit.iloc[0] == "test_fold10":
        raise SystemExit("this PTB-XL record is in the test fold (fold 10), which is spent; refusing")


def load_csv(path: Path) -> tuple[np.ndarray, list[str], int]:
    try:
        df = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SystemExit(f"cannot read CSV {path}: {exc}")
    cols = [str(c).strip().upper() for c in df.columns]
    unknown = [c for c in cols if c not in LEAD_INDEX]
    if unknown or len(set(cols)) != len(cols) or not cols:
        raise SystemExit(f"CSV header must be distinct lead names from {list(LEAD_NAMES)}; got {list(df.columns)}")
    x = np.zeros((len(LEAD_NAMES), len(df)), dtype=np.float64)
    for c, raw in zip(cols, df.columns):
        try:
            x[LEAD_INDEX[c]] = pd.to_numeric(df[raw], errors="raise").to_numpy(np.float64)
        except (ValueError, TypeError):
            raise SystemExit(f"column {raw!r} contains non-numeric values")
    return x, [lead for lead in LEAD_NAMES if lead in cols], len(df)


def load_wfdb(path: Path, observed: list[str]) -> tuple[np.ndarray, list[str], int]:
    import wfdb
    rec = wfdb.rdrecord(str(path), physical=True)
    names = [str(n).upper() for n in rec.sig_name]
    if names != list(LEAD_NAMES) or rec.fs != 500 or any(u != "mV" for u in rec.units):
        raise SystemExit(f"WFDB record must be 500 Hz, mV, leads {list(LEAD_NAMES)}; got {rec.fs} Hz, {names}, {rec.units}")
    full = np.asarray(rec.p_signal, dtype=np.float64).T
    x = np.zeros_like(full)
    obs = [lead for lead in LEAD_NAMES if lead in {o.strip().upper() for o in observed}]
    if len(obs) != len(observed):
        raise SystemExit(f"--observed must name distinct leads from {list(LEAD_NAMES)}; got {observed}")
    for lead in obs:                                       # every unlisted lead is discarded here
        x[LEAD_INDEX[lead]] = full[LEAD_INDEX[lead]]
    return x, obs, full.shape[1]


def prepare(x: np.ndarray, observed: list[str], n_in: int) -> tuple[np.ndarray, list[str]]:
    """Checks the input and cuts it to the validated length; returns (12, n) mV with missing leads = 0, and warnings."""
    warnings = []
    if frozenset(observed) not in LEAD_SETS:
        raise SystemExit(f"observed leads {observed} are not one of the 258 L2G lead sets (any single lead, or 2-7 of "
                         "I, III, V1-V6); the gate and the reconstructor were never evaluated on this set")
    if n_in < gf.WINDOW[1]:
        raise SystemExit(f"need at least {gf.WINDOW[1]} samples at 500 Hz (8.192 s); got {n_in}")
    if n_in > N_SAMPLES:
        x = x[:, :N_SAMPLES]
        warnings.append(f"input cut to its first {N_SAMPLES} samples (10 s), as in PTB-XL")
    elif n_in < N_SAMPLES:
        warnings.append(f"{n_in} samples instead of {N_SAMPLES}: filter edge effects differ from the validated setting")
    if not np.isfinite(x).all():
        raise SystemExit("input contains NaN or infinite values")
    peak = float(np.abs(x[[LEAD_INDEX[lead] for lead in observed]]).max())
    if peak >= MAX_ABS_MV:                                 # e.g. microvolts: the G2 quality features would be meaningless
        raise SystemExit(f"observed leads reach {peak:.1f} in magnitude; millivolt ECGs stay below {MAX_ABS_MV:.0f}, "
                         "so the input is probably not in mV; refusing")
    if peak < MIN_PEAK_MV:
        warnings.append(f"every observed lead stays below {MIN_PEAK_MV} mV: near-flat input, or not in mV")
    return x, warnings


_CACHE: dict = {}


def gate_assets() -> dict:
    """Frozen Gate B model and calibrator (each checked against its stage record's SHA-256), protocol and locked threshold."""
    if "gate" not in _CACHE:
        _, models = m4b1.load_models(REPO)
        _, cals = m4b1.load_calibrators(REPO)
        tl = json.loads((REPO / m4b1.THRESHOLD_LOCK).read_text(encoding="utf-8"))
        _CACHE["gate"] = {"model": models[METHOD], "cal": cals[METHOD], "tl": tl,
                          "t": pdec.threshold_from_record(tl["methods"][METHOD]["t"]),
                          "p2": io.load_config(REPO / "configs/protocol_v0_2.yaml")}
    return _CACHE["gate"]


def gate_score(x: np.ndarray, observed: list[str]) -> dict:
    """G1/G2 features from the observed raw 500 Hz leads only, then the frozen Gate B model, Platt calibrator and lock."""
    g = gate_assets()
    feats = {"row_id": "demo:0", **gf.row_features(x, observed)}
    mat = m4b1.method_matrix(pd.DataFrame([feats]), METHOD, g["p2"])   # whitelist and column order of the pre-registration
    raw = m4b1.score(g["model"], mat.drop(columns="row_id"))
    s = float(m4b1.calibrated(g["cal"], raw, METHOD)[0])
    return {"features": {k: v for k, v in feats.items() if k != "row_id"}, "raw_score": float(raw[0]),
            "calibrated_score": s, "threshold_repr": repr(g["t"]), "decision": "ACCEPT" if s <= g["t"] else "ABSTAIN",
            "model_sha256": g["tl"]["methods"][METHOD]["model_sha256"],
            "calibrator_sha256": g["tl"]["methods"][METHOD]["calibrator_sha256"]}


def generator() -> dict:
    import torch
    from src.eval import generate_d1 as gd
    from src.models import l2g

    if "gen" not in _CACHE:
        cfg = io.load_config(REPO / "configs/d1_generation_v0_1.yaml")
        if io.sha256_file(REPO / cfg["frozen"]["g_checkpoint"]) != cfg["frozen"]["g_sha256"]:
            raise SystemExit("the L2G checkpoint differs from its pinned SHA-256; refusing to run")
        gd.set_deterministic(cfg["scoring"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen = l2g.ECGen(24).to(device)
        gen.load_state_dict(torch.load(REPO / cfg["frozen"]["g_checkpoint"], map_location=device, weights_only=False)["generator"])
        gen.eval()
        _CACHE["gen"] = {"gen": gen, "device": device, "cfg": cfg, "prep": io.load_config(REPO / "configs/preprocessing.yaml")}
    return _CACHE["gen"]


def l2g_input(x: np.ndarray) -> np.ndarray:
    """The model input exactly as on the D1 scoring path: filtered and resampled, stored as float32 (as cached), then the
    2,048-sample window and per-lead min-max."""
    cfg = io.load_config(REPO / "configs/d1_generation_v0_1.yaml")
    prep = io.load_config(REPO / "configs/preprocessing.yaml")
    y = pp.filter_resample(x, 500, prep).astype(np.float32)
    return pp.crop_normalize(y, prep, int(cfg["data"]["window_offset"])).x.astype(np.float32)


def release_check(x_in: np.ndarray, x_hat: np.ndarray, observed: list[str], violations: int) -> list[str]:
    """Conditions an accepted reconstruction must meet before release (fail closed: any problem blocks release)."""
    if x_hat.shape != x_in.shape:
        return [f"reconstruction shape {x_hat.shape} differs from the model input shape {x_in.shape}"]
    problems = []
    if violations:
        problems.append(f"{violations} copy-through violations reported by the reconstruction step")
    idx = [LEAD_INDEX[lead] for lead in observed]
    if not np.array_equal(x_hat[idx], x_in[idx]):
        problems.append("observed samples are not reproduced exactly (copy-through invariant broken)")
    if not np.isfinite(x_hat).all():
        problems.append("reconstruction contains NaN or infinite values")
    return problems


def reconstruct(x: np.ndarray, observed: list[str]) -> tuple[np.ndarray, dict]:
    """The D1 scoring path's preprocessing (float32 after filtering and resampling, as cached), then L2G with copy-through."""
    import torch
    from src.eval import generate_d1 as gd

    G = generator()
    cfg, device = G["cfg"], G["device"]
    xin = l2g_input(x)
    lm = torch.from_numpy(mz.lead_mask(observed)[None]).to(device)
    x_hat, violations = gd.reconstruct(G["gen"], torch.from_numpy(xin[None]).to(device), lm, int(cfg["scoring"]["batch_rows"]))
    return x_hat[0].cpu().numpy(), {"device": str(device), "copy_through_violations": int(violations),
                                    "g_sha256": cfg["frozen"]["g_sha256"], "batch_rows": int(cfg["scoring"]["batch_rows"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=STATUS)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path)
    src.add_argument("--wfdb", type=Path)
    ap.add_argument("--observed", help="with --wfdb: comma-separated observed leads, e.g. I,III,V1")
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    if a.out_dir.exists():
        raise SystemExit(f"{a.out_dir} exists; refusing to overwrite")
    frozen = check_frozen()
    if a.csv:
        refuse_fold10(a.csv)
        x, observed, n_in = load_csv(a.csv)
        source = {"csv": str(a.csv), "sha256": hashlib.sha256(a.csv.read_bytes()).hexdigest()}
    else:
        if not a.observed:
            raise SystemExit("--wfdb needs --observed")
        refuse_fold10(a.wfdb)
        x, observed, n_in = load_wfdb(a.wfdb, a.observed.split(","))
        source = {"wfdb": str(a.wfdb), "observed_argument": a.observed}
    x, warnings = prepare(x, observed, n_in)
    gate = gate_score(x, observed)                         # the gate decision never depends on a reconstruction
    res = {"status": STATUS, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "input": source,
           "samples_in": n_in, "samples_used": int(x.shape[1]), "observed_leads": observed, "warnings": warnings,
           "gate": gate, "frozen_inputs_sha256": frozen, "demo_sha256": _sha("demo/infer_partial_ecg.py"),
           "policy": "ACCEPT releases the reconstruction; ABSTAIN means the reconstruction should not be relied on",
           "limitations": ["evaluated once on PTB-XL fold 10 with synthetic whole-lead missingness; no external validation",
                           "Gate B withholds most BBB (especially CLBBB) ECGs; ACCEPT is not evidence of diagnostic fidelity",
                           "D1 (the gate's target) is disagreement with one frozen classifier, not clinical harm or truth"]}
    recon = None
    if gate["decision"] == "ACCEPT":
        x_hat, info = reconstruct(x, observed)
        problems = release_check(l2g_input(x), x_hat, observed, info["copy_through_violations"])
        res["release_check"] = {"passed": not problems, "problems": problems}
        if problems:                                       # fail closed: an invalid reconstruction is never released
            res.update(outcome="REFUSED_INVALID_RECONSTRUCTION", released=False)
            res["reconstruction"] = {**info, "file": None, "status": "not released: " + "; ".join(problems)}
        else:
            res.update(outcome="RELEASED", released=True)
            res["reconstruction"] = {**info, "file": "reconstruction_250Hz_minmax.csv",
                                     "units": "per-lead min-max scaled to [0, 1], 250 Hz, 2,048 samples (8.192 s); not mV"}
            recon = x_hat
    else:
        res.update(outcome="ABSTAINED", released=False)
        res["reconstruction"] = "not produced: the gate abstained"
    a.out_dir.mkdir(parents=True)                          # created only after every computation has succeeded
    if recon is not None:                                  # complete file or none: written aside, then renamed
        part = a.out_dir / "reconstruction_250Hz_minmax.csv.partial"
        pd.DataFrame(recon.T, columns=list(LEAD_NAMES)).to_csv(part, index=False, float_format="%.9g")
        os.replace(part, a.out_dir / "reconstruction_250Hz_minmax.csv")
    (a.out_dir / "decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")   # last: marks completion
    print(json.dumps({"outcome": res["outcome"], "released": res["released"], "gate_decision": gate["decision"],
                      "calibrated_score": gate["calibrated_score"], "threshold": gate["threshold_repr"],
                      "observed_leads": observed, "warnings": warnings, "out": str(a.out_dir)}, indent=1))
    return 3 if res["outcome"] == "REFUSED_INVALID_RECONSTRUCTION" else 0


if __name__ == "__main__":
    sys.exit(main())
