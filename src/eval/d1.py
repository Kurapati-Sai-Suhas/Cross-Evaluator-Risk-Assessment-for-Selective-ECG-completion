"""Offline D1 targets (protocol v0.1 Section 2) and the guard that keeps them out of gate features.

D1 is EVALUATOR-RELATIVE DISAGREEMENT between the frozen E1 on the complete ECG and on the reconstruction. It is not a
clinical error, not clinical harm and not a safety measure.

For each output y in protocol targets.primary_outputs (CLBBB, CRBBB), with the fixed E1 threshold tau_y:
    z_full_y = 1[p_full_y >= tau_y]          z_rec_y = 1[p_rec_y >= tau_y]
    D1        = 1[ OR_y (z_full_y != z_rec_y) ]
    D1_miss_y = 1[z_full_y == 1 and z_rec_y == 0]   (E1 calls BBB on the complete ECG, not on the reconstruction)
    D1_false_y= 1[z_full_y == 0 and z_rec_y == 1]
Comparisons are made in float32, the precision in which E1 produced both the probabilities and tau.

Separation rule (protocol Section 8): everything in an offline label table (probabilities, thresholds, decisions, D1, PTB-XL
labels) and every identifier is forbidden as a gate feature. ``check_feature_columns`` lists violations and
``attach_d1_target`` is the only supported way to put D1 next to a feature matrix: it joins on ``row_id`` one-to-one and
copies the D1 column only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

KEY_COLUMNS = ("row_id", "ecg_id", "patient_id", "role", "strat_fold", "mask_index", "mask_id", "observed_leads",
               "n_observed")
IDENTIFIER_COLUMNS = frozenset({"row_id", "ecg_id", "patient_id", "role", "strat_fold", "mask_index", "mask_id"})
OFFLINE_PREFIXES = ("p_full_", "p_rec_", "tau_", "z_full_", "z_rec_", "d1", "y_", "near_tau_", "prob_", "logit_",
                    "e1_", "e2_", "edir_", "vttd", "true_", "reference_", "complete_", "label")


def load_thresholds(e1_summary: Path, outputs: Sequence[str]) -> dict[str, np.float32]:
    """The fixed Youden thresholds of the frozen E1, exactly as stored by its training run (never re-estimated)."""
    s = json.loads(Path(e1_summary).read_text(encoding="utf-8"))
    if s.get("eligible") is not True:
        raise RuntimeError(f"{e1_summary}: E1 is not marked eligible; D1 must not be generated (protocol Section 5)")
    if list(s["outputs"]) != list(outputs):
        raise RuntimeError(f"E1 outputs {s['outputs']} differ from the protocol outputs {list(outputs)}")
    return {k: np.float32(s["per_output"][k]["youden_tau_probability"]) for k in outputs}


def decide(p: np.ndarray, tau: np.float32) -> np.ndarray:
    """E1 decision 1[p >= tau] in float32 (a probability exactly equal to tau is positive)."""
    return np.asarray(p, dtype=np.float32) >= np.float32(tau)


def d1_columns(p_full: np.ndarray, p_rec: np.ndarray, tau: dict[str, np.float32],
               outputs: Sequence[str]) -> dict[str, np.ndarray]:
    """All decision and disagreement columns for rows with probabilities p_full, p_rec of shape (N, len(outputs))."""
    p_full, p_rec = np.asarray(p_full, dtype=np.float32), np.asarray(p_rec, dtype=np.float32)
    if p_full.shape != p_rec.shape or p_full.shape[1] != len(outputs):
        raise ValueError(f"shape mismatch: {p_full.shape} vs {p_rec.shape} for outputs {list(outputs)}")
    cols: dict[str, np.ndarray] = {}
    any_disagree = np.zeros(len(p_full), dtype=bool)
    for j, y in enumerate(outputs):
        zf, zr = decide(p_full[:, j], tau[y]), decide(p_rec[:, j], tau[y])
        cols[f"z_full_{y}"], cols[f"z_rec_{y}"] = zf.astype(np.uint8), zr.astype(np.uint8)
        cols[f"D1_miss_{y}"] = (zf & ~zr).astype(np.uint8)
        cols[f"D1_false_{y}"] = (~zf & zr).astype(np.uint8)
        any_disagree |= zf != zr
    cols["D1"] = any_disagree.astype(np.uint8)
    return cols


def near_threshold(p: np.ndarray, tau: np.float32, eps: float) -> np.ndarray:
    """|p - tau| <= eps, computed in float64 from the float32 values (for the boundary report only)."""
    return np.abs(np.asarray(p, dtype=np.float32).astype(np.float64) - float(np.float32(tau))) <= eps


def forbidden_names(protocol: dict[str, Any]) -> set[str]:
    return set(protocol["gate"]["forbidden_inputs"]) | set(IDENTIFIER_COLUMNS)


def check_feature_columns(columns: Sequence[str], protocol: dict[str, Any]) -> list[str]:
    """Names in a would-be gate feature matrix that are forbidden (empty list = no violation found by name)."""
    forbidden = {c.lower() for c in forbidden_names(protocol)}
    bad = []
    for c in columns:
        low = str(c).lower()
        if low in forbidden or low.startswith(OFFLINE_PREFIXES):
            bad.append(str(c))
    return bad


def attach_d1_target(features: pd.DataFrame, labels: pd.DataFrame,
                     protocol: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) for gate fitting: X = features without row_id, y = D1. Only ``row_id`` and ``D1`` are read from labels."""
    if "row_id" not in features.columns:
        raise ValueError("features need a row_id column to be matched with their D1 label")
    problems = check_feature_columns([c for c in features.columns if c != "row_id"], protocol)
    if problems:
        raise ValueError(f"forbidden columns in the feature matrix: {problems}")
    target = labels[["row_id", "D1"]]
    merged = features.merge(target, on="row_id", how="left", validate="one_to_one")
    if merged["D1"].isna().any():
        raise ValueError(f"{int(merged['D1'].isna().sum())} feature rows have no D1 label")
    return merged.drop(columns=["row_id", "D1"]), merged["D1"].astype(int)
