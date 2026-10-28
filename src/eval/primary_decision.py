"""Fixed-sequence H1 -> H2 decision and threshold serialisation for protocol v0.2 (proposal P4).

Reference implementation only, tested on synthetic inputs; it reads no data and is not the M5 pipeline. The contrast
replicates come from ``src.eval.selective`` (delta = risk(method) - risk(comparator), +inf for an undefined replicate).
The decision rule is a bootstrap-based empirical rule, not a finite-sample, formal or clinical-safety guarantee.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from src.eval import selective as sel


def fixed_sequence(h1_point: float, h1_replicates: np.ndarray,
                   h2_replicates: Callable[[], np.ndarray] | np.ndarray | None, alpha: float = 0.05) -> dict[str, Any]:
    """H1 first; H2 is tested only if H1 is rejected (family-wise alpha by the fixed sequence).

    An undefined H1 point estimate (NaN: Gate B accepted no row) means H1 is not rejected and H2 is skipped, whatever the
    replicates say. ``h2_replicates`` may be a callable so that H2 is not even computed when it is not tested.
    """
    h1 = sel.one_sided_decision(h1_replicates, alpha)
    h1_defined = h1_point is not None and not math.isnan(h1_point)
    h1_reject = bool(h1_defined and h1["reject"])
    out: dict[str, Any] = {"H1": {**h1, "point": h1_point if h1_defined else None, "point_defined": h1_defined,
                                  "reject": h1_reject},
                           "H2": {"tested": False, "reject": False}}
    if not h1_defined:
        out["H1"]["note"] = "undefined point estimate (no accepted row): H1 not rejected, H2 not tested"
    if h1_reject:
        reps = h2_replicates() if callable(h2_replicates) else h2_replicates
        if reps is None:
            raise ValueError("H1 was rejected but no H2 replicates were given")
        out["H2"] = {**sel.one_sided_decision(reps, alpha), "tested": True}
    return out


def threshold_record(t: float) -> dict[str, str]:
    """Exact serialisation of a float64 threshold: repr for people, float.hex for an exact round trip."""
    t = float(t)
    if not math.isfinite(t):
        raise ValueError("threshold must be finite")
    return {"repr": repr(t), "hex": t.hex()}


def threshold_from_record(rec: dict[str, str]) -> float:
    """Inverse of ``threshold_record``; refuses a record whose two forms disagree."""
    t = float.fromhex(rec["hex"])
    if repr(t) != rec["repr"]:
        raise ValueError("threshold record: repr and hex disagree")
    return t
