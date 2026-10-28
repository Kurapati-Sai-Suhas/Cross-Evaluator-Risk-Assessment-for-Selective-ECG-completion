"""Patient-cluster bootstrap, as fixed by protocol v0.1 Section 4 (configs/protocol_v0_1.yaml: uncertainty).

Rows (records x masks) of one patient are correlated, so patients - not rows - are resampled.
The point estimate is the row-weighted mean; intervals are bootstrap percentiles.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def patient_cluster_bootstrap(values: np.ndarray, patient_ids: np.ndarray, replicates: int = 10_000,
                              seed: int = 20261030, level: float = 0.95, chunk: int = 500) -> dict[str, Any]:
    """Row-weighted mean of ``values`` with a patient-cluster percentile CI.

    Rows with NaN are dropped first (count reported). For a paired comparison, pass the per-row
    difference between two methods evaluated on the same rows.
    """
    values = np.asarray(values, dtype=float)
    patient_ids = np.asarray(patient_ids)
    keep = np.isfinite(values)
    values, patient_ids = values[keep], patient_ids[keep]
    if values.size == 0:
        raise ValueError("no finite values to bootstrap")
    _, inverse = np.unique(patient_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    n_pat = sums.size
    rng = np.random.default_rng(seed)
    stats = np.empty(replicates)
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        idx = rng.integers(0, n_pat, size=(stop - start, n_pat), dtype=np.int32)
        stats[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1 - level) / 2
    return {
        "point": float(values.mean()),
        "ci_low": float(np.quantile(stats, alpha)),
        "ci_high": float(np.quantile(stats, 1 - alpha)),
        "level": level,
        "n_rows": int(values.size),
        "n_patients": int(n_pat),
        "n_rows_dropped_nan": int((~keep).sum()),
        "replicates": replicates,
        "method": "patient-cluster bootstrap, percentile",
    }


def bootstrap_or_undefined(values: np.ndarray, patient_ids: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Like patient_cluster_bootstrap, but a metric that is undefined for every row (e.g. Pearson r of a
    constant baseline) is reported as undefined instead of raising."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return {"point": None, "ci_low": None, "ci_high": None, "n_rows": 0,
                "note": "undefined for every row (e.g. correlation with a constant signal)"}
    return patient_cluster_bootstrap(values, patient_ids, **kwargs)
