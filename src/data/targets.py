"""Diagnostic targets for E1/E2/E_dir, exactly as fixed by protocol v0.1 Section 1 (D-010).

y_k = 1 iff code k (CLBBB or CRBBB) is listed in the record's scp_codes with likelihood >= the protocol's
positive_min_likelihood (0 = listed at any likelihood). Everything else is 0, including records whose only
BBB code is ILBBB and/or IRBBB.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data import ptbxl_io as io


def target_matrix(db: pd.DataFrame, ecg_ids: list[int], protocol: dict[str, Any]) -> np.ndarray:
    """(N, len(primary_outputs)) uint8 labels in the order of ``ecg_ids``."""
    outputs = list(protocol["targets"]["primary_outputs"])
    threshold = float(protocol["targets"]["positive_min_likelihood"])
    raw = db.set_index("ecg_id").loc[ecg_ids, "scp_codes"]
    rows = []
    for text in raw:
        codes = io.parse_scp_codes(text)
        rows.append([int(c in codes and codes[c] >= threshold) for c in outputs])
    return np.asarray(rows, dtype=np.uint8)
