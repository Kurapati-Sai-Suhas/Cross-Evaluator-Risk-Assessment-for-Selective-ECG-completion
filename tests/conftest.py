"""Shared test fixtures.

Two kinds of tests use these fixtures:
  * Synthetic tests: tiny hand-made tables whose correct answers are known in
    advance. They run anywhere, even without the PTB-XL files.
  * Real-data tests: run against data/raw/ptbxl/*.csv. They are SKIPPED (not
    failed) with a clear reason if the files have not been downloaded yet.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import ptbxl_io as io

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "data_audit.yaml"
TARGETS = ["CLBBB", "ILBBB", "CRBBB", "IRBBB"]


@pytest.fixture
def config() -> dict:
    return io.load_config(CONFIG_PATH)


@pytest.fixture
def statements() -> pd.DataFrame:
    """Synthetic scp_statements table (descriptions are placeholders, not PTB-XL text)."""
    codes = ["CLBBB", "ILBBB", "CRBBB", "IRBBB", "NORM", "AMI", "SR"]
    return pd.DataFrame(
        {
            "description": [f"synthetic {c}" for c in codes],
            "diagnostic": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
            "form": [np.nan] * 7,
            "rhythm": [np.nan] * 6 + [1.0],
            "diagnostic_class": ["CD", "CD", "CD", "CD", "NORM", "MI", np.nan],
            "diagnostic_subclass": ["CLBBB", "ILBBB", "CRBBB", "IRBBB", "NORM", "AMI", np.nan],
        },
        index=pd.Index(codes, name="code"),
    )


def _row(ecg_id: int, patient_id: float, fold: int, scp: str) -> dict:
    return {
        "ecg_id": ecg_id,
        "patient_id": patient_id,
        "age": 60.0,
        "sex": 1,
        "scp_codes": scp,
        "strat_fold": fold,
        "filename_lr": f"records100/00000/{ecg_id:05d}_lr",
        "filename_hr": f"records500/00000/{ecg_id:05d}_hr",
    }


@pytest.fixture
def small_db() -> pd.DataFrame:
    """Hand-made database with known answers (see test_audit_ptbxl.py for the expectations).

    patient 101: two CLBBB records (likelihood 100 and 50), fold 1
    patient 102: IRBBB + NORM on one record, fold 2
    patient 103: CRBBB (likelihood 0 = unknown) in one record, IRBBB in another, fold 9
    patient 104: CLBBB + ILBBB on the same record, fold 10
    patients 105-110: NORM only, folds 3-8 (so all ten folds exist)
    """
    rows = [
        _row(1, 101.0, 1, "{'CLBBB': 100.0, 'SR': 0.0}"),
        _row(2, 101.0, 1, "{'CLBBB': 50.0}"),
        _row(3, 102.0, 2, "{'IRBBB': 100.0, 'NORM': 100.0}"),
        _row(4, 103.0, 9, "{'CRBBB': 0.0, 'AMI': 80.0}"),
        _row(5, 103.0, 9, "{'IRBBB': 100.0}"),
        _row(6, 104.0, 10, "{'ILBBB': 100.0, 'CLBBB': 100.0}"),
    ]
    rows += [_row(6 + i, 104.0 + i, 2 + i, "{'NORM': 100.0, 'SR': 0.0}") for i in range(1, 7)]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_db() -> pd.DataFrame:
    """Larger random database (seeded) for split tests: 600 patients, 1-3 records each."""
    rng = np.random.default_rng(7)
    rows, ecg_id = [], 1
    probs = {"CLBBB": 0.05, "ILBBB": 0.01, "CRBBB": 0.05, "IRBBB": 0.08}
    for pid in range(1, 601):
        fold = int(rng.integers(1, 11))
        for _ in range(int(rng.integers(1, 4))):
            codes = {c: 100.0 for c, p in probs.items() if rng.random() < p} or {"NORM": 100.0}
            rows.append(_row(ecg_id, float(pid), fold, repr(codes)))
            ecg_id += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def real_data():
    """Real PTB-XL metadata, or skip with the next action if it is not downloaded."""
    cfg = io.load_config(CONFIG_PATH)
    db_path = REPO_ROOT / cfg["dataset"]["database_csv"]
    st_path = REPO_ROOT / cfg["dataset"]["statements_csv"]
    if not (db_path.is_file() and st_path.is_file()):
        pytest.skip(
            "Real PTB-XL CSVs not found in data/raw/ptbxl/. "
            "Run: .\\.venv\\Scripts\\python.exe -m src.data.download_ptbxl_metadata"
        )
    db = io.load_database(db_path)
    statements = io.load_statements(st_path)
    codes, failures = io.parse_all_scp_codes(db)
    return SimpleNamespace(config=cfg, db=db, statements=statements, codes=codes,
                           failures=failures, db_path=db_path, st_path=st_path)
