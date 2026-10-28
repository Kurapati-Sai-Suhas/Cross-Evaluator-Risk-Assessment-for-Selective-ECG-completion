"""Load PTB-XL v1.0.3 ECG waveforms with the WFDB reader.

Every array returned here is LEAD-FIRST: shape (12, n_samples), physical units (mV),
float64, in PTB-XL lead order:

    I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6

This order was read from the .hea header files and checked on every record by
src/data/verify_ptbxl_waveforms.py. It is also the order used by L2G-ECG
(paper p. 1: a = 1 is I, a = 2 is II, a = 3 is III, a = 4 is aVR, ...).
The loader refuses any record whose sampling rate, length, lead names or units differ.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from src.data import ptbxl_io as io

# Lead names exactly as written in PTB-XL .hea files.
LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
LEAD_INDEX = {name: i for i, name in enumerate(LEAD_NAMES)}

# "hr" = 500 Hz (records500), "lr" = 100 Hz (records100). Values from the PTB-XL headers.
RESOLUTIONS: dict[str, dict[str, Any]] = {
    "hr": {"column": "filename_hr", "fs": 500, "n_samples": 5000},
    "lr": {"column": "filename_lr", "fs": 100, "n_samples": 1000},
}


class WaveformError(ValueError):
    """A record exists but does not have the expected structure."""


def resolve_waveform_root(config: dict[str, Any], repo_root: Path) -> Path:
    """Folder that contains records500/ and records100/.

    The environment variable PTBXL_ROOT overrides the config value, so another
    machine can use a different location without editing the config file.
    """
    raw = os.environ.get("PTBXL_ROOT") or config["dataset"].get("waveform_root")
    if not raw:
        raise io.MissingInputError(
            "BLOCKER: no waveform folder configured.\n"
            "Next action: set dataset.waveform_root in configs/data_audit.yaml "
            '(or run  $env:PTBXL_ROOT = "C:\\PTB-XL"  in PowerShell).'
        )
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    if not (path / "records500").is_dir() or not (path / "records100").is_dir():
        raise io.MissingInputError(
            f"BLOCKER: PTB-XL waveform folders not found under {path}.\n"
            "Next action: check that this folder contains records500\\ and records100\\, "
            "or point dataset.waveform_root / PTBXL_ROOT at the right folder."
        )
    return path


def load_record(waveform_root: str | Path, relative_path: str, resolution: str = "hr") -> np.ndarray:
    """Read one record, e.g. relative_path='records500/00000/00001_hr'. Returns (12, n) mV."""
    expected = RESOLUTIONS[resolution]
    base = Path(waveform_root) / relative_path
    if not Path(str(base) + ".hea").is_file() or not Path(str(base) + ".dat").is_file():
        raise io.MissingInputError(f"BLOCKER: {base}.hea / .dat not found.")
    record = wfdb.rdrecord(str(base), physical=True)
    if record.fs != expected["fs"]:
        raise WaveformError(f"{relative_path}: sampling rate {record.fs} Hz, expected {expected['fs']} Hz")
    if record.sig_len != expected["n_samples"]:
        raise WaveformError(f"{relative_path}: {record.sig_len} samples, expected {expected['n_samples']}")
    if tuple(record.sig_name) != LEAD_NAMES:
        raise WaveformError(f"{relative_path}: lead order {record.sig_name}, expected {list(LEAD_NAMES)}")
    if any(unit != "mV" for unit in record.units):
        raise WaveformError(f"{relative_path}: units {record.units}, expected mV for every lead")
    signal = np.asarray(record.p_signal, dtype=np.float64).T  # (n, 12) -> (12, n)
    if not np.isfinite(signal).all():
        raise WaveformError(f"{relative_path}: contains NaN or infinite values")
    return signal


def load_by_ecg_id(db: pd.DataFrame, waveform_root: str | Path, ecg_id: int, resolution: str = "hr") -> np.ndarray:
    """Look up an ecg_id in ptbxl_database.csv and load its waveform."""
    rows = db.loc[db["ecg_id"] == ecg_id]
    if len(rows) != 1:
        raise KeyError(f"ecg_id {ecg_id} found {len(rows)} times in the metadata (expected exactly once)")
    return load_record(waveform_root, str(rows.iloc[0][RESOLUTIONS[resolution]["column"]]), resolution)
