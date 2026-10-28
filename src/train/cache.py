"""Cache of filtered + resampled signals for one split role (Milestone 2c).

Each record goes through ``preprocess.filter_resample`` (the tested L2G preprocessing up to, but not
including, windowing and min-max) once and is stored as float32 (12, 2500) in mV. Windowing and
normalisation happen later per sample, so the training windows at offsets 0 and 452 come from one cache.

Files (outside OneDrive; derived data, can always be rebuilt):
  <cache_dir>/<role>_signals.npy   (N, 12, 2500) float32
  <cache_dir>/<role>_index.csv     ecg_id, patient_id, strat_fold in the same order
  <cache_dir>/<role>_meta.json     hashes of everything the cache depends on (stale-cache guard)
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import preprocess as pp
from src.data import ptbxl_io as io
from src.data.ptbxl_waveforms import load_record


def _filter_one(task: tuple[str, str, dict[str, Any]]) -> np.ndarray:
    root, rel, cfg = task
    return pp.filter_resample(load_record(root, rel, "hr"), 500, cfg).astype(np.float32)


def cache_key(prep_cfg_path: Path, db_path: Path, manifest_sha: str, role: str, ecg_ids: list[int]) -> dict[str, Any]:
    """Everything the cached arrays depend on. If any value changes, the cache is rebuilt."""
    return {
        "role": role,
        "preprocessing_yaml_sha256": io.sha256_file(prep_cfg_path),
        "preprocess_py_sha256": io.sha256_file(Path(pp.__file__)),
        "ptbxl_database_sha256": io.sha256_file(db_path),
        "frozen_manifest_sha256": manifest_sha,
        "n_records": len(ecg_ids),
        "ecg_ids_sha256": hashlib.sha256(",".join(map(str, ecg_ids)).encode()).hexdigest(),
    }


def ensure_cache(role: str, rows: pd.DataFrame, waveform_root: Path, db: pd.DataFrame, prep_cfg: dict[str, Any],
                 key: dict[str, Any], cache_dir: Path, workers: int = 8) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Return (signals, index, meta) for ``rows`` (ecg_id, patient_id, strat_fold), building the cache if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig_path, idx_path, meta_path = (cache_dir / f"{role}_signals.npy", cache_dir / f"{role}_index.csv",
                                     cache_dir / f"{role}_meta.json")
    if sig_path.is_file() and idx_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            signals = np.load(sig_path)
            index = pd.read_csv(idx_path)
            if list(index["ecg_id"]) == list(rows["ecg_id"]) and signals.shape[0] == len(rows):
                meta["reused"] = True
                return signals, index, meta
    filenames = db.set_index("ecg_id").loc[rows["ecg_id"], "filename_hr"].tolist()
    tasks = [(str(waveform_root), f, prep_cfg) for f in filenames]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        arrays = list(pool.map(_filter_one, tasks, chunksize=32))
    signals = np.stack(arrays).astype(np.float32)
    if not np.isfinite(signals).all():
        raise ValueError(f"non-finite values in the {role} cache")
    np.save(sig_path, signals)
    index = rows[["ecg_id", "patient_id", "strat_fold"]].reset_index(drop=True)
    index.to_csv(idx_path, index=False)
    meta = {"key": key, "shape": list(signals.shape), "dtype": "float32",
            "signals_sha256": io.sha256_file(sig_path),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reused": False}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return signals, index, meta
