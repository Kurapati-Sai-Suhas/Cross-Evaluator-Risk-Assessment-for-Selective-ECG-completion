"""Shared, tested helpers for reading PTB-XL metadata files.

Both ``audit_ptbxl.py`` and ``make_split_manifest.py`` import from here so that
the CSV-loading and ``scp_codes``-parsing rules exist in exactly one place.
Nothing in this module trains or loads any model.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Columns the Milestone 1 scripts rely on. Verified against the PhysioNet
# v1.0.3 ptbxl_database.csv header before this list was written.
REQUIRED_DATABASE_COLUMNS = [
    "ecg_id",
    "patient_id",
    "scp_codes",
    "strat_fold",
    "filename_lr",
    "filename_hr",
]
REQUIRED_STATEMENT_COLUMNS = [
    "description",
    "diagnostic",
    "diagnostic_class",
    "diagnostic_subclass",
]

DOWNLOAD_HINT = (
    "Next action: run\n"
    "    .\\.venv\\Scripts\\python.exe -m src.data.download_ptbxl_metadata\n"
    "or download the file manually from https://physionet.org/content/ptb-xl/1.0.3/ "
    "and save it at the path shown above."
)


class MissingInputError(FileNotFoundError):
    """Raised when a required input file is absent. The message says what to do next."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the YAML config file into a plain dictionary."""
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(
            f"Config file not found: {path}\n"
            "Next action: run the command from the repository root folder "
            "(the folder that contains README.md), or pass --config <path>."
        )
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_file(path: str | Path, what: str) -> Path:
    """Return ``path`` if it exists, otherwise raise MissingInputError with a next action."""
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(f"BLOCKER: {what} not found at: {path}\n{DOWNLOAD_HINT}")
    return path


def load_database(path: str | Path) -> pd.DataFrame:
    """Load ptbxl_database.csv WITHOUT setting an index, so duplicate IDs stay visible."""
    path = require_file(path, "ptbxl_database.csv")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_DATABASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}. "
            "Check that this is the PTB-XL v1.0.3 ptbxl_database.csv file."
        )
    return df


def load_statements(path: str | Path) -> pd.DataFrame:
    """Load scp_statements.csv indexed by SCP code (its first, unnamed column)."""
    path = require_file(path, "scp_statements.csv")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.index.name = "code"
    missing = [c for c in REQUIRED_STATEMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing}. "
            "Check that this is the PTB-XL v1.0.3 scp_statements.csv file."
        )
    if df.index.duplicated().any():
        dupes = sorted(df.index[df.index.duplicated()].unique())
        raise ValueError(f"{path} lists these SCP codes more than once: {dupes}")
    return df


def parse_scp_codes(text: Any) -> dict[str, float]:
    """Parse one ``scp_codes`` cell such as ``"{'NORM': 100.0, 'SR': 0.0}"``.

    Returns {code: likelihood}. Raises ValueError for anything that is not a
    dictionary of string keys and finite numeric likelihoods. ``ast.literal_eval``
    is used (never ``eval``) so the cell cannot execute code.
    """
    if not isinstance(text, str):
        raise ValueError(f"scp_codes value is not text: {text!r}")
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"scp_codes value is not a valid literal: {text!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"scp_codes value is not a dictionary: {text!r}")
    parsed: dict[str, float] = {}
    for key, likelihood in value.items():
        if not isinstance(key, str):
            raise ValueError(f"scp_codes key is not text: {key!r} in {text!r}")
        if isinstance(likelihood, bool) or not isinstance(likelihood, (int, float)):
            raise ValueError(f"likelihood for {key!r} is not a number in {text!r}")
        if not math.isfinite(float(likelihood)):
            raise ValueError(f"likelihood for {key!r} is not finite in {text!r}")
        parsed[key.strip()] = float(likelihood)
    return parsed


def parse_all_scp_codes(db: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Parse every ``scp_codes`` cell.

    Returns (codes, failures): ``codes`` is a Series of dicts aligned with
    ``db.index`` (empty dict where parsing failed); ``failures`` lists the
    ecg_id, raw value and error for every cell that could not be parsed.
    """
    parsed: list[dict[str, float]] = []
    failures: list[dict[str, Any]] = []
    for ecg_id, raw in zip(db["ecg_id"], db["scp_codes"]):
        try:
            parsed.append(parse_scp_codes(raw))
        except ValueError as exc:
            parsed.append({})
            failures.append({"ecg_id": ecg_id, "raw_value": raw, "error": str(exc)})
    codes = pd.Series(parsed, index=db.index, name="scp_codes_parsed", dtype=object)
    return codes, pd.DataFrame(failures, columns=["ecg_id", "raw_value", "error"])


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file (used to pin the exact input version)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_official_checksums(path: str | Path) -> dict[str, str]:
    """Parse PhysioNet's SHA256SUMS.txt (``<hash> <relative path>`` per line)."""
    checksums: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                digest, name = parts
                checksums[name.lstrip("*").strip()] = digest.lower()
    return checksums


def verify_checksum(path: str | Path, checksums_file: str | Path | None) -> dict[str, str]:
    """Compare a file's SHA-256 with the official PhysioNet value.

    Returns a dict with ``status`` in {"MATCH", "MISMATCH", "NOT_LISTED", "NOT_CHECKED"}.
    """
    path = Path(path)
    local = sha256_file(path)
    result = {"file": path.name, "sha256": local, "official_sha256": "", "status": "NOT_CHECKED"}
    if checksums_file is None or not Path(checksums_file).is_file():
        return result
    official = read_official_checksums(checksums_file).get(path.name)
    if official is None:
        result["status"] = "NOT_LISTED"
        return result
    result["official_sha256"] = official
    result["status"] = "MATCH" if official == local else "MISMATCH"
    return result


def label_matrix(
    codes: pd.Series, target_codes: list[str], min_likelihood: float
) -> pd.DataFrame:
    """Boolean record-by-code matrix: True when the code is listed with likelihood >= threshold.

    Codes are kept as separate columns; nothing is merged.
    """
    data = {
        code: [code in d and d[code] >= min_likelihood for d in codes] for code in target_codes
    }
    return pd.DataFrame(data, index=codes.index, dtype=bool)
