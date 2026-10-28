"""Versioned request schema for one partial ECG, validated before anything is computed.

A request is a JSON object with exactly these keys (any other key, e.g. a complete ECG, labels or D1, is refused):
    schema_version    "partial-ecg-csv/1"
    sampling_rate_hz  500 (the only validated rate)
    units             "mV"
    observed_leads    the observed lead names; must equal the CSV header (this is the whole-lead mask)
    csv               CSV text: one column per observed lead, header names from I II III AVR AVL AVF V1-V6
    source_name       optional; the uploaded file name, which must end in .csv
After these checks the stage 10 demo's own checks apply unchanged (demo/infer_partial_ecg.py): the lead set must be one of
the 258 L2G sets, at least 4,096 samples (longer input is cut to 5,000), finite values, below 50 mV; a PTB-XL file name
from fold 10 is refused.
"""
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .pipeline import LEAD_NAMES, demo

SCHEMA_VERSION = "partial-ecg-csv/1"
REQUIRED = ("schema_version", "sampling_rate_hz", "units", "observed_leads", "csv")
OPTIONAL = ("source_name",)
MAX_CSV_CHARS = 3_000_000
MAX_ROWS = 60_000


class InputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ParsedInput:
    x: np.ndarray                     # (12, n) mV at 500 Hz; missing leads are zero
    observed: list
    samples_in: int
    warnings: list = field(default_factory=list)
    source_name: str | None = None


def _demo_check(fn, *args, code="INVALID_INPUT"):
    """The demo refuses with SystemExit; the service turns that into an InputError (nothing is released)."""
    try:
        return fn(*args)
    except SystemExit as exc:
        raise InputError(code, str(exc)) from None


def check_rectangular(text: str) -> None:
    """Every row must have exactly as many fields as the header. Without this, pandas silently uses an extra first column
    as the index and shifts every lead by one column (found in stage 14; the demo's own loader does not check it)."""
    rows = csv.reader(io.StringIO(text))
    header = next(rows, None)
    if not header or not any(h.strip() for h in header):
        raise InputError("INVALID_CSV", "the CSV has no header row")
    for i, row in enumerate(rows, start=2):
        if row and len(row) != len(header):
            raise InputError("INVALID_CSV", f"row {i} has {len(row)} fields but the header has {len(header)}")


def parse_request(payload) -> ParsedInput:
    if not isinstance(payload, dict):
        raise InputError("INVALID_REQUEST", "the request must be a JSON object")
    unknown = sorted(set(payload) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise InputError("FORBIDDEN_OR_UNKNOWN_FIELD",
                         f"unknown fields {unknown}: only observed leads are accepted; a complete ECG, labels, D1 or any "
                         "other offline data must never be sent to the gate")
    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        raise InputError("INVALID_REQUEST", f"missing fields {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise InputError("INVALID_REQUEST", f"schema_version must be {SCHEMA_VERSION!r}")
    fs = payload["sampling_rate_hz"]
    if isinstance(fs, bool) or not isinstance(fs, (int, float)) or fs != 500:
        raise InputError("UNSUPPORTED_SAMPLING_RATE", "only 500 Hz input is supported (the validated PTB-XL rate)")
    if payload["units"] != "mV":
        raise InputError("UNSUPPORTED_UNITS", "only millivolt (mV) input is supported")
    obs = payload["observed_leads"]
    if (not isinstance(obs, list) or not obs or not all(isinstance(o, str) for o in obs)
            or len(set(obs)) != len(obs) or any(o not in LEAD_NAMES for o in obs)):
        raise InputError("INVALID_MASK", f"observed_leads must be distinct lead names from {LEAD_NAMES}")
    src = payload.get("source_name")
    if src is not None:
        if not isinstance(src, str) or len(src) > 255 or not src.lower().endswith(".csv"):
            raise InputError("UNSUPPORTED_FILE_TYPE", "only .csv files are supported")
        _demo_check(demo.refuse_fold10, Path(src), code="FOLD10_REFUSED")
    text = payload["csv"]
    if not isinstance(text, str) or not text.strip():
        raise InputError("INVALID_CSV", "csv must be non-empty text")
    if len(text) > MAX_CSV_CHARS or text.count("\n") > MAX_ROWS + 1:
        raise InputError("INPUT_TOO_LARGE", f"at most {MAX_CSV_CHARS:,} characters and {MAX_ROWS:,} rows")
    if "\x00" in text:
        raise InputError("UNSUPPORTED_FILE_TYPE", "binary content is not a CSV file")
    check_rectangular(text)
    try:
        x, header_leads, n_in = demo.load_csv(io.StringIO(text))
    except SystemExit as exc:
        raise InputError("INVALID_CSV", str(exc).split(":", 1)[-1].strip() if "cannot read CSV" in str(exc) else str(exc)) from None
    if set(header_leads) != set(obs):
        raise InputError("INVALID_MASK", f"observed_leads {obs} do not match the CSV header {header_leads}")
    x, warnings = _demo_check(demo.prepare, x, header_leads, n_in)
    return ParsedInput(x=x, observed=header_leads, samples_in=int(n_in), warnings=warnings, source_name=src)
