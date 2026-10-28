"""Input helpers for the frontend (pure functions, unit-tested). Only observed leads are ever loaded or sent."""
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SCHEMA_VERSION = "partial-ecg-csv/1"
FS = 500
# Public release: the PTB-XL record-17 development fixtures are not distributed (no patient recordings in the repository).
# The built-in examples are the generated synthetic signals from examples/synthetic/ (make_synthetic_ecg.py); each file
# holds only its observed leads. Recorded outcomes: cand01 ACCEPT (released), cand04 ABSTAIN (screening_log.json).
FIXTURES = {
    "Synthetic example cand01: 6 observed leads (generated; NOT an ECG recording)": REPO / "examples/synthetic/synthetic_cand01.csv",
    "Synthetic example cand04: lead V2 only (generated; NOT an ECG recording)": REPO / "examples/synthetic/synthetic_cand04.csv",
}
SYNTHETIC = "Synthetic signal (generated here; NOT an ECG recording)"
UPLOAD = "Upload a CSV file"
MAX_UPLOAD_BYTES = 3_000_000


class InputProblem(ValueError):
    pass


def check_upload_name(name: str) -> None:
    if not name or not name.lower().endswith(".csv"):
        raise InputProblem("only .csv files are supported (one column per observed lead, 500 Hz, mV)")


def read_csv_text(raw: bytes) -> str:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InputProblem(f"the file is larger than {MAX_UPLOAD_BYTES:,} bytes")
    if b"\x00" in raw:
        raise InputProblem("the file is binary, not a CSV text file")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputProblem("the file is not UTF-8 text") from None


def frame_from_text(text: str) -> pd.DataFrame:
    rows = csv.reader(io.StringIO(text))           # same rule as the backend: pandas would silently shift ragged rows
    header = next(rows, None)
    if not header:
        raise InputProblem("the CSV has no header row")
    for i, row in enumerate(rows, start=2):
        if row and len(row) != len(header):
            raise InputProblem(f"row {i} has {len(row)} fields but the header has {len(header)}")
    try:
        df = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise InputProblem(f"cannot read the CSV: {exc}") from None
    df.columns = [str(c).strip().upper() for c in df.columns]
    unknown = [c for c in df.columns if c not in LEAD_NAMES]
    if unknown or df.columns.duplicated().any() or df.empty:
        raise InputProblem(f"the header must be distinct lead names from {LEAD_NAMES}; got {list(df.columns)}")
    return df


def load_fixture(label: str) -> tuple[pd.DataFrame, str]:
    text = FIXTURES[label].read_text(encoding="utf-8")
    return frame_from_text(text), text


def synthetic_frame(leads=("I", "III", "V1", "V2", "V3", "V5"), n=5000, seed=0) -> pd.DataFrame:
    """A smooth synthetic waveform for exercising the workflow only. It is not an ECG; the gate's output on it means nothing."""
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    beat = lambda tt: (np.exp(-((tt % 0.85) - 0.30) ** 2 / 2e-4) * 1.1 - np.exp(-((tt % 0.85) - 0.33) ** 2 / 3e-4) * 0.3  # noqa: E731
                       + np.exp(-((tt % 0.85) - 0.55) ** 2 / 2e-3) * 0.25 + np.exp(-((tt % 0.85) - 0.12) ** 2 / 1e-3) * 0.1)
    return pd.DataFrame({lead: (0.6 + 0.1 * k) * beat(t + 0.01 * k) + 0.02 * rng.standard_normal(n)
                         for k, lead in enumerate(leads)})


def build_payload(df: pd.DataFrame, observed: list[str], source_name: str | None = None) -> dict:
    """The request holds only the selected observed leads; the other columns are dropped before anything is sent."""
    if not observed:
        raise InputProblem("select at least one observed lead")
    missing = [o for o in observed if o not in df.columns]
    if missing:
        raise InputProblem(f"leads {missing} are not in the input")
    ordered = [l for l in LEAD_NAMES if l in observed]
    payload = {"schema_version": SCHEMA_VERSION, "sampling_rate_hz": FS, "units": "mV", "observed_leads": ordered,
               "csv": df[ordered].to_csv(index=False)}       # full precision: values reach the gate exactly as parsed
    if source_name:
        payload["source_name"] = source_name
    return payload


def is_supported_set(observed, lead_sets) -> bool:
    return frozenset(observed) in {frozenset(s) for s in lead_sets}
