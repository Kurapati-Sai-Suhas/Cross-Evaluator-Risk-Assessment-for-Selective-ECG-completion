"""Append rows to artifacts/reports/evidence_ledger.csv (docs/milestone_reporting_standard.md).

Every row must have exactly the standard columns; missing values must be written explicitly as
PENDING / NOT MEASURED / NOT RECORDED / N/A rather than left empty.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

COLUMNS = ["milestone_id", "date", "objective", "commit", "protocol_hash", "manifest_hash", "config_hash", "seed",
           "split", "input_artifacts", "command", "output_artifacts", "metric", "estimate", "uncertainty_method",
           "figure_id", "table_id", "claim_supported", "deviations", "failures"]


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        missing = [c for c in COLUMNS if c not in row or row[c] in (None, "")]
        extra = [c for c in row if c not in COLUMNS]
        if missing or extra:
            raise ValueError(f"ledger row has missing {missing} / unknown {extra} columns: {row.get('metric')}")
    new_file = not path.is_file()
    if not new_file:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        if header != COLUMNS:
            raise ValueError(f"{path} has unexpected columns {header}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
