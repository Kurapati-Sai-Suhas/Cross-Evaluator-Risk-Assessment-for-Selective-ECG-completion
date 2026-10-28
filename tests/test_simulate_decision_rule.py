"""Tests of the archived-simulation runner (src/eval/simulate_decision_rule.py): seeds do not depend on sharding, a resumed
shard file equals an uninterrupted one, and tampered or truncated files are refused. Synthetic data only, small B."""
from __future__ import annotations

import csv

import pytest

from src.eval import simulate_decision_rule as sdr

NAME = "H2_null_exchangeable_tied_scores"


def run(tmp, sims=3, B=200, shard=0, shards=1, resume=False):
    sdr.run_scenario(NAME, sims, B, 60, 20261021, tmp, shard, shards, resume)
    return tmp / f"{NAME}_raw_shard{shard:02d}of{shards:02d}.csv", tmp / f"{NAME}_meta_shard{shard:02d}of{shards:02d}.json"


def test_seeds_do_not_depend_on_sharding():
    seeds = sdr.sim_seeds(NAME, 10, 20261021)
    assert seeds == sdr.sim_seeds(NAME, 10, 20261021) and len(set(seeds)) == 10
    assert sdr.sim_seeds(NAME, 5, 20261021) == seeds[:5]                 # prefix-stable


def test_resumed_file_equals_uninterrupted_run(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    raw_a, _ = run(a)
    full = raw_a.read_bytes()
    raw_b = b / raw_a.name
    raw_b.write_bytes(b"".join(full.splitlines(keepends=True)[:2]))     # header + first row, as after an interruption
    raw_b2, meta_b = run(b, resume=True)
    assert raw_b2.read_bytes() == full and '"resumed_after_rows": 1' in meta_b.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):                                       # finished shard: never rewritten
        run(b, resume=True)


def test_resume_refuses_truncated_or_foreign_rows(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    raw, meta = run(a)
    meta.unlink()
    text = raw.read_text(encoding="utf-8")
    raw.write_text(text[:-5], encoding="utf-8")                           # last row cut off
    with pytest.raises(SystemExit):
        run(a, resume=True)
    rows = list(csv.DictReader(text.splitlines()))
    rows[0]["replicate_seed"] = "1"                                       # tampered seed
    with raw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sdr.FIELDS)
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit):
        run(a, resume=True)
    with pytest.raises(SystemExit):                                       # existing file without --resume
        run(a)
