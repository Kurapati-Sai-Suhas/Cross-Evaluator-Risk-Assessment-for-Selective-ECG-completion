"""Consistency tests for configs/protocol_v0_1.yaml (decision D-020).

These do not test any model (none exists yet). They stop the machine-readable protocol from
drifting away from the frozen split, the approved target (D-010) and the data-quality exclusions.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.data import ptbxl_io as io
from tests.conftest import REPO_ROOT


@pytest.fixture(scope="module")
def protocol() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "protocol_v0_1.yaml")


def test_status_is_declared(protocol):
    assert protocol["status"] in {"PROPOSED", "APPROVED"}
    assert protocol["version"] == "0.1"


def test_approved_protocol_is_pinned_and_unchanged(protocol):
    """Once APPROVED, the YAML must match the SHA-256 in its read-only lock file (D-020)."""
    if protocol["status"] != "APPROVED":
        pytest.skip("protocol not approved yet")
    lock_path = REPO_ROOT / "artifacts" / "locks" / "protocol_v0_1.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    yaml_path = REPO_ROOT / lock["file"]
    assert io.sha256_file(yaml_path) == lock["sha256"], "protocol_v0_1.yaml was edited after approval"
    assert lock["status"] == "APPROVED"
    import os
    assert not os.access(yaml_path, os.W_OK) and not os.access(lock_path, os.W_OK)


def test_targets_match_approved_decision_and_exclusions(protocol):
    audit_cfg = io.load_config(REPO_ROOT / "configs" / "data_audit.yaml")
    prep_cfg = io.load_config(REPO_ROOT / "configs" / "preprocessing.yaml")
    assert protocol["targets"]["primary_outputs"] == audit_cfg["diagnostic_target"]["primary_outputs"]
    assert set(protocol["targets"]["excluded_ecg_ids"]) == {e["ecg_id"] for e in prep_cfg["quality_exclusions"]}
    assert protocol["seed"] == audit_cfg["split"]["seed"] == prep_cfg["masks"]["seed"]


def test_every_frozen_role_has_exactly_one_purpose(protocol):
    meta = json.loads((REPO_ROOT / "artifacts" / "manifests" / "split_manifest_v1_frozen_metadata.json")
                      .read_text(encoding="utf-8"))
    frozen_roles = {r["role"] for r in meta["role_counts"]}
    assigned = [role for roles in protocol["roles"].values() for role in roles]
    assert sorted(assigned) == sorted(frozen_roles)  # all six covered, none used twice
    assert "A_fit" not in protocol["d1"]["roles"]  # D1 rows must be out-of-sample for G and E1


def test_threshold_rule_is_fully_specified(protocol):
    th = protocol["threshold"]
    grid = np.round(np.arange(th["grid_start"], th["grid_stop"] + th["grid_step"] / 2, th["grid_step"]), 6)
    assert len(grid) == 1001 and grid[-1] == 1.0  # t = 1.0 always gives coverage 1 -> always feasible
    assert 0 < th["min_coverage"] < 1
    assert th["tie_break"] == ["lowest_risk", "highest_coverage", "smallest_t"]
    assert th["selected_on"] == protocol["roles"]["threshold_selection"][0]


def test_forbidden_gate_inputs_cover_the_hard_rule(protocol):
    forbidden = set(protocol["gate"]["forbidden_inputs"])
    required = {"complete_reference_ecg", "true_reconstruction_error", "d1", "labels",
                "e1_outputs_complete", "e2_outputs_complete", "vttd_score"}
    assert required <= forbidden


def test_masks_and_bootstrap_settings_are_valid(protocol):
    assert 1 <= protocol["d1"]["masks_per_record"] <= 258
    assert protocol["uncertainty"]["unit"] == "patient"
    assert protocol["uncertainty"]["replicates"] >= 1000
