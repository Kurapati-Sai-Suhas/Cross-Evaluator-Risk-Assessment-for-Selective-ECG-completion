"""Guards of the fold-10 pathway (src/eval/fold10_pathway.py). Synthetic inputs and file-existence checks only: no test
opens a fold-10 record."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eval import fold10_pathway as f10
from tests.conftest import REPO_ROOT


def test_modes_refuse_fold10_without_final_authorisation():
    f10.check_mode("dry_run", "9b_threshold", final_authorised=False)
    for mode, role, auth in [("dry_run", "test_fold10", True), ("dry_run", "B_gate", False), ("final", "test_fold10", False),
                             ("final", "9b_threshold", True), ("evaluate", "test_fold10", True)]:
        with pytest.raises(RuntimeError):
            f10.check_mode(mode, role, auth)
    f10.check_mode("final", "test_fold10", final_authorised=True)        # the only accepted final combination


def test_row_guard_matches_role_and_fold():
    ok9b = pd.DataFrame({"role": ["9b_threshold"] * 2, "strat_fold": [9, 9]})
    f10.check_rows(ok9b, "9b_threshold")
    with pytest.raises(RuntimeError):
        f10.check_rows(ok9b.assign(strat_fold=[9, 10]), "9b_threshold")   # a fold-10 record in a dry-run role
    with pytest.raises(RuntimeError):
        f10.check_rows(ok9b.assign(role="test_fold10"), "test_fold10")    # test_fold10 rows must all be fold 10
    with pytest.raises(RuntimeError):
        f10.check_rows(ok9b.iloc[:0], "9b_threshold")


def test_fold10_path_scan(tmp_path):
    (tmp_path / "artifacts").mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    assert f10.fold10_paths(tmp_path, cache) == []
    (cache / "test_fold10_signals.npy").write_bytes(b"")
    assert f10.fold10_paths(tmp_path, cache) == [(cache / "test_fold10_signals.npy").as_posix()]


def test_csv_hash_is_deterministic():
    df = pd.DataFrame({"row_id": ["a", "b"], "x": [0.1, 1 / 3]})
    assert f10.csv_sha(df, "%.17g") == f10.csv_sha(df.copy(), "%.17g") != f10.csv_sha(df.assign(x=[0.1, 0.3]), "%.17g")


def test_final_authorisation_needs_the_exact_phrase_and_current_hashes(tmp_path):
    import json
    from src.data import ptbxl_io as io
    (tmp_path / "artifacts/locks").mkdir(parents=True)
    (tmp_path / "artifacts/authorisations").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts/locks/threshold_lock_v0_2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "configs/protocol_v0_2.yaml").write_text("x: 1\n", encoding="utf-8")
    tl, pr = io.sha256_file(tmp_path / "artifacts/locks/threshold_lock_v0_2.json"), io.sha256_file(tmp_path / "configs/protocol_v0_2.yaml")
    auth = tmp_path / f10.FINAL_AUTHORISATION
    assert f10.final_authorisation_ok(tmp_path)[0] is False                         # no file
    for statement, tl_rec in [("AUTHORISE FINAL FOLD-10 RUN " + "0" * 64, tl),    # wrong hash in the owner's words
                              ("I authorise the final run", tl),                    # phrase missing
                              (f"AUTHORISE FINAL FOLD-10 RUN {tl}", "0" * 64)]:     # recorded hash wrong
        auth.write_text(json.dumps({"owner_statement": statement, "threshold_lock_sha256": tl_rec, "protocol_v0_2_sha256": pr}), encoding="utf-8")
        assert f10.final_authorisation_ok(tmp_path)[0] is False
    auth.write_text(json.dumps({"owner_statement": f"AUTHORISE FINAL FOLD-10 RUN {tl}", "threshold_lock_sha256": tl,
                                "protocol_v0_2_sha256": pr}), encoding="utf-8")
    assert f10.final_authorisation_ok(tmp_path)[0] is True


def test_real_repository_final_run_is_refused_and_fold10_untouched():
    assert not (REPO_ROOT / f10.FINAL_AUTHORISATION).exists()
    with pytest.raises(RuntimeError):
        f10.run(REPO_ROOT, "final", "test_fold10")
    assert f10.fold10_paths(REPO_ROOT, f10.CACHE_DIR) == []
