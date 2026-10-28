"""Tests of the evidence-ZIP verifier (src/utils/verify_evidence_bundle.py) on a tiny synthetic repository and ZIP."""
from __future__ import annotations

import hashlib
import zipfile

from src.utils.verify_evidence_bundle import verify


def make(tmp, extra_entry=False):
    repo = tmp / "repo"
    (repo / "a").mkdir(parents=True)
    (repo / "a/x.txt").write_bytes(b"x\n")
    (repo / "big.bin").write_bytes(b"not bundled\n")
    h = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    zp = tmp / "bundle.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("a/x.txt", b"x\n")
        if extra_entry:
            z.writestr("stray.txt", b"?")
        z.writestr("SHA256SUMS.txt", f"{h(b'x' + bytes([10]))}  a/x.txt\n")
        z.writestr("NOT_BUNDLED.txt", f"path\tbytes\tsha256\treason\nbig.bin\t12\t{h(b'not bundled' + bytes([10]))}\tsize\n")
    (tmp / "bundle.zip.sha256").write_text(f"{hashlib.sha256(zp.read_bytes()).hexdigest()}  bundle.zip\n", encoding="utf-8")
    return repo, zp


def test_clean_bundle_passes(tmp_path):
    repo, zp = make(tmp_path)
    res = verify(zp, repo)
    assert res["all_ok"] and res["zip_entries"] == 3 and res["files_listed_in_SHA256SUMS"] == 1
    assert res["not_bundled_files_rechecked"] == 1 and res["sidecar_matches"]


def test_changed_repository_file_or_omitted_file_is_reported(tmp_path):
    repo, zp = make(tmp_path)
    (repo / "a/x.txt").write_bytes(b"y\n")
    (repo / "big.bin").write_bytes(b"changed\n")
    res = verify(zp, repo)
    assert not res["all_ok"] and res["zip_internally_consistent"] and not res["matches_current_repository"]
    assert res["repository_hash_mismatches"] == ["a/x.txt"] and res["not_bundled_hash_mismatches"] == ["big.bin"]
    assert res["in_zip_hash_mismatches"] == []


def test_unlisted_entry_or_wrong_sidecar_fails(tmp_path):
    repo, zp = make(tmp_path, extra_entry=True)
    res = verify(zp, repo)
    assert not res["all_ok"] and not res["zip_internally_consistent"] and "stray.txt" in res["entries_not_listed"]
    repo2, zp2 = make(tmp_path / "second")
    (tmp_path / "second/bundle.zip.sha256").write_text("0" * 64 + "  bundle.zip\n", encoding="utf-8")
    assert not verify(zp2, repo2)["sidecar_matches"]
