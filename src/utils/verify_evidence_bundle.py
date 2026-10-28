"""Re-check an evidence ZIP written by build_evidence_bundle: it opens, every entry listed in SHA256SUMS.txt has that SHA-256
inside the ZIP and in the repository, nothing else is in it except the two lists, the <zip>.sha256 file beside it matches,
and every file named in NOT_BUNDLED.txt still has its recorded SHA-256 in the repository.

    .\\.venv\\Scripts\\python.exe -m src.utils.verify_evidence_bundle ZIP OUT.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LISTS = {"SHA256SUMS.txt", "NOT_BUNDLED.txt"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(zip_path: Path, repo: Path = REPO) -> dict:
    digest = sha(zip_path)
    side = Path(str(zip_path) + ".sha256")
    with zipfile.ZipFile(zip_path) as z:
        bad_crc = z.testzip()
        names = z.namelist()
        sums = [line.split("  ", 1) for line in z.read("SHA256SUMS.txt").decode("utf-8").splitlines() if line]
        listed = {rel for _, rel in sums}
        in_zip_mismatch = [rel for h, rel in sums if rel not in names or hashlib.sha256(z.read(rel)).hexdigest() != h]
        nb_rows = [r.split("\t") for r in z.read("NOT_BUNDLED.txt").decode("utf-8").splitlines()[1:] if r]
    repo_mismatch = [rel for h, rel in sums if not (repo / rel).is_file() or sha(repo / rel) != h]
    nb_mismatch = [r[0] for r in nb_rows if r[2] != "-" and (not (repo / r[0]).is_file() or sha(repo / r[0]) != r[2])]
    res = {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zip": zip_path.name, "zip_bytes": zip_path.stat().st_size, "zip_sha256": digest,
        "sidecar_sha256_file": side.name if side.exists() else None,
        "sidecar_matches": side.exists() and side.read_text(encoding="utf-8").split()[0] == digest,
        "crc_ok": bad_crc is None, "zip_entries": len(names), "files_listed_in_SHA256SUMS": len(sums),
        "entries_not_listed": sorted(set(names) - listed),
        "in_zip_hash_mismatches": in_zip_mismatch, "repository_hash_mismatches": repo_mismatch,
        "not_bundled_rows": len(nb_rows), "not_bundled_files_rechecked": sum(r[2] != "-" for r in nb_rows),
        "not_bundled_hash_mismatches": nb_mismatch,
    }
    # the ZIP can be intact while the repository has moved on since it was built; the two are reported separately
    res["zip_internally_consistent"] = (res["crc_ok"] and res["sidecar_matches"] and not in_zip_mismatch
                                        and set(res["entries_not_listed"]) == LISTS)
    res["matches_current_repository"] = not repo_mismatch and not nb_mismatch
    res["all_ok"] = res["zip_internally_consistent"] and res["matches_current_repository"]
    return res


def main() -> None:
    zip_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    res = verify(zip_path)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
