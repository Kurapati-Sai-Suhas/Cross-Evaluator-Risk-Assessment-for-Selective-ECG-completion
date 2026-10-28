"""Recompute the SHA-256 of every frozen artifact and lock and compare with the values recorded at freeze time.

    .\\.venv\\Scripts\\python.exe -m src.utils.check_frozen_artifacts OUT.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FROZEN = {
    "artifacts/manifests/split_manifest_v1_frozen.csv": "898d0ac707cf86ec9201eb4d3a9d97a49e8b84767374313b01d1a5996f42b83b",
    "configs/protocol_v0_1.yaml": "55a1ceb6ad4a2078f5b856752bd8c8dd9954ea2168e547760eb6529b8ba32b43",
    "artifacts/locks/protocol_v0_1.lock.json": "a2f4e12d9c6690a5709d0ee86ac3e8d81af377804da52da602292c029089fb2e",
    "artifacts/runs/l2g_v0_1_run02/best.pt": "d515875a7783be5a54934001e13b182ff4f75a5e35554f0e9590d5a1c602bb3b",
    "artifacts/runs/e1_resnet1d_v0_1_run01/attempt_2/best.pt": "e10c250c78782fe23449fa9dc39dfb7fc981baac00eb690511444386a4357c1f",
    "artifacts/runs/e1_resnet1d_v0_1_run01/summary.json": "805469cf1aca8b57701b6160087e905a7a530979039c1dd8f0c179f9faffeff1",
    "artifacts/runs/edir_resnet1d_v0_1_run02/attempt_1/best.pt": "4b94af6ac6afe21be10fe589cf4f59831399fb17ca6fb9822c51e847844fcbdd",
    "artifacts/d1/v0_1/hash_manifest.json": "001db6d1545ef529ebae7d98ce270bd8cbf97e7f35ab2c2d14b9fc2aa9a8668e",
    "artifacts/gate/v0_2/hash_manifest.json": "f9371c72b22008df403f0ebc7ed2e93498d74d7f554dc653a367f029fc6a0afd",
    # pinned 2026-10-24 after the owner's approval (D-037)
    "configs/protocol_v0_2.yaml": "38ad0a8669015b442bf80dac35438d7336c3da9bfca4f7804442b95fe5260773",
    "configs/gate_m4b_v0_2.yaml": "80fdd7a4c7aec27005eeb84c93b47228923b86d2867ce1da540b12e0aa264690",
    "artifacts/locks/protocol_v0_2.lock.json": "6562b8553e9297bffea4bb6c3aaf4d4214467f818aaf3cf6da280e52f2224a01",
    # written in M4b-1 stage 03 (2026-10-24, owner's M4b-1 authorisation)
    "artifacts/locks/threshold_lock_v0_2.json": "bc69c3011739a342b43a8b42b55840a3a8c25d9a5a7bc3ba9c9c3de52568d8de",
}
LOCKS = ["protocol_v0_1.lock.json", "protocol_v0_2.lock.json", "threshold_lock_v0_2.json"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    rows = [{"path": p, "expected": e, "actual": sha(REPO / p), "read_only": not os.access(REPO / p, os.W_OK)} for p, e in FROZEN.items()]
    for manifest in ("artifacts/d1/v0_1/hash_manifest.json", "artifacts/gate/v0_2/hash_manifest.json"):
        for rel, e in json.loads((REPO / manifest).read_text(encoding="utf-8"))["files"].items():
            rows.append({"path": rel, "expected": e, "actual": sha(REPO / rel), "listed_in": manifest})
    locks = sorted(p.name for p in (REPO / "artifacts/locks").iterdir())
    ok = all(r["expected"] == r["actual"] for r in rows) and locks == LOCKS
    out.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "all_match": ok,
                               "locks_folder": locks, "checked": len(rows), "rows": rows}, indent=2), encoding="utf-8")
    print("all match:", ok, "| files checked:", len(rows), "| locks:", locks)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
