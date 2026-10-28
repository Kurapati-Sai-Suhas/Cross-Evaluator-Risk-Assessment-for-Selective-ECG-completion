"""Repository SHA-256 snapshots and before/after diffs (evidence for milestone reports).

``snapshot``: SHA-256 of every file under the repository except .venv and caches; the PTB-XL waveform folder can be included
or hashed separately because it holds about 87,000 files. ``diff``: added / removed / changed paths between two snapshots,
written as JSON (machine-readable) and Markdown (human-readable, with a reason per path taken from a reasons file).

    .\\.venv\\Scripts\\python.exe -m src.utils.repo_hash_snapshot snapshot OUT.json [--with-ptbxl | --only-ptbxl]
    .\\.venv\\Scripts\\python.exe -m src.utils.repo_hash_snapshot diff BEFORE.json AFTER.json OUT_STEM [--reasons R.json]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKIP = {".venv", "__pycache__", ".pytest_cache"}


def load(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["files"] if "files" in data else data


def snapshot(out: Path, mode: str = "no-ptbxl") -> None:
    files = {}
    base = REPO / "PTB-XL" if mode == "only-ptbxl" else REPO
    for p in base.rglob("*"):
        rel = p.relative_to(REPO).as_posix()
        if not p.is_file() or SKIP & set(p.parts) or (mode == "no-ptbxl" and rel.startswith("PTB-XL/")):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        files[rel] = h.hexdigest()
    out.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode,
                               "files": files}, indent=1), encoding="utf-8")
    print(len(files), "files ->", out)


def diff(before: Path, after: Path, stem: Path, reasons: Path | None = None) -> None:
    a, b = load(before), load(after)
    a = {k: v for k, v in a.items() if not k.startswith("PTB-XL/")} if not any(k.startswith("PTB-XL/") for k in b) else a
    why = json.loads(reasons.read_text(encoding="utf-8")) if reasons else {}

    def reason(path: str) -> str:
        if path in why:
            return why[path]
        hits = [v for k, v in why.items() if any(ch in k for ch in "*?[") and fnmatch.fnmatch(path, k)]
        return hits[0] if hits else "NO REASON RECORDED"

    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "before": {"file": before.name, "sha256": hashlib.sha256(before.read_bytes()).hexdigest(), "files": len(a)},
           "after": {"file": after.name, "sha256": hashlib.sha256(after.read_bytes()).hexdigest(), "files": len(b)},
           "unchanged": len(set(a) & set(b)) - len(changed),
           "added": [{"path": k, "sha256": b[k], "reason": reason(k)} for k in added],
           "removed": [{"path": k, "sha256": a[k], "reason": reason(k)} for k in removed],
           "changed": [{"path": k, "before": a[k], "after": b[k], "reason": reason(k)} for k in changed]}
    stem.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [f"# Hash diff: {before.name} → {after.name}", "",
             f"Files before {len(a)}, after {len(b)}; unchanged {res['unchanged']}; added {len(added)}; removed {len(removed)}; "
             f"changed {len(changed)}.", "", "| Change | Path | SHA-256 (after) | Why |", "| --- | --- | --- | --- |"]
    for kind, items in (("added", res["added"]), ("changed", res["changed"]), ("removed", res["removed"])):
        for x in items:
            sha = x.get("sha256") or x.get("after")
            lines.append(f"| {kind} | `{x['path']}` | `{sha[:16]}…` | {x['reason']} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"unchanged {res['unchanged']} added {len(added)} removed {len(removed)} changed {len(changed)}; "
          f"without reason: {sum(x['reason'] == 'NO REASON RECORDED' for x in res['added'] + res['changed'] + res['removed'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("out", type=Path)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--with-ptbxl", action="store_true")
    g.add_argument("--only-ptbxl", action="store_true")
    d = sub.add_parser("diff")
    d.add_argument("before", type=Path)
    d.add_argument("after", type=Path)
    d.add_argument("stem", type=Path)
    d.add_argument("--reasons", type=Path)
    a = ap.parse_args()
    if a.cmd == "snapshot":
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; refusing to overwrite")
        snapshot(a.out, "with-ptbxl" if a.with_ptbxl else "only-ptbxl" if a.only_ptbxl else "no-ptbxl")
    else:
        diff(a.before, a.after, a.stem, a.reasons)


if __name__ == "__main__":
    main()
