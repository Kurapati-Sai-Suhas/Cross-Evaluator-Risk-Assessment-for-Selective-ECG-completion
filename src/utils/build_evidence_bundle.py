"""Build the approval-packet evidence ZIP from a list of repository paths (files or folders).

Each file is stored under its repository-relative path; SHA256SUMS.txt inside the ZIP lists every file's SHA-256 as it was
on disk; NOT_BUNDLED.txt lists the referenced files that are deliberately left out (path, size, SHA-256, reason). The ZIP's
own SHA-256 is written beside it (<zip>.sha256).

    .\\.venv\\Scripts\\python.exe -m src.utils.build_evidence_bundle LIST.json OUT.zip
LIST.json: {"include": [paths or folders], "not_bundled": {"path": "reason", ...}}
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    spec, out = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    files = []
    for rel in spec["include"]:
        p = REPO / rel
        if not p.exists():
            raise SystemExit(f"missing: {rel}")
        files += sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    files = sorted(set(files))
    sums, skipped = [], []
    for rel, why in spec.get("not_bundled", {}).items():
        p = REPO / rel
        skipped.append(f"{rel}\t{p.stat().st_size if p.is_file() else 'folder'}\t{sha(p) if p.is_file() else '-'}\t{why}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            rel = p.relative_to(REPO).as_posix()
            z.write(p, rel)
            sums.append(f"{sha(p)}  {rel}")
        z.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
        z.writestr("NOT_BUNDLED.txt", "path\tbytes\tsha256\treason\n" + "\n".join(skipped) + "\n")
    digest = sha(out)
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"{len(files)} files, {out.stat().st_size:,} bytes, sha256 {digest}")


if __name__ == "__main__":
    main()
