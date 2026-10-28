"""Check the frozen runtime artifacts against artifacts/runtime/manifest.json (size and SHA-256), and optionally load them.
Nothing is trained, fitted, calibrated or written. Exit code 0 = all checks pass, 1 = a problem was found.

    python tools/verify_runtime_artifacts.py            # hashes and sizes only (any Python 3)
    python tools/verify_runtime_artifacts.py --load     # also load every model in inference mode (needs the .venv)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/runtime/manifest.json"
CHECKSUMS = ROOT / "artifacts/runtime/checksums.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_files(man: dict) -> list[str]:
    problems = []
    listed = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    for a in man["artifacts"]:
        p = ROOT / a["path"]
        if listed.get(a["path"]) != a["sha256"]:
            problems.append(f"{a['path']}: checksums.sha256 and manifest.json disagree")
        if not p.is_file():
            problems.append(f"{a['path']}: MISSING")
            continue
        if p.stat().st_size != a["bytes"]:
            problems.append(f"{a['path']}: size {p.stat().st_size} != {a['bytes']}")
        elif sha256(p) != a["sha256"]:
            problems.append(f"{a['path']}: SHA-256 differs")
        else:
            print(f"OK   {a['sha256'][:12]}  {a['bytes']:>10,}  {a['path']}")
    return problems


def check_load() -> list[str]:
    """Load through the application's own path: the startup integrity check, the hash-checked gate loaders and the
    generator (eval mode). This is exactly what the backend does at start-up."""
    sys.path.insert(0, str(ROOT))
    from app.backend import integrity
    from app.backend.pipeline import demo
    try:
        integrity.verify()
        g = demo.gate_assets()
        gen = demo.generator()
    except Exception as exc:  # noqa: BLE001 - report any refusal as a failed check
        return [f"load failed: {type(exc).__name__}: {exc}"]
    print(f"LOAD gate: {type(g['model']).__name__} with {g['model'].n_features_in_} features; "
          f"calibrator {type(g['cal']).__name__}; threshold {g['t']!r}")
    print(f"LOAD reconstructor: {type(gen['gen']).__name__}, eval mode = {not gen['gen'].training}, device = {gen['device']}")
    return [] if not gen["gen"].training else ["the reconstructor is not in eval mode"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--load", action="store_true", help="also load every model (needs torch and scikit-learn)")
    a = ap.parse_args()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = check_files(man)
    if not problems and a.load:
        problems += check_load()
    if problems:
        print("FAIL:\n  " + "\n  ".join(problems))
        return 1
    print(f"PASS: {len(man['artifacts'])} runtime artifacts match the manifest" + (" and load for inference" if a.load else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
