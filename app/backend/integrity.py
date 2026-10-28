"""Startup asset-integrity check: the service refuses to start unless every frozen asset it relies on is unchanged."""
import hashlib
import json

from .pipeline import REPO

PINNED = {
    "artifacts/runs/l2g_v0_1_run02/best.pt": "d515875a7783be5a54934001e13b182ff4f75a5e35554f0e9590d5a1c602bb3b",
    "artifacts/gate/v0_2/m4b1/01_fit/models/gate_B.joblib": "7b863410bd34321adb6cec972a2264eae4e897e522358a54a3d4c2ebafac5a5d",
    "artifacts/gate/v0_2/m4b1/02_calibration/calibrator_gate_B.joblib":
        "341ba86a3e2b58c287524ad95ab57a2e535be694e4e9018f987e685c177b45ca",
    "artifacts/locks/threshold_lock_v0_2.json": "bc69c3011739a342b43a8b42b55840a3a8c25d9a5a7bc3ba9c9c3de52568d8de",
    "configs/protocol_v0_2.yaml": "38ad0a8669015b442bf80dac35438d7336c3da9bfca4f7804442b95fe5260773",
    "demo/infer_partial_ecg.py": "62d05bc78055c2f465ab608d54402d8a28cc3de6c4ec2010678df095692c4e57",   # stage 10 demo v2
}
THRESHOLD_REPR = "0.05927309439311807"


class IntegrityError(RuntimeError):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(repo=REPO, pinned=None) -> dict:
    """SHA-256 of every pinned asset, plus the lock's own record of the Gate B model, calibrator and threshold.
    Raises IntegrityError on any mismatch or missing file."""
    pinned = PINNED if pinned is None else pinned
    files, problems = {}, []
    for rel, expected in pinned.items():
        p = repo / rel
        actual = sha256(p) if p.is_file() else None
        files[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            problems.append(f"{rel}: {'missing' if actual is None else 'SHA-256 differs'}")
    lock_rel = "artifacts/locks/threshold_lock_v0_2.json"
    if lock_rel in pinned and files[lock_rel]["match"]:
        gb = json.loads((repo / lock_rel).read_text(encoding="utf-8"))["methods"]["gate_B"]
        if gb["t"]["repr"] != THRESHOLD_REPR:
            problems.append("threshold lock: Gate B threshold differs from the audited value")
        for key, rel in (("model_sha256", "artifacts/gate/v0_2/m4b1/01_fit/models/gate_B.joblib"),
                         ("calibrator_sha256", "artifacts/gate/v0_2/m4b1/02_calibration/calibrator_gate_B.joblib")):
            if rel in pinned and gb[key] != pinned[rel]:
                problems.append(f"threshold lock records a different {key}")
    report = {"ok": not problems, "files": files, "problems": problems, "threshold_repr": THRESHOLD_REPR}
    if problems:
        raise IntegrityError("; ".join(problems))
    return report
