"""Stage 05 preflight for the single fold-10 run (M4b-1). Opens no fold-10 record; writes one JSON report.

    .\\.venv\\Scripts\\python.exe -m src.eval.fold10_preflight DRY_RUN_DIR TEST_LOG CODE_TREE_JSON OUT.json

GO (to REQUEST the owner's final-run authorisation) only if every check passes: protocol and threshold locks verify; the
locked 9b calibrated scores are reproduced bit for bit; the dry run on 9b passed every check and reproduced every frozen
9b file; the code that the final run would execute is byte-identical to the code of that dry run; the latest full test run
passed on the current code tree; no fold-10 artifact or cache exists; the final output folder is absent; a GPU is available.
GO is never an authorisation: the final run still refuses without the owner's exact phrase (fold10_pathway.final_authorisation_ok).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data import ptbxl_io as io
from src.eval import fold10_pathway as f10
from src.gate import m4b1
from src.utils.provenance import code_tree_sha256


def preflight(repo: Path, dry_dir: Path, test_log: Path, tree_json: Path) -> dict[str, Any]:
    import torch
    ok = f10.Check()
    results: dict[str, Any] = {}
    try:
        tl = f10.verify_locks(repo, ok)
        f10.reproduce_locked_9b(repo, tl, ok)
        s = json.loads((dry_dir / "summary.json").read_text(encoding="utf-8"))
        ok("dry_run_mode_and_role", s["mode"] == "dry_run" and s["role"] == "9b_threshold", dry_dir.relative_to(repo).as_posix())
        ok("dry_run_checks_all_pass", all(c["status"] == "PASS" for c in s["checks"]), len(s["checks"]))
        comp = s["comparisons_with_frozen"]
        ok("dry_run_reproduced_frozen_9b_files", all(comp[k]["identical"] for k in ("keys", "offline_labels", "features", "M2", "calibrated_scores")),
           {k: comp[k]["identical"] for k in ("keys", "offline_labels", "features", "M2", "calibrated_scores")})
        ok("dry_run_locked_coverages_equal_lock", all(comp["locked_coverage_equals_lock"].values()), comp["locked_coverage_equals_lock"])
        ok("dry_run_full_replicates", s.get("replicates") == 10000, s.get("replicates"))
        man = json.loads((dry_dir / "hash_manifest.json").read_text(encoding="utf-8"))
        ok("dry_run_outputs_unchanged", all(io.sha256_file(repo / k) == v for k, v in man.items()), len(man))
        ok("dry_run_touched_no_fold10", s["fold10_paths_before"] == [] == s["fold10_paths_after"], "before and after empty")
        now = {rel: io.sha256_file(repo / rel) for rel in s["code_files_sha256"]}
        ok("final_code_equals_dry_run_code", now == s["code_files_sha256"], {k: v[:12] for k, v in now.items()})
        log = test_log.read_text(encoding="utf-8").splitlines()
        tree_now = code_tree_sha256(repo)["tree_sha256"]
        tree_log = json.loads(tree_json.read_text(encoding="utf-8"))["tree_sha256"]
        ok("latest_full_test_run_passed", log[-1] == "exit=0" and any(" passed" in l and "failed" not in l for l in log), [l for l in log if " passed" in l][-1:])
        ok("tests_ran_on_current_code_tree", tree_now == tree_log, tree_now)
        paths = f10.fold10_paths(repo, f10.CACHE_DIR)
        ok("no_fold10_artifact_or_cache", paths == [], paths)
        ok("final_output_folder_absent", not (repo / f10.OUT["final"]).exists(), f10.OUT["final"])
        ok("gpu_available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        auth, why = f10.final_authorisation_ok(repo)
        results["final_authorisation_present"] = auth
        results["final_authorisation_detail"] = why
        results["phrase_the_owner_must_send"] = f"AUTHORISE FINAL FOLD-10 RUN {io.sha256_file(repo / m4b1.THRESHOLD_LOCK)}"
        results["decision"] = "GO TO REQUEST FINAL-RUN AUTHORISATION" if not auth else "AUTHORISATION ALREADY RECORDED"
    except RuntimeError as e:
        results["decision"] = f"NO-GO: {e}"
    results.update({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": ok.items,
                    "fold10_opened": False})
    return results


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    repo = f10.REPO
    dry, log, tree, out = (repo / a for a in argv[:4])
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    res = preflight(repo, dry, log, tree)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(res["decision"])
    return 0 if res["decision"].startswith("GO") else 1


if __name__ == "__main__":
    sys.exit(main())
