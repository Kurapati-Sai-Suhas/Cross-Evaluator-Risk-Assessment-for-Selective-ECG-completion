"""Milestone 2b smoke test for the locked L2G baseline (docs/architecture_lock_v0_1.md, Section 6).

This is NOT training. It checks that the locked model can be built and run correctly before any
expensive training is started. Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.train.smoke_l2g

Steps (each PASS / FAIL; results in artifacts/reports/smoke_l2g_v0_1.json and .md):
  1 environment   - PyTorch version and GPU visibility
  2 parameters    - ECGen 24-input 7,980,076; ECGen 12-input 7,972,396; VTTD 14,184
  3 shapes        - (B, 24, 2048) -> (B, 12, 2048); VTTD (B, 24, 400) -> (B, 24)
  4 real_data     - 8 A_fit records through the existing pipeline; no fold 9/10 record is read
  5 losses        - SSIM loss, discriminator loss, staged generator loss, one backward pass each
  6 batch_256     - forward + backward at the paper's batch size 256 on the GPU (peak memory recorded)
  7 copy_through  - observed samples bit-exact after copy-through of real generator outputs
  8 tiny_overfit  - 8 records, fixed masks, ECGen only, locked SGD recipe: loss <= 0.5 x initial within 500 steps
Exit code: 0 = every step PASS, 1 = at least one step did not pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import missingness as mz
from src.data import preprocess as pp
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.data.ptbxl_waveforms import load_by_ecg_id, resolve_waveform_root
from src.models import l2g

N_RECORDS, OVERFIT_STEPS, OVERFIT_TARGET = 8, 500, 0.5


def _step(steps: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    steps.append({"step": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_smoke_batch(repo: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    """8 A_fit records (lowest ecg_ids, quality exclusions removed), preprocessed, with fixed evaluation masks."""
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    prep_cfg = io.load_config(repo / "configs" / "preprocessing.yaml")
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    manifest = load_frozen_manifest(audit_cfg, repo)
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    pool = manifest[(manifest["role"] == "A_fit") & ~manifest["ecg_id"].isin(excluded)].sort_values("ecg_id")
    chosen = pool.head(N_RECORDS)
    if set(chosen["role"]) != {"A_fit"} or not chosen["strat_fold"].between(1, 8).all():
        raise RuntimeError("smoke batch must contain A_fit records from folds 1-8 only")
    db = io.load_database(repo / audit_cfg["dataset"]["database_csv"])
    root = resolve_waveform_root(audit_cfg, repo)
    xs, masks = [], []
    for ecg_id in chosen["ecg_id"]:
        x = pp.preprocess(load_by_ecg_id(db, root, int(ecg_id), "hr"), 500, prep_cfg).x
        observed = mz.evaluation_combinations(int(ecg_id), protocol["seed"], k=1)[0]
        xs.append(x)
        masks.append(mz.signal_mask(observed, x.shape[1]))
    x = np.stack(xs)
    m = np.stack(masks)
    return x, mz.apply_mask(x, m), m, [int(e) for e in chosen["ecg_id"]], protocol


def run(repo: Path) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    torch.manual_seed(20261030)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    env = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": cuda,
           "device": torch.cuda.get_device_name(0) if cuda else "cpu",
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if cuda else None}
    _step(steps, "1_environment", cuda, env)

    gen24, gen12, vttd = l2g.ECGen(24), l2g.ECGen(12), l2g.VTTD()
    counts = {"ecgen_24_inputs": l2g.count_parameters(gen24), "ecgen_12_inputs": l2g.count_parameters(gen12),
              "vttd": l2g.count_parameters(vttd)}
    _step(steps, "2_parameters", counts == l2g.expected_parameter_counts(),
          {"measured": counts, "required": l2g.expected_parameter_counts()})
    del gen12

    gen24, vttd = gen24.to(device), vttd.to(device)
    with torch.no_grad():
        out = gen24.eval()(torch.rand(2, 24, 2048, device=device))
        prob = vttd.eval()(torch.rand(2, 24, 400, device=device))
    _step(steps, "3_shapes", tuple(out.shape) == (2, 12, 2048) and tuple(prob.shape) == (2, 24),
          {"generator": list(out.shape), "vttd": list(prob.shape)})

    x, x_obs, m, ecg_ids, protocol = load_smoke_batch(repo)
    ok = x.shape == (N_RECORDS, 12, 2048) and np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    _step(steps, "4_real_data", ok, {"ecg_ids": ecg_ids, "shape": list(x.shape), "role": "A_fit",
                                     "observed_leads": [int(mi[:, 0].sum()) for mi in m]})

    xt, xo, mt = (torch.from_numpy(a).to(device) for a in (x, x_obs, m))
    inp = l2g.model_input(xo, mt)
    ssim = l2g.SSIM1D().to(device)
    gen24.train()
    vttd.train()
    raw = gen24(inp)
    rec = l2g.reconstruction_loss(ssim, raw, xt)
    d_loss = l2g.discriminator_loss(vttd, l2g.minmax_per_lead(raw).detach(), xt)
    d_loss.backward()
    vttd.zero_grad()
    g_loss = l2g.generator_total_loss(ssim, vttd, raw, xt, epoch=10)
    g_loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in gen24.parameters())
    _step(steps, "5_losses", bool(torch.isfinite(rec) and torch.isfinite(d_loss) and torch.isfinite(g_loss) and grads_ok),
          {"one_minus_ssim": round(float(rec), 4), "vttd_bce": round(float(d_loss), 4),
           "generator_total": round(float(g_loss), 4), "finite_generator_grads": grads_ok})
    gen24.zero_grad()
    vttd.zero_grad()

    if cuda:
        rng = np.random.default_rng(1)
        combos = mz.l2g_lead_combinations()
        big_x = np.repeat(x, 256 // N_RECORDS, axis=0)
        big_m = np.stack([mz.signal_mask(mz.sample_training_combination(rng, combos), 2048) for _ in range(256)])
        bx, bm = torch.from_numpy(big_x).to(device), torch.from_numpy(big_m).to(device)
        bo = bx * bm
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            braw = gen24(l2g.model_input(bo, bm))
            l2g.discriminator_loss(vttd, l2g.minmax_per_lead(braw).detach(), bx).backward()
            vttd.zero_grad()
            l2g.generator_total_loss(ssim, vttd, braw, bx, epoch=10).backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            _step(steps, "6_batch_256", True, {"peak_memory_MiB": round(peak, 1),
                                               "seconds_forward_backward": round(time.perf_counter() - t0, 2)})
        except torch.cuda.OutOfMemoryError as exc:
            _step(steps, "6_batch_256", False, f"out of memory at batch 256: {exc}".splitlines()[0])
        finally:
            gen24.zero_grad()
            vttd.zero_grad()
            del bx, bm, bo
            torch.cuda.empty_cache()
    else:
        _step(steps, "6_batch_256", False, "no CUDA GPU visible; memory check not run")

    gen24.eval()
    with torch.no_grad():
        x_gen = l2g.minmax_per_lead(gen24(inp)).cpu().numpy()
    exact = all(np.array_equal(mz.copy_through(x_obs[i], x_gen[i], m[i])[m[i] == 1].view(np.uint32),
                               x[i][m[i] == 1].view(np.uint32)) for i in range(N_RECORDS))
    _step(steps, "7_copy_through", exact, "observed samples bit-exact for all 8 records" if exact else "MISMATCH")

    torch.manual_seed(20261030)
    fresh = l2g.ECGen(24).to(device).train()
    opt = torch.optim.SGD(fresh.parameters(), lr=0.01, momentum=0.5)  # locked recipe (PAPER lr, CODE momentum)
    curve, reached, t0 = [], None, time.perf_counter()
    for step in range(OVERFIT_STEPS + 1):
        loss = l2g.reconstruction_loss(ssim, fresh(inp), xt)
        if step % 25 == 0:
            curve.append([step, round(float(loss), 4)])
        if step == 0:
            initial = float(loss)
        if reached is None and float(loss) <= OVERFIT_TARGET * initial:
            reached = step
        if step == OVERFIT_STEPS:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    _step(steps, "8_tiny_overfit", reached is not None,
          {"initial_loss": round(initial, 4), "final_loss": round(float(loss), 4),
           "target": round(OVERFIT_TARGET * initial, 4), "step_target_reached": reached,
           "seconds": round(time.perf_counter() - t0, 1), "loss_every_25_steps": curve})

    lock = json.loads((repo / "artifacts" / "locks" / "protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture_lock": "docs/architecture_lock_v0_1.md (D-021)",
        "protocol_sha256": lock["sha256"],
        "environment": env,
        "roles_loaded": ["A_fit"],
        "ecg_ids": ecg_ids,
        "steps": steps,
        "status": "PASS" if all(s["status"] == "PASS" for s in steps) else "FAIL",
        "full_training_started": False,
        "note": "Smoke test only. The 8-record overfit check is a correctness test, not model training.",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# L2G baseline smoke test (Milestone 2b)", "",
             f"Generated by `src/train/smoke_l2g.py` at {report['created_utc']} (UTC). "
             f"**Overall: {report['status']}.** Full training started: {report['full_training_started']}.", "",
             f"Environment: `{report['environment']}`", "", "| Step | Status | Detail |", "| --- | --- | --- |"]
    lines += [f"| {s['step']} | {s['status']} | {json.dumps(s['detail'])} |" for s in report["steps"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milestone 2b smoke test (no training).")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    repo = Path(args.root).resolve()
    report = run(repo)
    out = repo / "artifacts" / "reports"
    (out / "smoke_l2g_v0_1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "smoke_l2g_v0_1.md").write_text(_markdown(report), encoding="utf-8")
    print(f"Smoke test overall: {report['status']}  (reports in {out})")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
