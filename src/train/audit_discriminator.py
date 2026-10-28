"""M2d: discriminator (VTTD) gradient audit. Diagnostic only.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.train.audit_discriminator

The frozen reconstructor checkpoint is only READ (copied into memory); nothing is trained or saved except the
audit report and figure. Uses A_fit (training data) for gradient checks and a C_dev batch for held-out
discriminator accuracy (both roles permitted for development checks by protocol v0.1).

Checks
  C1 discriminator step: every VTTD layer receives a non-zero gradient, and one SGD step changes its weights
  C2 isolation: the discriminator loss sends NO gradient into the generator (x_gen is detached)
  C3 adversarial gradient: -alpha*BCE gives the generator a non-zero gradient, measured apart from SSIM
  C4 sign: a small generator step on the adversarial term alone RAISES the VTTD loss (the generator fools D)
  C5 conventions: shuffle labels mark exactly the generated channels; both orders occur
  C6 controlled comparison: 20 identical steps from the frozen G with alpha = 0.1 vs alpha = 0
  C7 capacity: can a fresh VTTD separate real from generated leads? (a) locked SGD lr 1e-3, momentum 0.5;
     (b) DIAGNOSTIC ONLY Adam lr 1e-3. Also the accuracy of the trained run02 VTTD on held-out C_dev rows.
Classification: A = no implementation bug (C1-C5 pass); B = likely bug (any of C1-C5 fails); C = inconclusive.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.data import missingness as mz
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.models import l2g
from src.train import recon_metrics as rm
from src.train.train_l2g import select_role
from src.utils.provenance import code_tree_sha256, environment

SEED = 20261030
G_SHA = "d515875a7783be5a54934001e13b182ff4f75a5e35554f0e9590d5a1c602bb3b"


def _flat(grads: list[torch.Tensor | None], params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def check_discriminator_step(gen, vttd, inp, x, seed: int) -> dict[str, Any]:
    """C1 + C2: gradients of the D loss on VTTD layers and on the generator; one SGD step on a copy."""
    vttd = copy.deepcopy(vttd).train()
    gen.zero_grad(set_to_none=True)
    x_gen = l2g.minmax_per_lead(gen(inp)).detach()
    loss = l2g.discriminator_loss(vttd, x_gen, x, generator=torch.Generator().manual_seed(seed))
    vttd.zero_grad(set_to_none=True)
    loss.backward()
    per_layer = {n: float(p.grad.norm()) if p.grad is not None else 0.0 for n, p in vttd.named_parameters()}
    before = [p.detach().clone() for p in vttd.parameters()]
    opt = torch.optim.SGD(vttd.parameters(), lr=1e-3, momentum=0.5)
    opt.step()
    change = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(vttd.parameters(), before))))
    gen_grads = [p.grad for p in gen.parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    return {"vttd_bce": float(loss), "vttd_grad_norm_per_layer": per_layer,
            "all_vttd_layers_nonzero": all(v > 0 for v in per_layer.values()), "vttd_param_change_l2": change,
            "generator_params_with_grad_from_d_loss": len(gen_grads),
            "C1_pass": all(v > 0 for v in per_layer.values()) and change > 0, "C2_pass": len(gen_grads) == 0}


def check_adversarial_gradient(gen, vttd, inp, x, seed: int, alpha: float = l2g.ALPHA,
                               adv_fn: Callable = l2g.discriminator_loss) -> dict[str, Any]:
    """C3: gradient of the SSIM term and of -alpha*BCE on the generator, computed separately on one forward pass."""
    params = [p for p in gen.parameters() if p.requires_grad]
    ssim = l2g.SSIM1D().to(x.device)
    raw = gen(inp)
    rec = l2g.reconstruction_loss(ssim, raw, x)
    adv = adv_fn(vttd, l2g.minmax_per_lead(raw), x, generator=torch.Generator().manual_seed(seed))
    g_rec = _flat(torch.autograd.grad(rec, params, retain_graph=True, allow_unused=True), params)
    if adv.requires_grad:
        g_adv = _flat(torch.autograd.grad(-alpha * adv, params, allow_unused=True), params)
    else:
        g_adv = torch.zeros_like(g_rec)
    n_rec, n_adv = float(g_rec.norm()), float(g_adv.norm())
    cos = float(F.cosine_similarity(g_rec, g_adv, dim=0)) if n_adv > 0 else float("nan")
    return {"ssim_loss": float(rec), "vttd_bce": float(adv), "grad_norm_ssim_term": n_rec,
            "grad_norm_adversarial_term": n_adv, "ratio_adv_to_ssim": n_adv / n_rec if n_rec else float("nan"),
            "cosine_ssim_vs_adv": cos, "C3_pass": n_adv > 0}


def check_adversarial_sign(gen, vttd, inp, x, seed: int, lr: float = 1e-3) -> dict[str, Any]:
    """C4: the generator objective must contain the VTTD loss with coefficient exactly -alpha, i.e. the generator
    MAXIMISES the discriminator's error (PAPER Stage 3). Checked algebraically on one forward pass with identical
    crop/shuffle draws. A small empirical step is reported as a descriptive number only."""
    gen = copy.deepcopy(gen).train()
    vttd = copy.deepcopy(vttd).eval()  # D fixed
    ssim = l2g.SSIM1D().to(x.device)
    raw = gen(inp)
    total = l2g.generator_total_loss(ssim, vttd, raw, x, epoch=l2g.ADVERSARIAL_START_EPOCH,
                                     generator=torch.Generator().manual_seed(seed))
    rec = l2g.reconstruction_loss(ssim, raw, x)
    bce = l2g.discriminator_loss(vttd, l2g.minmax_per_lead(raw), x, generator=torch.Generator().manual_seed(seed))
    coefficient = float((total - rec) / bce)
    opt = torch.optim.SGD(gen.parameters(), lr=lr)
    opt.zero_grad()
    (-l2g.ALPHA * bce).backward()
    opt.step()
    with torch.no_grad():
        after = float(l2g.discriminator_loss(vttd, l2g.minmax_per_lead(gen(inp)), x,
                                             generator=torch.Generator().manual_seed(seed)))
    return {"coefficient_on_vttd_bce": coefficient, "expected_coefficient": -l2g.ALPHA,
            "vttd_bce_before": float(bce), "vttd_bce_after_small_step_descriptive": after,
            "C4_pass": abs(coefficient + l2g.ALPHA) < 1e-4}


def check_training_loop_source() -> dict[str, Any]:
    """C4b: static inspection of the training loop that produced run02 (src/train/train_l2g.py)."""
    import inspect

    from src.train import train_l2g
    src = inspect.getsource(train_l2g.train)
    needles = {
        "discriminator_step_uses_detached_generator_output": "l2g.minmax_per_lead(raw).detach(), x, generator=torch_gen",
        "generator_step_uses_non_detached_output": "l2g.discriminator_loss(vttd, l2g.minmax_per_lead(raw), x, generator=torch_gen)",
        "generator_total_is_rec_minus_alpha_adv": "g_total = rec - alpha * adv",
        "d_grads_zeroed_before_d_step": "opt_d.zero_grad()",
    }
    found = {k: v in src for k, v in needles.items()}
    return {**found, "C4b_pass": all(found.values())}


def check_shuffle_convention(seed: int) -> dict[str, Any]:
    """C5: labels = 1 exactly where the generated lead was placed."""
    g, r = torch.rand(16, 12, 40), torch.rand(16, 12, 40) + 2.0  # disjoint ranges make the source identifiable
    d, labels = l2g.channel_shuffle(g, r, generator=torch.Generator().manual_seed(seed))
    is_generated = (d.amax(dim=-1) < 1.5).float()
    return {"labels_match_generated_channels": bool(torch.equal(is_generated, labels)),
            "share_generated_first": float(labels.reshape(16, 12, 2)[..., 0].mean()),
            "C5_pass": bool(torch.equal(is_generated, labels)) and 0 < float(labels.reshape(16, 12, 2)[..., 0].mean()) < 1}


def pair_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Per lead pair: correct if the higher output is on the generated channel. Chance = 0.5."""
    p, lab = probs.reshape(-1, 12, 2), labels.reshape(-1, 12, 2)
    return float((p.argmax(-1) == lab.argmax(-1)).float().mean())


def _batch(signals, idx, device, rng, combo_masks):
    sig = torch.from_numpy(signals[idx]).to(device)
    x = rm.gpu_crop_normalize(sig, torch.zeros(len(idx), dtype=torch.long, device=device))
    lm = combo_masks[torch.from_numpy(rng.integers(0, len(combo_masks), size=len(idx))).to(device)]
    m = lm.unsqueeze(-1).expand_as(x)
    return torch.cat([x * m, m], dim=1), x


def run(repo: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    ckpt = repo / "artifacts" / "runs" / "l2g_v0_1_run02" / "best.pt"
    if io.sha256_file(ckpt) != G_SHA:
        raise RuntimeError("frozen G v0.1 checkpoint hash mismatch; refusing to audit")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    gen = l2g.ECGen(24).to(device)
    gen.load_state_dict(state["generator"])
    vttd_trained = l2g.VTTD().to(device)
    vttd_trained.load_state_dict(state["vttd"])

    manifest = load_frozen_manifest(audit_cfg, repo)
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    a_rows = select_role(manifest, "A_fit", excluded)
    c_rows = select_role(manifest, "C_dev", excluded)
    cache = Path("C:/ecg_cache/v0_1")
    a_sig = np.load(cache / "A_fit_signals.npy", mmap_mode="r")
    c_sig = np.load(cache / "C_dev_signals.npy", mmap_mode="r")
    if len(a_sig) != len(a_rows) or len(c_sig) != len(c_rows):
        raise RuntimeError("cache does not match the frozen roles")
    combo_masks = torch.tensor(np.stack([mz.lead_mask(c) for c in mz.l2g_lead_combinations()]),
                               dtype=torch.float32, device=device)

    inp, x = _batch(a_sig, np.sort(rng.choice(len(a_sig), 64, replace=False)), device, rng, combo_masks)
    gen.eval()  # pristine frozen G: every check below works on a deep copy (BatchNorm statistics untouched)
    vttd_trained.eval()

    def g_copy():
        return copy.deepcopy(gen).train()

    results: dict[str, Any] = {}
    results["C1_C2"] = check_discriminator_step(g_copy(), vttd_trained, inp, x, SEED)
    results["C3"] = check_adversarial_gradient(g_copy(), copy.deepcopy(vttd_trained).train(), inp, x, SEED)
    results["C4"] = check_adversarial_sign(g_copy(), vttd_trained, inp, x, SEED)
    results["C4b"] = check_training_loop_source()
    results["C5"] = check_shuffle_convention(SEED)

    # C6: 20 identical steps from the frozen G, alpha = 0.1 vs alpha = 0 (D updated identically in both)
    eval_inp, eval_x = _batch(c_sig, np.arange(64), device, np.random.default_rng(SEED + 1), combo_masks)
    ssim = l2g.SSIM1D().to(device)
    traj: dict[str, list[float]] = {}
    finals = {}
    for alpha in (0.1, 0.0):
        g, d = g_copy(), copy.deepcopy(vttd_trained).train()
        og, od = torch.optim.SGD(g.parameters(), lr=0.01, momentum=0.5), torch.optim.SGD(d.parameters(), lr=1e-3, momentum=0.5)
        brng, tg = np.random.default_rng(SEED + 2), torch.Generator().manual_seed(SEED + 3)
        start = [p.detach().clone() for p in g.parameters()]
        for _ in range(20):
            b_inp, b_x = _batch(a_sig, np.sort(brng.choice(len(a_sig), 64, replace=False)), device, brng, combo_masks)
            raw = g(b_inp)
            dl = l2g.discriminator_loss(d, l2g.minmax_per_lead(raw).detach(), b_x, generator=tg)
            od.zero_grad()
            dl.backward()
            od.step()
            loss = l2g.reconstruction_loss(ssim, raw, b_x) - alpha * l2g.discriminator_loss(d, l2g.minmax_per_lead(raw), b_x, generator=tg)
            og.zero_grad()
            loss.backward()
            og.step()
        g.eval()
        with torch.no_grad():
            traj[f"alpha_{alpha}"] = float(l2g.reconstruction_loss(ssim, g(eval_inp), eval_x))
        finals[alpha] = [p.detach().clone() for p in g.parameters()]
        finals[f"start_{alpha}"] = start
    diff = float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(finals[0.1], finals[0.0]))))
    moved = float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(finals[0.0], finals["start_0.0"]))))
    results["C6"] = {"heldout_one_minus_ssim_after_20_steps": traj, "param_diff_alpha01_vs_alpha0_l2": diff,
                     "param_change_alpha0_l2": moved, "relative_effect_of_adversarial_term": diff / moved if moved else float("nan")}

    # C7: discriminator capacity
    held_inp, held_x = _batch(c_sig, np.arange(64, 320), device, np.random.default_rng(SEED + 4), combo_masks)
    gen.eval()
    with torch.no_grad():
        held_gen = l2g.minmax_per_lead(gen(held_inp))

    def evaluate_d(d) -> dict[str, float]:
        d.eval()
        with torch.no_grad():
            gc, rc = l2g.random_crop(held_gen, held_x, generator=torch.Generator().manual_seed(SEED + 5))
            dd, lab = l2g.channel_shuffle(gc, rc, generator=torch.Generator().manual_seed(SEED + 6))
            probs = d(dd)
        d.train()
        return {"bce": float(F.binary_cross_entropy(probs, lab)), "pair_accuracy": pair_accuracy(probs, lab)}

    results["C7_trained_run02_vttd_heldout"] = evaluate_d(copy.deepcopy(vttd_trained))
    curves: dict[str, list[list[float]]] = {}
    for name, make_opt in (("locked_sgd_lr1e-3_m0.5", lambda p: torch.optim.SGD(p, lr=1e-3, momentum=0.5)),
                           ("diagnostic_adam_lr1e-3", lambda p: torch.optim.Adam(p, lr=1e-3))):
        torch.manual_seed(SEED + 7)
        d = l2g.VTTD().to(device).train()
        opt = make_opt(d.parameters())
        crng, tg = np.random.default_rng(SEED + 8), torch.Generator().manual_seed(SEED + 9)
        curve = []
        for step in range(400):
            b_inp, b_x = _batch(a_sig, np.sort(crng.choice(len(a_sig), 64, replace=False)), device, crng, combo_masks)
            with torch.no_grad():
                b_gen = l2g.minmax_per_lead(gen(b_inp))
            loss = l2g.discriminator_loss(d, b_gen, b_x, generator=tg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == 399:
                curve.append([step, float(loss), evaluate_d(d)["pair_accuracy"]])
        curves[name] = curve
        results[f"C7_fresh_{name}"] = {"final_train_bce": curve[-1][1], **{f"heldout_{k}": v for k, v in evaluate_d(d).items()}}

    passes = {k: results[s][k] for s, k in (("C1_C2", "C1_pass"), ("C1_C2", "C2_pass"), ("C3", "C3_pass"),
                                              ("C4", "C4_pass"), ("C4b", "C4b_pass"), ("C5", "C5_pass"))}
    classification = "A" if all(passes.values()) else "B"
    report = {
        "milestone": "M2d", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classification": classification,
        "classification_meaning": {"A": "no implementation bug found; adversarial benefit remains unproven",
                                   "B": "likely implementation bug", "C": "inconclusive"}[classification],
        "check_pass": passes, "results": results, "c7_curves": curves,
        "inputs": {"frozen_G_checkpoint": str(ckpt.relative_to(repo)), "frozen_G_sha256": G_SHA,
                   "protocol_sha256": io.sha256_file(repo / "configs" / "protocol_v0_1.yaml"),
                   "frozen_manifest_sha256": audit_cfg["split"]["frozen_manifest_sha256"],
                   "roles_used": ["A_fit (gradient checks, C7 training batches)", "C_dev (held-out evaluation batches)"]},
        "frozen_checkpoint_unchanged_after_audit": io.sha256_file(ckpt) == G_SHA,
        "code_tree_sha256": code_tree_sha256(repo)["tree_sha256"], "environment": environment(),
        "seconds": round(time.perf_counter() - t0, 1), "seed": SEED,
    }
    _plot(repo, curves, results)
    return report


def _plot(repo: Path, curves: dict[str, list[list[float]]], results: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, c in curves.items():
        c = np.array(c)
        ax[0].plot(c[:, 0], c[:, 1], label=name)
        ax[1].plot(c[:, 0], c[:, 2], label=name)
    ax[0].axhline(np.log(2), color="grey", ls="--", lw=0.8, label="chance BCE = ln 2")
    ax[1].axhline(0.5, color="grey", ls="--", lw=0.8, label="chance accuracy")
    ax[1].axhline(results["C7_trained_run02_vttd_heldout"]["pair_accuracy"], color="red", ls=":", lw=0.8,
                  label="trained run02 VTTD (held-out)")
    ax[0].set_title("fresh VTTD training BCE (A_fit batches)")
    ax[1].set_title("held-out pair accuracy (C_dev batch)")
    for a in ax:
        a.set_xlabel("step")
        a.legend(fontsize=7)
    fig.tight_layout()
    out = repo / "artifacts" / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_m2d_vttd_capacity.png", dpi=130)
    plt.close(fig)


def main() -> int:
    repo = Path(".").resolve()
    report = run(repo)
    path = repo / "artifacts" / "reports" / "m2d_discriminator_audit.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"classification": report["classification"], "check_pass": report["check_pass"],
                      "C3": {k: report["results"]["C3"][k] for k in ("grad_norm_ssim_term", "grad_norm_adversarial_term",
                                                                     "ratio_adv_to_ssim", "cosine_ssim_vs_adv")},
                      "C4": report["results"]["C4"], "C6": report["results"]["C6"],
                      "C7": {k: v for k, v in report["results"].items() if k.startswith("C7")},
                      "checkpoint_unchanged": report["frozen_checkpoint_unchanged_after_audit"]}, indent=1, default=str))
    return 0 if report["frozen_checkpoint_unchanged_after_audit"] else 1


if __name__ == "__main__":
    sys.exit(main())
