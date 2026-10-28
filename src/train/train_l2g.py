"""Milestone 2c: controlled L2G reconstruction training (architecture lock v0.1, protocol v0.1).

Run from the repository root AFTER `python -m src.train.preflight` has passed:
    .\\.venv\\Scripts\\python.exe -m src.train.train_l2g --config configs/train_l2g_v0_1.yaml

Data use (protocol Section 3): fits on A_fit only (minus quality exclusions); C_dev only for early stopping and
best-so-far checkpoint selection; B_gate, 9a, 9b and fold 10 are never loaded (asserted).
Outputs go to artifacts/runs/<run_name>/ (refuses to overwrite): train_log.jsonl, history.csv, checkpoints with
SHA-256, config snapshot, validation_rows_best.csv, figures/, run_summary.json.
Stops on a non-finite loss or an out-of-memory error and records why. Exit code 0 = finished (see adequacy gate
in run_summary.json), 1 = stopped by a failure.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data import missingness as mz
from src.data import preprocess as pp
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.data.ptbxl_waveforms import resolve_waveform_root
from src.eval.stats import bootstrap_or_undefined
from src.models import l2g
from src.train import cache as cachemod
from src.train import recon_metrics as rm
from src.utils.provenance import code_tree_sha256, environment

METRICS = ["ssim_missing", "mse_missing", "pcc_missing", "ssim_full", "mse_full"]


class TrainingStopped(RuntimeError):
    """Raised for a stop condition (non-finite loss). The reason is recorded in run_summary.json."""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def select_role(manifest: pd.DataFrame, role: str, excluded: set[int]) -> pd.DataFrame:
    rows = manifest[(manifest["role"] == role) & ~manifest["ecg_id"].isin(excluded)].sort_values("ecg_id")
    if not rows["strat_fold"].between(1, 8).all():
        raise RuntimeError(f"role {role} contains records outside folds 1-8; refusing to train")
    return rows.reset_index(drop=True)


def validation_lead_masks(ecg_ids: list[int], seed: int, k: int) -> np.ndarray:
    """(N, k, 12) uint8 masks from the protocol's deterministic evaluation_combinations."""
    return np.stack([np.stack([mz.lead_mask(c) for c in mz.evaluation_combinations(e, seed, k=k)]) for e in ecg_ids])


def windows_at(signals: np.ndarray, prep_cfg: dict[str, Any], offset: int) -> np.ndarray:
    """Exact numpy windowing + min-max (preprocess.crop_normalize) for every cached record."""
    return np.stack([pp.crop_normalize(s, prep_cfg, offset).x for s in signals]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(gen: l2g.ECGen, ssim: l2g.SSIM1D, val_x: np.ndarray, val_lm: np.ndarray, device: torch.device,
             corr: torch.Tensor | None = None, rows_per_batch: int = 256) -> dict[str, np.ndarray]:
    """Per-row metrics on C_dev (rows = records x evaluation masks). Baselines B0/B1 too if ``corr`` is given."""
    gen.eval()
    n, k = val_lm.shape[:2]
    per = max(1, rows_per_batch // k)
    out: dict[str, list[np.ndarray]] = {}

    def add(name: str, value: torch.Tensor) -> None:
        out.setdefault(name, []).append(value.detach().cpu().numpy())

    for s in range(0, n, per):
        x = torch.from_numpy(val_x[s:s + per]).to(device).repeat_interleave(k, dim=0)
        lm = torch.from_numpy(val_lm[s:s + per]).to(device).float().reshape(-1, 12)
        m = lm.unsqueeze(-1).expand_as(x)
        x_obs = x * m
        x_gen = l2g.minmax_per_lead(gen(torch.cat([x_obs, m], dim=1)))
        add("selection_loss", 1.0 - ssim.per_lead(x_gen, x).mean(dim=-1))  # training-loss quantity, all leads
        x_hat = torch.where(m.bool(), x_obs, x_gen)  # exact copy-through
        for name, value in rm.row_metrics(ssim, x_hat, x, lm).items():
            add(f"l2g_{name}", value)
        add("n_observed", lm.sum(dim=-1))
        if corr is not None:
            add("l2g_lead_ssim", ssim.per_lead(x_hat, x))
            add("lead_mask", lm)
            for tag, filled in (("b0", rm.baseline_constant(x_obs, lm)),
                                ("b1", rm.baseline_best_correlated(x_obs, lm, corr))):
                for name, value in rm.row_metrics(ssim, filled, x, lm).items():
                    add(f"{tag}_{name}", value)
                if tag == "b1":
                    add("b1_lead_ssim", ssim.per_lead(filled, x))
    gen.train()
    return {name: np.concatenate(parts) for name, parts in out.items()}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: dict[str, Any], repo: Path, config_path: Path) -> dict[str, Any]:
    t_start = time.perf_counter()
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    prep_path = repo / "configs" / "preprocessing.yaml"
    prep_cfg = io.load_config(prep_path)
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    lock = json.loads((repo / "artifacts" / "locks" / "protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    if io.sha256_file(repo / "configs" / "protocol_v0_1.yaml") != lock["sha256"]:
        raise RuntimeError("protocol_v0_1.yaml does not match its lock; refusing to train")
    seed, tr, dcfg = int(cfg["seed"]), cfg["training"], cfg["data"]

    run_dir = repo / cfg["output_dir"] / cfg["run_name"]
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; use a new run_name (runs are never overwritten)")
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "config_snapshot").mkdir()
    for p in (config_path, prep_path, repo / "configs" / "protocol_v0_1.yaml", repo / "configs" / "data_audit.yaml"):
        shutil.copy(p, run_dir / "config_snapshot" / p.name)

    manifest = load_frozen_manifest(audit_cfg, repo)  # verifies the pinned SHA-256
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    train_rows = select_role(manifest, dcfg["train_role"], excluded)
    val_rows = select_role(manifest, dcfg["validation_role"], excluded)
    if set(train_rows["patient_id"]) & set(val_rows["patient_id"]):
        raise RuntimeError("train/validation patient overlap; refusing to train")
    roles_loaded = sorted({dcfg["train_role"], dcfg["validation_role"]})

    db = io.load_database(repo / audit_cfg["dataset"]["database_csv"])
    root = resolve_waveform_root(audit_cfg, repo)
    cache_dir = Path(dcfg["cache_dir"])
    manifest_sha = audit_cfg["split"]["frozen_manifest_sha256"]
    caches = {}
    t_cache = time.perf_counter()
    for role, rows in ((dcfg["train_role"], train_rows), (dcfg["validation_role"], val_rows)):
        key = cachemod.cache_key(prep_path, repo / audit_cfg["dataset"]["database_csv"], manifest_sha, role,
                                 rows["ecg_id"].tolist())
        caches[role] = cachemod.ensure_cache(role, rows, root, db, prep_cfg, key, cache_dir)
        print(f"cache {role}: {caches[role][0].shape} reused={caches[role][2]['reused']}", flush=True)
    cache_seconds = time.perf_counter() - t_cache
    train_sig = caches[dcfg["train_role"]][0]
    val_sig = caches[dcfg["validation_role"]][0]

    val_x = windows_at(val_sig, prep_cfg, int(dcfg["validation_window_offset"]))
    val_ids = val_rows["ecg_id"].tolist()
    k = int(dcfg["validation_masks_per_record"])
    val_lm = validation_lead_masks(val_ids, seed, k)
    afit_windows = windows_at(train_sig, prep_cfg, 0)
    corr_np = rm.lead_correlation_matrix(afit_windows)
    del afit_windows
    np.save(run_dir / "afit_lead_correlation.npy", corr_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen_model = l2g.ECGen(int(cfg["model"]["in_channels"])).to(device)
    vttd = l2g.VTTD().to(device)
    ssim = l2g.SSIM1D().to(device)
    ocfg = cfg["optimizer"]
    opt_g = torch.optim.SGD(gen_model.parameters(), lr=ocfg["generator"]["lr"], momentum=ocfg["generator"]["momentum"],
                            weight_decay=ocfg["generator"]["weight_decay"])
    opt_d = torch.optim.SGD(vttd.parameters(), lr=ocfg["discriminator"]["lr"],
                            momentum=ocfg["discriminator"]["momentum"], weight_decay=ocfg["discriminator"]["weight_decay"])
    sched_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, patience=ocfg["discriminator_scheduler"]["patience"])
    torch_gen = torch.Generator().manual_seed(seed)  # crops and channel shuffles
    combos = mz.l2g_lead_combinations()
    combo_masks = torch.tensor(np.stack([mz.lead_mask(c) for c in combos]), dtype=torch.float32, device=device)

    offsets = np.asarray(dcfg["train_window_offsets"], dtype=np.int64)
    rec_index = np.repeat(np.arange(len(train_rows)), len(offsets))
    win_offset = np.tile(offsets, len(train_rows))
    n_samples, bs = len(rec_index), int(tr["batch_size"])
    adv_start, alpha = int(tr["adversarial_start_epoch"]), float(tr["alpha"])

    history: list[dict[str, Any]] = []
    best = {"selection_loss": float("inf"), "epoch": None}
    patience_left = int(tr["early_stopping_patience"])
    stop_reason, status = "max_epochs_reached", "FINISHED"
    log = (run_dir / "train_log.jsonl").open("w", encoding="utf-8")

    def save_ckpt(path: Path, epoch: int, metrics: dict[str, float]) -> str:
        torch.save({"generator": gen_model.state_dict(), "vttd": vttd.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "sched_d": sched_d.state_dict(), "epoch": epoch,
                    "validation": metrics, "run_name": cfg["run_name"], "protocol_sha256": lock["sha256"],
                    "frozen_manifest_sha256": manifest_sha}, path)
        return io.sha256_file(path)

    try:
        for epoch in range(int(tr["max_epochs"])):
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rng = np.random.default_rng([seed, epoch])
            order = rng.permutation(n_samples)
            combo_idx = torch.from_numpy(rng.integers(0, len(combos), size=n_samples)).to(device)
            sums = {"rec": 0.0, "d": 0.0, "adv": 0.0, "g_total": 0.0}
            seen = 0
            gen_model.train()
            vttd.train()
            for start in range(0, n_samples, bs):
                sel = order[start:start + bs]
                sig = torch.from_numpy(train_sig[rec_index[sel]]).to(device, non_blocking=True)
                x = rm.gpu_crop_normalize(sig, torch.from_numpy(win_offset[sel]).to(device))
                lm = combo_masks[combo_idx[torch.from_numpy(sel).to(device)]]
                m = lm.unsqueeze(-1).expand_as(x)
                x_obs = x * m
                raw = gen_model(torch.cat([x_obs, m], dim=1))
                rec = l2g.reconstruction_loss(ssim, raw, x)
                if epoch >= adv_start:
                    d_loss = l2g.discriminator_loss(vttd, l2g.minmax_per_lead(raw).detach(), x, generator=torch_gen)
                    opt_d.zero_grad()
                    d_loss.backward()
                    opt_d.step()
                    adv = l2g.discriminator_loss(vttd, l2g.minmax_per_lead(raw), x, generator=torch_gen)
                    g_total = rec - alpha * adv
                    sums["d"] += float(d_loss) * len(sel)
                    sums["adv"] += float(adv) * len(sel)
                else:
                    g_total = rec
                if not torch.isfinite(g_total):
                    raise TrainingStopped(f"non-finite generator loss at epoch {epoch}, batch start {start}")
                opt_g.zero_grad()
                g_total.backward()
                opt_g.step()
                sums["rec"] += float(rec) * len(sel)
                sums["g_total"] += float(g_total) * len(sel)
                seen += len(sel)
            if epoch >= adv_start:
                sched_d.step(sums["d"] / seen)
            train_seconds = time.perf_counter() - t0
            val = evaluate(gen_model, ssim, val_x, val_lm, device)
            row = {
                "epoch": epoch, "phase": "adversarial" if epoch >= adv_start else "ssim_only",
                "train_one_minus_ssim": sums["rec"] / seen, "train_generator_total": sums["g_total"] / seen,
                "train_vttd_bce": sums["d"] / seen if epoch >= adv_start else None,
                "train_generator_adv_bce": sums["adv"] / seen if epoch >= adv_start else None,
                "val_selection_loss": float(val["selection_loss"].mean()),
                **{f"val_{name}": float(np.nanmean(val[f"l2g_{name}"])) for name in METRICS},
                "lr_generator": opt_g.param_groups[0]["lr"], "lr_vttd": opt_d.param_groups[0]["lr"],
                "train_seconds": round(train_seconds, 2), "epoch_seconds": round(time.perf_counter() - t0, 2),
                "peak_gpu_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device.type == "cuda" else None,
            }
            for key in ("train_one_minus_ssim", "val_selection_loss"):
                if not np.isfinite(row[key]):
                    raise TrainingStopped(f"non-finite {key} at epoch {epoch}")
            improved = row["val_selection_loss"] < best["selection_loss"]
            if improved:
                best = {"selection_loss": row["val_selection_loss"], "epoch": epoch,
                        "sha256": save_ckpt(run_dir / "best.pt", epoch, row)}
                patience_left = int(tr["early_stopping_patience"])
            else:
                patience_left -= 1
            row["improved"] = improved
            row["best_sha256"] = best["sha256"] if improved else None  # logged so a crash cannot lose it
            row["last_sha256"] = save_ckpt(run_dir / "last.pt", epoch, row)
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"epoch {epoch:2d} [{row['phase']}] train 1-SSIM {row['train_one_minus_ssim']:.4f} | "
                  f"val sel {row['val_selection_loss']:.4f} | val missing SSIM {row['val_ssim_missing']:.4f} "
                  f"MSE {row['val_mse_missing']:.5f} | {row['epoch_seconds']:.0f}s | "
                  f"{'*best*' if improved else f'patience {patience_left}'}", flush=True)
            if patience_left <= 0:
                stop_reason = f"early stopping: no improvement for {tr['early_stopping_patience']} epochs"
                break
    except TrainingStopped as exc:
        status, stop_reason = "FAILED_NON_FINITE", str(exc)
    except torch.cuda.OutOfMemoryError as exc:
        status, stop_reason = "FAILED_OOM", str(exc).splitlines()[0]
    finally:
        log.close()

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    summary: dict[str, Any] = {
        "run_name": cfg["run_name"], "status": status, "stop_reason": stop_reason,
        "epochs_run": len(history), "best_epoch": best["epoch"],
        "best_val_selection_loss": best["selection_loss"] if best["epoch"] is not None else None,
        "roles_loaded": roles_loaded, "train_records": len(train_rows),
        "train_patients": int(train_rows["patient_id"].nunique()), "train_samples_per_epoch": n_samples,
        "val_records": len(val_rows), "val_patients": int(val_rows["patient_id"].nunique()), "val_rows": int(len(val_rows) * k),
        "excluded_ecg_ids": sorted(excluded), "cache_seconds": round(cache_seconds, 1),
        "cache_meta": {r: c[2] for r, c in caches.items()},
        "hashes": {"training_config": io.sha256_file(config_path), "protocol": lock["sha256"],
                   "frozen_manifest": manifest_sha, "code_tree": code_tree_sha256(repo)["tree_sha256"]},
        "environment": environment(), "full_training": True,
        "downstream_components_trained": False, "fold10_loaded": False,
    }
    if status == "FINISHED" and best["epoch"] is not None:
        summary.update(final_evaluation(run_dir, best, gen_model, ssim, val_x, val_lm, val_rows, k, corr_np, device, cfg))
    summary["timing"] = {"total_seconds": round(time.perf_counter() - t_start, 1),
                         "mean_epoch_seconds": round(float(np.mean([h["epoch_seconds"] for h in history])), 1) if history else None,
                         "max_peak_gpu_MiB": max((h["peak_gpu_MiB"] or 0) for h in history) if history else None}
    make_plots(run_dir, history, summary, adv_start)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# Final evaluation of the selected checkpoint (C_dev only)
# --------------------------------------------------------------------------- #
def final_evaluation(run_dir: Path, best: dict[str, Any], gen_model: l2g.ECGen, ssim: l2g.SSIM1D, val_x: np.ndarray,
                     val_lm: np.ndarray, val_rows: pd.DataFrame, k: int, corr_np: np.ndarray,
                     device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    ckpt_path = run_dir / "best.pt"
    file_sha = io.sha256_file(ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)  # our own file, hash-checked first
    gen_model.load_state_dict(state["generator"])
    corr = torch.tensor(corr_np, dtype=torch.float32, device=device)
    res = evaluate(gen_model, ssim, val_x, val_lm, device, corr=corr)
    reloaded = float(res["selection_loss"].mean())
    logged_sha = best.get("sha256")
    integrity = {"best_checkpoint": str(ckpt_path.relative_to(run_dir.parents[2])),
                 "sha256_at_save": logged_sha or "NOT LOGGED (kept only in memory by the run02 code version)",
                 "sha256_now": file_sha, "checkpoint_epoch": int(state["epoch"]), "logged_best_epoch": best["epoch"],
                 "logged_val_selection_loss": best["selection_loss"], "reloaded_val_selection_loss": reloaded,
                 "pass": (logged_sha is None or file_sha == logged_sha) and int(state["epoch"]) == best["epoch"]
                 and abs(reloaded - best["selection_loss"]) < 1e-4}

    ecg = np.repeat(val_rows["ecg_id"].to_numpy(), k)
    pat = np.repeat(val_rows["patient_id"].to_numpy(), k)
    rows = pd.DataFrame({"ecg_id": ecg, "patient_id": pat, "mask_index": np.tile(np.arange(k), len(val_rows)),
                         "n_observed": res["n_observed"].astype(int)})
    for tag in ("l2g", "b0", "b1"):
        for name in METRICS:
            rows[f"{tag}_{name}"] = res[f"{tag}_{name}"]
    rows.to_csv(run_dir / "validation_rows_best.csv", index=False)

    ev = cfg["evaluation"]["bootstrap"]
    boot = lambda v: bootstrap_or_undefined(v, pat, replicates=ev["replicates"], seed=cfg["seed"], level=ev["level"])  # noqa: E731
    estimates = {f"{tag}_{name}": boot(rows[f"{tag}_{name}"].to_numpy())
                 for tag in ("l2g", "b0", "b1") for name in METRICS}
    diffs = {}
    for name in ("ssim_missing", "mse_missing", "pcc_missing"):
        for base in ("b0", "b1"):
            d = rows[f"l2g_{name}"].to_numpy() - rows[f"{base}_{name}"].to_numpy()
            if np.isfinite(d).any():
                diffs[f"l2g_minus_{base}_{name}"] = boot(d)
    gate = {
        "G1_integrity": integrity["pass"],
        "G2_missing_ssim_beats_baselines": all(diffs[f"l2g_minus_{b}_ssim_missing"]["ci_low"] > 0 for b in ("b0", "b1")),
        "G3_missing_mse_beats_baselines": all(diffs[f"l2g_minus_{b}_mse_missing"]["ci_high"] < 0 for b in ("b0", "b1")),
        "G4_no_leakage": True,  # select_role() and the overlap check raise before training otherwise
    }
    by_nobs = (rows.groupby("n_observed")
               .agg(rows_=("ecg_id", "size"), patients=("patient_id", "nunique"),
                    l2g_ssim_missing=("l2g_ssim_missing", "mean"), b1_ssim_missing=("b1_ssim_missing", "mean"),
                    b0_ssim_missing=("b0_ssim_missing", "mean"), l2g_mse_missing=("l2g_mse_missing", "mean"),
                    b1_mse_missing=("b1_mse_missing", "mean"))
               .reset_index().rename(columns={"rows_": "rows"}))
    missing = res["lead_mask"] == 0
    by_lead = pd.DataFrame({
        "lead": mz.LEAD_NAMES,
        "rows_missing": missing.sum(axis=0),
        "l2g_ssim": [float(res["l2g_lead_ssim"][missing[:, j], j].mean()) for j in range(12)],
        "b1_ssim": [float(res["b1_lead_ssim"][missing[:, j], j].mean()) for j in range(12)],
    })
    by_nobs.to_csv(run_dir / "missing_metrics_by_n_observed.csv", index=False)
    by_lead.to_csv(run_dir / "missing_ssim_by_lead.csv", index=False)
    example_plot(run_dir, gen_model, val_x, val_lm, val_rows, device)
    return {"checkpoint_integrity": integrity, "validation_estimates": estimates, "paired_differences": diffs,
            "adequacy_gate": gate, "adequate_for_downstream": all(gate.values()),
            "by_n_observed": by_nobs.to_dict(orient="records"), "by_lead": by_lead.to_dict(orient="records"),
            "a_fit_lead_correlation_file": "afit_lead_correlation.npy"}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_plots(run_dir: Path, history: list[dict[str, Any]], summary: dict[str, Any], adv_start: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not history:
        return
    h = pd.DataFrame(history)
    for col in ("train_vttd_bce", "train_generator_adv_bce"):
        h[col] = pd.to_numeric(h[col], errors="coerce")  # None before the adversarial phase -> NaN
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(h.epoch, h.train_one_minus_ssim, label="train 1-SSIM (A_fit)")
    ax[0].plot(h.epoch, h.val_selection_loss, label="validation selection loss (C_dev)")
    ax[1].plot(h.epoch, h.val_ssim_missing, label="missing-lead SSIM (C_dev)")
    ax[1].plot(h.epoch, h.val_ssim_full, label="full-signal SSIM after copy-through (C_dev)")
    ax[2].plot(h.epoch, h.val_mse_missing, label="missing-lead MSE (C_dev)")
    if h.train_vttd_bce.notna().any():
        twin = ax[0].twinx()
        twin.plot(h.epoch, h.train_vttd_bce, "k:", label="VTTD BCE")
        twin.set_ylabel("VTTD BCE")
    for a in ax:
        a.axvline(adv_start - 0.5, color="grey", ls="--", lw=0.8)
        if summary.get("best_epoch") is not None:
            a.axvline(summary["best_epoch"], color="red", ls=":", lw=0.8)
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    fig.suptitle(f"{summary['run_name']}: grey dashed = adversarial phase starts, red dotted = selected checkpoint")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "fig_training_curves.png", dpi=130)
    plt.close(fig)
    if "by_n_observed" in summary:
        b = pd.DataFrame(summary["by_n_observed"])
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.27
        ax.bar(b.n_observed - w, b.l2g_ssim_missing, w, label="L2G (ours, v0.1)")
        ax.bar(b.n_observed, b.b1_ssim_missing, w, label="B1 best-correlated lead")
        ax.bar(b.n_observed + w, b.b0_ssim_missing, w, label="B0 constant 0.5")
        ax.set_xlabel("number of observed leads")
        ax.set_ylabel("missing-lead SSIM (C_dev)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / "fig_missing_ssim_by_n_observed.png", dpi=130)
        plt.close(fig)


@torch.no_grad()
def example_plot(run_dir: Path, gen_model: l2g.ECGen, val_x: np.ndarray, val_lm: np.ndarray,
                 val_rows: pd.DataFrame, device: torch.device) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gen_model.eval()
    x = torch.from_numpy(val_x[:1]).to(device)
    lm = torch.from_numpy(val_lm[0, :1]).to(device).float()
    m = lm.unsqueeze(-1).expand_as(x)
    x_hat = torch.where(m.bool(), x * m, l2g.minmax_per_lead(gen_model(torch.cat([x * m, m], 1))))
    x, x_hat, lm = x[0].cpu().numpy(), x_hat[0].cpu().numpy(), lm[0].cpu().numpy()
    fig, axes = plt.subplots(12, 1, figsize=(10, 14), sharex=True)
    t = np.arange(x.shape[1]) / 250
    for j, a in enumerate(axes):
        a.plot(t, x[j], color="black", lw=0.7, label="reference")
        if lm[j] == 0:
            a.plot(t, x_hat[j], color="tab:red", lw=0.7, label="reconstructed")
        a.set_ylabel(f"{mz.LEAD_NAMES[j]}{'' if lm[j] == 0 else ' (obs)'}", fontsize=7)
        a.set_yticks([])
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("seconds")
    fig.suptitle(f"C_dev ecg_id {int(val_rows['ecg_id'].iloc[0])}: observed leads copied exactly; red = reconstruction")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "fig_example_reconstruction.png", dpi=110)
    plt.close(fig)
    gen_model.train()


def evaluate_existing_run(run_dir: Path, repo: Path, reason: str) -> dict[str, Any]:
    """Re-run ONLY the final evaluation of a finished run from its saved best checkpoint (no training).

    Uses the run's own config snapshot, requires the data cache to be reused unchanged, and checks that the
    checkpoint's epoch equals the best epoch in train_log.jsonl. Writes run_summary.json and figures.
    """
    t0 = time.perf_counter()
    snap = run_dir / "config_snapshot" / "train_l2g_v0_1.yaml"
    cfg = io.load_config(snap)
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    prep_path = repo / "configs" / "preprocessing.yaml"
    prep_cfg = io.load_config(prep_path)
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    lock = json.loads((repo / "artifacts" / "locks" / "protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    if io.sha256_file(repo / "configs" / "protocol_v0_1.yaml") != lock["sha256"]:
        raise RuntimeError("protocol_v0_1.yaml does not match its lock")
    for name in ("preprocessing.yaml", "data_audit.yaml"):
        if io.sha256_file(snap.parent / name) != io.sha256_file(repo / "configs" / name):
            raise RuntimeError(f"configs/{name} changed since the run; refusing to evaluate with different settings")
    dcfg = cfg["data"]
    manifest = load_frozen_manifest(audit_cfg, repo)
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    train_rows = select_role(manifest, dcfg["train_role"], excluded)
    val_rows = select_role(manifest, dcfg["validation_role"], excluded)
    if set(train_rows["patient_id"]) & set(val_rows["patient_id"]):
        raise RuntimeError("train/validation patient overlap")
    db = io.load_database(repo / audit_cfg["dataset"]["database_csv"])
    root = resolve_waveform_root(audit_cfg, repo)
    manifest_sha = audit_cfg["split"]["frozen_manifest_sha256"]
    caches = {}
    for role, rows in ((dcfg["train_role"], train_rows), (dcfg["validation_role"], val_rows)):
        key = cachemod.cache_key(prep_path, repo / audit_cfg["dataset"]["database_csv"], manifest_sha, role,
                                 rows["ecg_id"].tolist())
        caches[role] = cachemod.ensure_cache(role, rows, root, db, prep_cfg, key, Path(dcfg["cache_dir"]))
        if not caches[role][2]["reused"]:
            raise RuntimeError(f"{role} cache had to be rebuilt; the training-time data cannot be confirmed")
    k = int(dcfg["validation_masks_per_record"])
    val_x = windows_at(caches[dcfg["validation_role"]][0], prep_cfg, int(dcfg["validation_window_offset"]))
    val_lm = validation_lead_masks(val_rows["ecg_id"].tolist(), int(cfg["seed"]), k)
    corr_np = np.load(run_dir / "afit_lead_correlation.npy")

    history = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    best_row = [h for h in history if h["improved"]][-1]
    best = {"selection_loss": best_row["val_selection_loss"], "epoch": best_row["epoch"],
            "sha256": best_row.get("best_sha256")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_model = l2g.ECGen(int(cfg["model"]["in_channels"])).to(device)
    ssim = l2g.SSIM1D().to(device)
    tr = cfg["training"]
    epochs_run = len(history)
    stop_reason = ("max_epochs_reached" if epochs_run == int(tr["max_epochs"])
                   else f"early stopping: no improvement for {tr['early_stopping_patience']} epochs")
    summary: dict[str, Any] = {
        "run_name": cfg["run_name"], "status": "FINISHED", "stop_reason": stop_reason, "epochs_run": epochs_run,
        "best_epoch": best["epoch"], "best_val_selection_loss": best["selection_loss"],
        "roles_loaded": sorted({dcfg["train_role"], dcfg["validation_role"]}), "train_records": len(train_rows),
        "train_patients": int(train_rows["patient_id"].nunique()),
        "train_samples_per_epoch": len(train_rows) * len(dcfg["train_window_offsets"]),
        "val_records": len(val_rows), "val_patients": int(val_rows["patient_id"].nunique()),
        "val_rows": int(len(val_rows) * k), "excluded_ecg_ids": sorted(excluded),
        "cache_meta": {r: c[2] for r, c in caches.items()},
        "hashes": {"training_config": io.sha256_file(snap), "protocol": lock["sha256"], "frozen_manifest": manifest_sha,
                   "code_tree_at_evaluation": code_tree_sha256(repo)["tree_sha256"]},
        "environment": environment(), "full_training": True, "downstream_components_trained": False,
        "fold10_loaded": False,
        "final_evaluation_rerun": {"reason": reason, "training_was_repeated": False,
                                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    summary.update(final_evaluation(run_dir, best, gen_model, ssim, val_x, val_lm, val_rows, k, corr_np, device, cfg))
    summary["timing"] = {"training_seconds_sum_of_epochs": round(sum(h["epoch_seconds"] for h in history), 1),
                         "mean_epoch_seconds": round(float(np.mean([h["epoch_seconds"] for h in history])), 1),
                         "max_peak_gpu_MiB": max(h["peak_gpu_MiB"] or 0 for h in history),
                         "evaluation_rerun_seconds": round(time.perf_counter() - t0, 1)}
    make_plots(run_dir, history, summary, int(tr["adversarial_start_epoch"]))
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled L2G training (Milestone 2c).")
    parser.add_argument("--config", default="configs/train_l2g_v0_1.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--evaluate-only", metavar="RUN_DIR",
                        help="re-run only the final evaluation of a finished run (no training)")
    parser.add_argument("--reason", default="final evaluation re-run", help="recorded with --evaluate-only")
    args = parser.parse_args(argv)
    repo = Path(args.root).resolve()
    if args.evaluate_only:
        summary = evaluate_existing_run((repo / args.evaluate_only).resolve(), repo, args.reason)
        print(json.dumps({k: summary.get(k) for k in ("status", "best_epoch", "best_val_selection_loss",
                                                      "adequate_for_downstream")}, default=str))
        return 0 if summary.get("checkpoint_integrity", {}).get("pass") else 1
    config_path = (repo / args.config).resolve()
    summary = train(io.load_config(config_path), repo, config_path)
    print(json.dumps({k: summary.get(k) for k in ("status", "stop_reason", "epochs_run", "best_epoch",
                                                  "best_val_selection_loss", "adequate_for_downstream")}, default=str))
    return 0 if summary["status"] == "FINISHED" else 1


if __name__ == "__main__":
    sys.exit(main())
