"""Milestone 3a: train a diagnostic evaluator (E1; also E_dir/E2 with their own configs). Protocol v0.1 Section 5.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.train.train_evaluator --config configs/evaluator_e1_v0_1.yaml

* Fits on A_fit only; uses C_dev for early stopping, best-so-far checkpoint selection, the Youden threshold and
  the eligibility assessment. B_gate, 9a, 9b and fold 10 are never loaded.
* Runs every pre-declared attempt in the config (at most 3), keeps the attempt with the best C_dev macro-AUROC.
* Eligibility (floors read from the pinned protocol): for every output, the patient-bootstrap 95% CI lower bound of
  the C_dev AUROC >= auroc_lower_bound_floor (E-1) and the CI half-width <= max_ci_half_width (E-2).
* Saves per-record C_dev predictions (with patient_id), raw logits, per-epoch logs, checkpoints with SHA-256, figures
  and summary.json. The selected checkpoint is made read-only ONLY if eligibility passes.
For a comparator (config ``eligibility_applies: false``, i.e. E_dir) eligibility is reported as not applicable and
the selected checkpoint is frozen as it is.
Exit code: 0 = eligible (or not applicable), 3 = trained but NOT eligible (stop; owner decision), 1 = failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data import missingness as mz
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.data.ptbxl_waveforms import resolve_waveform_root
from src.data.targets import target_matrix
from src.eval import classification as cl
from src.models.evaluators import ResNet1dWang
from src.train import cache as cachemod
from src.train import recon_metrics as rm
from src.train.train_l2g import select_role, validation_lead_masks, windows_at
from src.utils.provenance import code_tree_sha256, environment


def build_model(cfg: dict[str, Any], n_outputs: int) -> torch.nn.Module:
    m = cfg["model"]
    if m["family"] != "resnet1d_wang":
        raise ValueError(f"model family {m['family']} not implemented in v0.1")
    return ResNet1dWang(int(m["in_channels"]), n_outputs, tuple(m["widths"]))


@torch.no_grad()
def predict(model, x: np.ndarray, device, masks: np.ndarray | None = None, batch: int = 256) -> np.ndarray:
    """Logits for complete windows (N, 12, T), or for masked rows when ``masks`` (N, k, 12) is given (N*k rows)."""
    model.eval()
    out = []
    k = 1 if masks is None else masks.shape[1]
    per = max(1, batch // k)
    for s in range(0, len(x), per):
        xb = torch.from_numpy(x[s:s + per]).to(device)
        if masks is not None:
            xb = xb.repeat_interleave(k, dim=0)
            lm = torch.from_numpy(masks[s:s + per]).to(device).float().reshape(-1, 12)
            m = lm.unsqueeze(-1).expand_as(xb)
            xb = torch.cat([xb * m, m], dim=1)
        out.append(model(xb).float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def macro_auroc(logits: np.ndarray, labels: np.ndarray) -> tuple[float, list[float]]:
    per = [cl.weighted_auroc(logits[:, j], labels[:, j]) for j in range(labels.shape[1])]
    return float(np.mean(per)), per


def train_attempt(att: dict[str, Any], cfg: dict[str, Any], data: dict[str, Any], out: Path, device) -> dict[str, Any]:
    tr = cfg["training"]
    torch.manual_seed(int(att["seed"]))
    model = build_model(cfg, data["y_train"].shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(att["lr"]), weight_decay=float(att["weight_decay"]))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data["pos_weight"], dtype=torch.float32, device=device))
    masked = cfg["input"] == "masked"
    combo_masks = torch.tensor(np.stack([mz.lead_mask(c) for c in mz.l2g_lead_combinations()]),
                               dtype=torch.float32, device=device)
    n, bs = len(data["rec_index"]), int(tr["batch_size"])
    best = {"score": -np.inf, "epoch": None, "sha256": None}
    patience = int(tr["early_stopping_patience"])
    log = (out / "train_log.jsonl").open("w", encoding="utf-8")
    history = []
    for epoch in range(int(tr["max_epochs"])):
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng([int(att["seed"]), epoch])
        order = rng.permutation(n)
        combo_idx = rng.integers(0, len(combo_masks), size=n)
        total, seen = 0.0, 0
        model.train()
        for s in range(0, n, bs):
            sel = order[s:s + bs]
            sig = torch.from_numpy(data["train_sig"][data["rec_index"][sel]]).to(device)
            x = rm.gpu_crop_normalize(sig, torch.from_numpy(data["offsets"][sel]).to(device))
            if masked:
                lm = combo_masks[torch.from_numpy(combo_idx[sel]).to(device)]
                m = lm.unsqueeze(-1).expand_as(x)
                x = torch.cat([x * m, m], dim=1)
            y = torch.from_numpy(data["y_train"][data["rec_index"][sel]]).to(device).float()
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)
            seen += len(sel)
        logits = predict(model, data["val_x"], device, data["val_masks"] if masked else None)
        score, per = macro_auroc(logits, data["y_val_rows"])
        improved = score > best["score"]
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch, "c_dev_macro_auroc": score, "attempt": att,
                        "config_name": cfg["run_name"]}, out / "best.pt")
            best = {"score": score, "epoch": epoch, "sha256": io.sha256_file(out / "best.pt")}
            patience = int(tr["early_stopping_patience"])
        else:
            patience -= 1
        row = {"attempt": att["id"], "epoch": epoch, "train_loss": total / seen, "c_dev_macro_auroc": score,
               **{f"c_dev_auroc_{name}": v for name, v in zip(data["outputs"], per)}, "improved": improved,
               "best_sha256": best["sha256"] if improved else None,
               "epoch_seconds": round(time.perf_counter() - t0, 1),
               "peak_gpu_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device.type == "cuda" else None}
        history.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"[{cfg['evaluator_id']} attempt {att['id']}] epoch {epoch:2d} loss {row['train_loss']:.4f} | C_dev macro AUROC "
              f"{score:.4f} ({', '.join(f'{v:.4f}' for v in per)}) | {row['epoch_seconds']:.0f}s | "
              f"{'*best*' if improved else f'patience {patience}'}", flush=True)
        if patience <= 0:
            break
    log.close()
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    return {"attempt": att, "best_epoch": best["epoch"], "best_c_dev_macro_auroc": best["score"],
            "best_sha256": best["sha256"], "epochs_run": len(history), "history": history}


def run(cfg: dict[str, Any], repo: Path, config_path: Path) -> dict[str, Any]:
    t_start = time.perf_counter()
    tree_at_start = code_tree_sha256(repo)["tree_sha256"]  # the code that actually runs
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    prep_path = repo / "configs" / "preprocessing.yaml"
    prep_cfg = io.load_config(prep_path)
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    lock = json.loads((repo / "artifacts" / "locks" / "protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    if io.sha256_file(repo / "configs" / "protocol_v0_1.yaml") != lock["sha256"]:
        raise RuntimeError("protocol does not match its lock")
    attempts = cfg["attempts"]
    if not 1 <= len(attempts) <= int(protocol["eligibility"]["max_training_attempts"]):
        raise RuntimeError("number of attempts exceeds the protocol maximum")
    run_dir = repo / cfg["output_dir"] / cfg["run_name"]
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} exists; use a new run_name")
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "config_snapshot").mkdir()
    for p in (config_path, prep_path, repo / "configs" / "protocol_v0_1.yaml", repo / "configs" / "data_audit.yaml"):
        shutil.copy(p, run_dir / "config_snapshot" / p.name)

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
    outputs = list(protocol["targets"]["primary_outputs"])
    y_train = target_matrix(db, train_rows["ecg_id"].tolist(), protocol)
    y_val = target_matrix(db, val_rows["ecg_id"].tolist(), protocol)
    masked = cfg["input"] == "masked"
    k = int(dcfg["validation_masks_per_record"]) if masked else 1
    val_masks = validation_lead_masks(val_rows["ecg_id"].tolist(), int(cfg["seed"]), k) if masked else None
    offsets = np.asarray(dcfg["train_window_offsets"], dtype=np.int64)
    data = {
        "train_sig": caches[dcfg["train_role"]][0], "outputs": outputs, "y_train": y_train,
        "rec_index": np.repeat(np.arange(len(train_rows)), len(offsets)), "offsets": np.tile(offsets, len(train_rows)),
        "val_x": windows_at(caches[dcfg["validation_role"]][0], prep_cfg, int(dcfg["validation_window_offset"])),
        "val_masks": val_masks, "y_val_rows": np.repeat(y_val, k, axis=0),
        "pos_weight": [(len(y_train) - y_train[:, j].sum()) / max(y_train[:, j].sum(), 1) for j in range(len(outputs))],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for att in attempts:
        out = run_dir / f"attempt_{att['id']}"
        out.mkdir()
        results.append(train_attempt(att, cfg, data, out, device))
    chosen = max(results, key=lambda r: r["best_c_dev_macro_auroc"])
    sel_dir = run_dir / f"attempt_{chosen['attempt']['id']}"
    ckpt = sel_dir / "best.pt"
    if io.sha256_file(ckpt) != chosen["best_sha256"]:
        raise RuntimeError("selected checkpoint changed after saving")
    model = build_model(cfg, len(outputs)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    logits = predict(model, data["val_x"], device, val_masks)
    reloaded_macro, _ = macro_auroc(logits, data["y_val_rows"])
    probs = 1 / (1 + np.exp(-logits))

    pid = np.repeat(val_rows["patient_id"].to_numpy(), k)
    pred = pd.DataFrame({"ecg_id": np.repeat(val_rows["ecg_id"].to_numpy(), k), "patient_id": pid,
                         "mask_index": np.tile(np.arange(k), len(val_rows))})
    if masked:
        pred["observed_leads"] = [";".join(np.asarray(mz.LEAD_NAMES)[row.astype(bool)])
                                  for row in val_masks.reshape(-1, 12)]
    ev = protocol["eligibility"]
    boot_cfg = protocol["uncertainty"]
    per_output = {}
    for j, name in enumerate(outputs):
        y = data["y_val_rows"][:, j]
        pred[f"y_{name}"], pred[f"logit_{name}"], pred[f"prob_{name}"] = y, logits[:, j], probs[:, j]
        auc = cl.cluster_bootstrap_metric(cl.auroc_fn(probs[:, j], y), y, pid, replicates=boot_cfg["replicates"],
                                          seed=int(cfg["seed"]), level=boot_cfg["two_sided_level"])
        ap = cl.cluster_bootstrap_metric(cl.average_precision_fn(probs[:, j], y), y, pid,
                                         replicates=boot_cfg["replicates"], seed=int(cfg["seed"]),
                                         level=boot_cfg["two_sided_level"])
        tau = cl.youden_threshold(probs[:, j], y)
        ops = {m: cl.cluster_bootstrap_metric(lambda w, m=m: cl.operating_point(probs[:, j], y, tau, w)[m], y, pid,
                                              replicates=boot_cfg["replicates"], seed=int(cfg["seed"]), level=boot_cfg["two_sided_level"])
               for m in ("sensitivity", "specificity", "ppv")}
        per_output[name] = {
            "auroc": auc, "auprc": ap, "prevalence_rows": float(y.mean()), "youden_tau_probability": tau,
            "operating_point_at_tau_in_sample_descriptive": ops,  # tau chosen on these same C_dev rows
            "E1_lower_bound_ok": auc["ci_low"] >= float(ev["auroc_lower_bound_floor"]),
            "E2_half_width_ok": auc["half_width"] <= float(ev["max_ci_half_width"]),
        }
    pred.to_csv(run_dir / "c_dev_predictions_selected.csv", index=False)
    applies = bool(cfg.get("eligibility_applies", True))  # False for the E_dir comparator
    eligible = all(v["E1_lower_bound_ok"] and v["E2_half_width_ok"] for v in per_output.values()) if applies else None
    freeze = eligible is True or not applies  # a comparator's selected checkpoint is frozen as it is
    if freeze:
        os.chmod(ckpt, stat.S_IREAD)
    _plots(run_dir, results, data, probs, outputs, per_output)
    summary = {
        "milestone": "M3a", "evaluator_id": cfg["evaluator_id"], "run_name": cfg["run_name"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": cfg["input"], "outputs": outputs, "roles_loaded": sorted({dcfg["train_role"], dcfg["validation_role"]}),
        "fold10_loaded": False, "d1_generated": False,
        "train_records": len(train_rows), "train_patients": int(train_rows["patient_id"].nunique()),
        "train_positive_records": {n: int(y_train[:, j].sum()) for j, n in enumerate(outputs)},
        "train_pos_weight": dict(zip(outputs, map(float, data["pos_weight"]))),
        "val_records": len(val_rows), "val_patients": int(val_rows["patient_id"].nunique()), "val_rows": int(len(pred)),
        "val_positive_records": {n: int(y_val[:, j].sum()) for j, n in enumerate(outputs)},
        "val_positive_patients": {n: int(val_rows.loc[y_val[:, j] == 1, "patient_id"].nunique()) for j, n in enumerate(outputs)},
        "attempts": [{k_: v for k_, v in r.items() if k_ != "history"} for r in results],
        "selected_attempt": chosen["attempt"]["id"], "selected_checkpoint": str(ckpt.relative_to(repo) if ckpt.is_relative_to(repo) else ckpt),
        "selected_checkpoint_sha256": chosen["best_sha256"], "reloaded_c_dev_macro_auroc": reloaded_macro,
        "reload_matches_logged": abs(reloaded_macro - chosen["best_c_dev_macro_auroc"]) < 1e-6,
        "eligibility_rule": {"auroc_lower_bound_floor": ev["auroc_lower_bound_floor"],
                             "max_ci_half_width": ev["max_ci_half_width"], "measured_on": ev["measured_on"]},
        "per_output": per_output, "eligibility_applies": applies, "eligible": eligible,
        "checkpoint_made_read_only": freeze,
        "hashes": {"config": io.sha256_file(config_path), "protocol": lock["sha256"], "frozen_manifest": manifest_sha,
                   "code_tree_at_start": tree_at_start, "code_tree_at_end": code_tree_sha256(repo)["tree_sha256"]},
        "environment": environment(), "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _plots(run_dir: Path, results, data, probs, outputs, per_output) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        h = pd.DataFrame(r["history"])
        ax[0].plot(h.epoch, h.train_loss, label=f"attempt {r['attempt']['id']}")
        ax[1].plot(h.epoch, h.c_dev_macro_auroc, label=f"attempt {r['attempt']['id']}")
    ax[0].set_title("training loss (A_fit, weighted BCE)")
    ax[1].set_title("C_dev macro AUROC")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "fig_training_curves.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, len(outputs), figsize=(5 * len(outputs), 4.5))
    for j, name in enumerate(outputs):
        y = data["y_val_rows"][:, j].astype(bool)
        order = np.argsort(-probs[:, j])
        tpr = np.r_[0, np.cumsum(y[order]) / y.sum()]
        fpr = np.r_[0, np.cumsum(~y[order]) / (~y).sum()]
        a = per_output[name]["auroc"]
        ax[j].plot(fpr, tpr, label=f"AUROC {a['point']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
        ax[j].plot([0, 1], [0, 1], "k:", lw=0.7)
        ax[j].set_title(f"{name} on C_dev (selected attempt)")
        ax[j].set_xlabel("1 - specificity")
        ax[j].set_ylabel("sensitivity")
        ax[j].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "fig_roc_c_dev.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a diagnostic evaluator (Milestone 3a).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    repo = Path(args.root).resolve()
    config_path = (repo / args.config).resolve()
    s = run(io.load_config(config_path), repo, config_path)
    print(json.dumps({"evaluator": s["evaluator_id"], "selected_attempt": s["selected_attempt"], "eligible": s["eligible"],
                      "auroc": {n: {k_: round(v["auroc"][k_], 4) for k_ in ("point", "ci_low", "ci_high", "half_width")}
                                for n, v in s["per_output"].items()}}, indent=1))
    return 0 if s["eligible"] is not False else 3


if __name__ == "__main__":
    sys.exit(main())
