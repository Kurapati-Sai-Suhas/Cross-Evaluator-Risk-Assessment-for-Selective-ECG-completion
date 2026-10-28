"""Figure fig_m3b_cdev_auroc_by_n_observed from artifacts/reports/m3a_cdev_evaluator_comparison.json (no recomputation).

    .\\.venv\\Scripts\\python.exe -m src.eval.plot_m3b_comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"e1_complete": "E1(x) complete ECG (reference)", "e1_recon": "E1(x_hat) on G v0.1 reconstruction",
          "edir": "E_dir(x_obs, m) direct reduced-lead"}


def main() -> int:
    repo = Path(".").resolve()
    s = json.loads((repo / "artifacts" / "reports" / "m3a_cdev_evaluator_comparison.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, len(s["results"]), figsize=(11, 4.2), sharey=True)
    for ax, (code, res) in zip(axes, s["results"].items()):
        bins = res["auroc_point_by_n_observed_descriptive"]
        names = list(bins)
        for m, label in LABELS.items():
            ax.plot(range(len(names)), [bins[b][m] for b in names], marker="o", label=label)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{b}\n({bins[b]['rows']} rows, {bins[b]['positive_rows']} pos)" for b in names], fontsize=7)
        ax.set_title(f"{code}: C_dev AUROC by observed leads")
        ax.set_xlabel("number of observed leads")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC (point estimate)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = repo / "artifacts" / "reports" / "figures" / "fig_m3b_cdev_auroc_by_n_observed.png"
    fig.savefig(out, dpi=130)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
