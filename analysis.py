from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import config
from data import load_dataset
from train import train_vae


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cast: dict = {}
            for k, v in row.items():
                try:
                    cast[k] = float(v)
                except (ValueError, TypeError):
                    cast[k] = v
            rows.append(cast)
    return rows


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(groups)


def print_summary_table(rows: list[dict]) -> None:
    if not rows:
        print("[WARN] No main_results.csv data available for summary table.")
        return

    metrics = ["acc_k10", "nmi_k10", "ssim", "uqi", "mse", "vif",
               "train_time_sec"]

    by_ds = _group(rows, "dataset")
    for ds, ds_rows in by_ds.items():
        std_rows  = [r for r in ds_rows if r["loss_type"] == "standard"]
        ssim_rows = [r for r in ds_rows if r["loss_type"] == "ssim_weighted"]

        print(f"\n{'='*70}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*70}")
        header = f"{'Metric':<20} {'Standard':>12} {'SSIM-Weighted':>14} {'Diff':>10}"
        print(header)
        print("-" * 70)

        for m in metrics:
            std_vals  = [r[m] for r in std_rows  if not isinstance(r[m], str)]
            ssim_vals = [r[m] for r in ssim_rows if not isinstance(r[m], str)]
            if not std_vals or not ssim_vals:
                continue
            s_mean = np.mean(std_vals)
            m_mean = np.mean(ssim_vals)
            diff   = m_mean - s_mean

            if "acc" in m:
                diff_str = f"{diff:+.4f} ({diff / max(s_mean, 1e-9)*100:+.1f}%)"
            else:
                diff_str = f"{diff:+.4f}"

            print(f"  {m:<18} {s_mean:>12.4f} {m_mean:>14.4f} {diff_str:>10}")

    print()


def compute_correlation_table(rows: list[dict]) -> None:
    if not rows:
        print("[WARN] No data for correlation analysis.")
        return

    quality_metrics = ["ssim", "uqi", "mse", "vif"]
    by_ds = _group(rows, "dataset")

    print("\n" + "="*70)
    print("Pearson Correlation: ACC@K=10 vs Image Quality (paper Table 8)")
    print("="*70)

    for ds, ds_rows in by_ds.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Metric':<8} {'r':>8} {'p-value':>12}  {'Significance'}")
        print("  " + "-" * 50)

        acc_vals = [r["acc_k10"] for r in ds_rows
                    if not isinstance(r.get("acc_k10", ""), str)]

        for qm in quality_metrics:
            qm_vals = [r[qm] for r in ds_rows
                       if not isinstance(r.get(qm, ""), str)]

            if len(acc_vals) < 3 or len(qm_vals) < 3:
                print(f"  {qm:<8}   N/A (insufficient data)")
                continue

            n = min(len(acc_vals), len(qm_vals))
            try:
                r, p = pearsonr(acc_vals[:n], qm_vals[:n])
                sig = ("***" if p < 0.001 else
                       "**"  if p < 0.01  else
                       "*"   if p < 0.05  else
                       "ns")
                print(f"  {qm:<8} {r:>8.4f} {p:>12.4e}  {sig}")
            except Exception as e:
                print(f"  {qm:<8}   Error: {e}")


_COLORS  = {"standard": "#4C72B0", "ssim_weighted": "#DD8452"}
_MARKERS = {"standard": "o",       "ssim_weighted": "s"}


def _plot_metric_vs_code(rows: list[dict], y_key: str, y_label: str,
                         title_prefix: str, filename: str) -> None:
    if not rows:
        return

    by_ds = _group(rows, "dataset")
    fig, axes = plt.subplots(1, len(by_ds), figsize=(6 * len(by_ds), 5),
                             squeeze=False)

    for ax, (ds, ds_rows) in zip(axes[0], by_ds.items()):
        by_loss = _group(ds_rows, "loss_type")
        for loss_type, l_rows in by_loss.items():
            sorted_rows = sorted(l_rows, key=lambda r: float(r["code_size"]))
            xs = [float(r["code_size"]) for r in sorted_rows]
            ys = [float(r[y_key])       for r in sorted_rows]
            ax.plot(xs, ys,
                    color=_COLORS.get(loss_type, "gray"),
                    marker=_MARKERS.get(loss_type, "^"),
                    linewidth=2, markersize=7,
                    label=loss_type.replace("_", " ").title())

        ax.set_title(f"{title_prefix} – {ds}", fontsize=13)
        ax.set_xlabel("Latent Dimension (Code Size)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = config.RESULTS_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_acc_vs_code_size(rows: list[dict]) -> None:
    _plot_metric_vs_code(rows, "acc_k10", "Clustering Accuracy (K=10)",
                         "Clustering Accuracy", "clustering_acc_vs_code_size.png")


def plot_ssim_vs_code_size(rows: list[dict]) -> None:
    _plot_metric_vs_code(rows, "ssim", "SSIM",
                         "SSIM", "ssim_vs_code_size.png")


def plot_n_exponent_vs_acc(n_exp_rows: list[dict]) -> None:
    if not n_exp_rows:
        print("[WARN] No n_exponent_results.csv data.")
        return

    by_ds = _group(n_exp_rows, "dataset")
    fig, axes = plt.subplots(1, len(by_ds), figsize=(6 * len(by_ds), 5),
                             squeeze=False)

    for ax, (ds, ds_rows) in zip(axes[0], by_ds.items()):
        sorted_rows = sorted(ds_rows, key=lambda r: float(r["n_exp"]))
        xs = [float(r["n_exp"])   for r in sorted_rows]
        ys = [float(r["acc_k10"]) for r in sorted_rows]
        ax.plot(xs, ys, color="#2ca02c", marker="D", linewidth=2, markersize=8)
        ax.set_title(f"n-Exponent vs ACC – {ds}", fontsize=13)
        ax.set_xlabel("SSIM Weight Exponent n", fontsize=11)
        ax.set_ylabel("Clustering Accuracy (K=10)", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = config.RESULTS_DIR / "n_exponent_vs_acc.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_generated_samples(dataset: str, latent_dim: int = 8,
                           epochs: int = 10, n_cols: int = 10,
                           n_rows: int = 5) -> None:
    print(f"  Generating sample grid for {dataset}, latent_dim={latent_dim}…")
    encoder, decoder, _vae, _ = train_vae(
        dataset=dataset,
        latent_dim=latent_dim,
        loss_type="standard",
        seed=config.GLOBAL_SEED,
        epochs=epochs,
        verbose=False,
    )

    import tensorflow as tf

    n_samples = n_cols * n_rows
    z_random  = tf.random.normal(shape=(n_samples, latent_dim),
                                 seed=config.GLOBAL_SEED)
    samples   = decoder(z_random, training=False).numpy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    for ax, img in zip(axes.flat, samples):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle(f"{dataset} – Generated Samples (latent_dim={latent_dim})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = config.RESULTS_DIR / f"generated_samples_z{latent_dim}_{dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis for VAE clustering experiment")
    parser.add_argument("--latent_dim",      type=int, default=8)
    parser.add_argument("--epochs",          type=int, default=10)
    parser.add_argument("--skip_generation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    main_csv  = config.RESULTS_DIR / "main_results.csv"
    n_exp_csv = config.RESULTS_DIR / "n_exponent_results.csv"

    rows       = _load_csv(main_csv)
    n_exp_rows = _load_csv(n_exp_csv)

    if not rows and not n_exp_rows:
        print("[ERROR] No results found. Run experiment.py first.")
        sys.exit(1)

    print("\n>>> Summary Table")
    print_summary_table(rows)

    print("\n>>> Pearson Correlation Table")
    compute_correlation_table(rows)

    print("\n>>> Generating plots…")
    plot_acc_vs_code_size(rows)
    plot_ssim_vs_code_size(rows)
    plot_n_exponent_vs_acc(n_exp_rows)

    if not args.skip_generation:
        for ds in config.DATASETS:
            plot_generated_samples(
                dataset=ds,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
            )
    else:
        print("  Skipping sample generation (--skip_generation).")

    print("\n✅ Analysis complete.  Plots saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
