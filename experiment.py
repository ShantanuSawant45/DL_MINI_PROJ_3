from __future__ import annotations

import argparse
import csv
import time
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf

import config
from data import load_dataset
from train import train_vae
from clustering import run_kmeans_clustering, encode_to_latent
from metrics import compute_image_metrics


def _reconstruct(encoder, decoder, x: np.ndarray,
                 batch_size: int = 256) -> np.ndarray:
    x_hat_parts = []
    for start in range(0, len(x), batch_size):
        batch   = x[start:start + batch_size]
        _, _, z = encoder(batch, training=False)
        x_hat   = decoder(z,    training=False)
        x_hat_parts.append(x_hat.numpy())
    return np.concatenate(x_hat_parts, axis=0)


def _append_csv(filepath: Path, row: dict) -> None:
    is_new = not filepath.exists()
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_main_sweep(
    datasets: list[str],
    code_sizes: list[int],
    loss_types: list[str],
    n_runs: int,
    epochs: int,
    output_path: Path,
    verbose: bool = True,
) -> None:
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset}")
        print(f"{'='*60}")

        data = load_dataset(dataset)

        for code_size in code_sizes:
            for loss_type in loss_types:
                n_exp = 1.0 if loss_type == "ssim_weighted" else 0.0

                acc_runs:  list[dict]  = []
                nmi_runs:  list[float] = []
                img_runs:  list[dict]  = []
                time_runs: list[float] = []

                for run_idx in range(n_runs):
                    seed = config.GLOBAL_SEED + run_idx
                    if verbose:
                        print(f"\n  [{dataset}] code={code_size} "
                              f"loss={loss_type} run={run_idx+1}/{n_runs}")

                    encoder, decoder, _vae, elapsed = train_vae(
                        dataset=dataset,
                        latent_dim=code_size,
                        loss_type=loss_type,
                        n_exponent=n_exp,
                        seed=seed,
                        epochs=epochs,
                        verbose=verbose,
                    )

                    clust = run_kmeans_clustering(
                        encoder,
                        data["x_all"],
                        data["y_all"],
                        seed=seed,
                    )
                    acc_runs.append({k: v["acc"] for k, v in clust.items()})
                    nmi_runs.append(clust["k10"]["nmi"])

                    x_hat = _reconstruct(encoder, decoder, data["x_test"])
                    img_m = compute_image_metrics(data["x_test"], x_hat,
                                                  max_samples=500)
                    img_runs.append(img_m)
                    time_runs.append(elapsed)

                def mean_of(key: str, dicts: list[dict]) -> float:
                    return float(np.mean([d[key] for d in dicts]))

                row = {
                    "dataset":        dataset,
                    "code_size":      code_size,
                    "loss_type":      loss_type,
                    "n_runs":         n_runs,
                    "epochs":         epochs,
                    "acc_k10":        mean_of("k10", acc_runs),
                    "acc_k20":        mean_of("k20", acc_runs),
                    "acc_k30":        mean_of("k30", acc_runs),
                    "nmi_k10":        float(np.mean(nmi_runs)),
                    "mse":            mean_of("mse",  img_runs),
                    "ssim":           mean_of("ssim", img_runs),
                    "uqi":            mean_of("uqi",  img_runs),
                    "vif":            mean_of("vif",  img_runs),
                    "train_time_sec": float(np.mean(time_runs)),
                }
                _append_csv(output_path, row)
                if verbose:
                    print(f"  → Saved row: {row}")


def run_n_exponent_sweep(
    datasets: list[str],
    best_code_sizes: dict[str, int],
    n_exponents: list[float],
    n_runs: int,
    epochs: int,
    output_path: Path,
    verbose: bool = True,
) -> None:
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"n-EXPONENT STUDY: {dataset}")
        print(f"{'='*60}")

        code_size = best_code_sizes.get(dataset, config.BEST_CODE_SIZE[dataset])
        data      = load_dataset(dataset)

        for n_exp in n_exponents:
            loss_type = "ssim_weighted" if n_exp > 0 else "standard"
            acc_runs: list[float] = []

            for run_idx in range(n_runs):
                seed = config.GLOBAL_SEED + run_idx
                if verbose:
                    print(f"\n  [{dataset}] code={code_size} n={n_exp} "
                          f"run={run_idx+1}/{n_runs}")

                encoder, _dec, _vae, _t = train_vae(
                    dataset=dataset,
                    latent_dim=code_size,
                    loss_type=loss_type,
                    n_exponent=n_exp,
                    seed=seed,
                    epochs=epochs,
                    verbose=verbose,
                )

                clust = run_kmeans_clustering(
                    encoder,
                    data["x_all"],
                    data["y_all"],
                    seed=seed,
                )
                acc_runs.append(clust["k10"]["acc"])

            row = {
                "dataset":   dataset,
                "code_size": code_size,
                "n_exp":     n_exp,
                "loss_type": loss_type,
                "n_runs":    n_runs,
                "acc_k10":   float(np.mean(acc_runs)),
            }
            _append_csv(output_path, row)
            if verbose:
                print(f"  → Saved row: {row}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAE clustering experiment (Abdella & Uysal, IEEE Access 2020)"
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=["mnist", "fashion_mnist", "all"],
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--runs",   type=int, default=config.N_RUNS)
    parser.add_argument("--quick",  action="store_true")
    parser.add_argument(
        "--sweep", default="all",
        choices=["main", "n_exp", "all"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    datasets   = config.DATASETS if args.dataset == "all" else [args.dataset]
    epochs     = args.epochs
    n_runs     = args.runs

    if args.quick:
        epochs      = 2
        n_runs      = 1
        code_sizes  = [2, 8]
        n_exponents = [0, 1]
        print("⚡ Quick mode: epochs=2, runs=1, code_sizes=[2,8]\n")
    else:
        code_sizes  = config.CODE_SIZES
        n_exponents = config.N_EXPONENTS

    loss_types = ["standard", "ssim_weighted"]
    main_csv   = config.RESULTS_DIR / "main_results.csv"
    n_exp_csv  = config.RESULTS_DIR / "n_exponent_results.csv"

    if args.sweep in ("main", "all"):
        print("\n>>> SWEEP 1: Main comparison\n")
        if main_csv.exists():
            main_csv.unlink()
        run_main_sweep(
            datasets=datasets,
            code_sizes=code_sizes,
            loss_types=loss_types,
            n_runs=n_runs,
            epochs=epochs,
            output_path=main_csv,
        )
        print(f"\nMain results saved to: {main_csv}")

    best_code: dict[str, int] = dict(config.BEST_CODE_SIZE)

    if main_csv.exists():
        import csv as _csv
        import collections

        best_acc: dict[str, float] = collections.defaultdict(float)
        with open(main_csv, newline="") as f:
            for row in _csv.DictReader(f):
                if row["loss_type"] != "standard":
                    continue
                ds   = row["dataset"]
                acc  = float(row["acc_k10"])
                code = int(row["code_size"])
                if acc > best_acc[ds]:
                    best_acc[ds]  = acc
                    best_code[ds] = code

    if args.sweep in ("n_exp", "all"):
        print("\n>>> SWEEP 2: n-exponent study\n")
        if n_exp_csv.exists():
            n_exp_csv.unlink()
        run_n_exponent_sweep(
            datasets=datasets,
            best_code_sizes=best_code,
            n_exponents=n_exponents,
            n_runs=n_runs,
            epochs=epochs,
            output_path=n_exp_csv,
        )
        print(f"\nn-exponent results saved to: {n_exp_csv}")

    print("\n✅ Experiment complete.")


if __name__ == "__main__":
    main()
