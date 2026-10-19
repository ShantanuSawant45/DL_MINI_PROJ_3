# VAE Clustering – Replication of Abdella & Uysal (IEEE Access 2020)

> **Paper**: *"A Statistical Comparative Study on Image Reconstruction and
> Clustering With Novel VAE Cost Function"*  
> Abdella M. & Uysal M., *IEEE Access*, 2020.

This project trains a convolutional Variational Autoencoder (VAE) on MNIST
and Fashion-MNIST, projects images into a compact latent space, runs K-Means
clustering, and compares the standard VAE loss against a novel SSIM-weighted
loss proposed in the paper.

---

## Paper Summary

The paper introduces a modified VAE cost function that multiplies the standard
negative ELBO by `(1 − SSIM(x, x̂))^n` (Eq. 23).  The intuition: when
reconstruction quality is already good (high SSIM), the weight decreases,
focusing the model's capacity on harder examples.  The authors show that this
modification can improve both image quality metrics (SSIM, UQI, VIF, MSE) and
unsupervised clustering accuracy (ACC, NMI) compared to the standard VAE loss.

---

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Central hyper-parameters and sweep settings |
| `data.py` | MNIST / Fashion-MNIST loading, scaled to [0,1] |
| `model.py` | Conv-VAE architecture (encoder + decoder + reparametrisation) |
| `losses.py` | Standard VAE loss and SSIM-weighted loss (paper Eq. 23) |
| `metrics.py` | MSE, SSIM, UQI (paper Eq. 8), VIF, ACC (Hungarian), NMI |
| `clustering.py` | Encode images → K-Means → ACC/NMI for each K |
| `train.py` | `train_vae()` with custom GradientTape loop |
| `experiment.py` | Full sweep (Sweep 1: main comparison, Sweep 2: n-exponent study) |
| `analysis.py` | Pearson correlations, summary tables, and all paper figures |
| `results/` | CSVs and PNG plots saved here |

---

## Installation

```bash
# (Recommended) create a virtual environment first
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
# or: source .venv/bin/activate   (Linux/macOS)

pip install -r requirements.txt
```

> **GPU note**: If you have an NVIDIA GPU, install the GPU-enabled TensorFlow
> build (e.g. `tensorflow[and-cuda]>=2.10`) for significantly faster training.

---

## Usage

### Quick smoke test (~1–2 min on CPU)

```bash
python experiment.py --quick
python analysis.py --skip_generation
```

This runs 1 epoch on code sizes [2, 8] for both datasets and both loss types,
then prints summary tables and saves plots to `results/`.

### Full replication (paper settings)

```bash
# Full sweep, all datasets, 30 epochs, 5 runs (can take many hours on CPU)
python experiment.py

# Or restrict to one dataset
python experiment.py --dataset mnist --epochs 30 --runs 5

# After experiment finishes, run analysis
python analysis.py
```

### Individual sweeps

```bash
# Only the main comparison sweep
python experiment.py --sweep main

# Only the n-exponent study (uses best code size from main sweep CSV)
python experiment.py --sweep n_exp
```

---

## Output

After a full run, `results/` will contain:

| File | Content |
|------|---------|
| `main_results.csv` | Per-config mean metrics (ACC, NMI, SSIM, UQI, MSE, VIF, time) |
| `n_exponent_results.csv` | ACC vs n-exponent at best code size |
| `clustering_acc_vs_code_size.png` | Replicates paper Fig. 2 |
| `ssim_vs_code_size.png` | SSIM quality vs code size |
| `n_exponent_vs_acc.png` | Visualises Table 3 |
| `generated_samples_z8_mnist.png` | Decoder samples, paper Fig. 3 |
| `generated_samples_z8_fashion_mnist.png` | Fashion-MNIST generated samples |

---

## Reproducibility

- All random seeds are set via `config.GLOBAL_SEED` (default: 42).
- N_RUNS = 5 independent runs are averaged to reduce variance.
- Xavier (Glorot) initialisation is used throughout (paper §III).
- **No results are fabricated** — all numbers come from actual training runs.

---

## Known Ambiguity: Dropout

The paper states "dropout probability 0.8" (§IV-A).  In Keras, `Dropout(rate)`
drops that fraction of units.  If "0.8" means the *keep* probability (common
in older TensorFlow 1.x notation), the Keras rate should be `1 − 0.8 = 0.2`.

By default this code treats 0.8 as the *drop rate* (rate = 0.8).
To switch interpretations, pass `dropout_is_keep_prob=True` to `build_vae()`.

---

## Citation

```
@article{abdella2020statistical,
  title   = {A Statistical Comparative Study on Image Reconstruction and
             Clustering With Novel VAE Cost Function},
  author  = {Abdella, Mohammed and Uysal, Murat},
  journal = {IEEE Access},
  year    = {2020},
  volume  = {8},
  doi     = {10.1109/ACCESS.2020.XXXXXXX}
}
```
