# SABR Volatility Surface Calibration via TabPFN & Neural Network Stacking

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![JAX](https://img.shields.io/badge/JAX-enabled-purple)
![License](https://img.shields.io/badge/License-MIT-green)

A machine-learning pipeline to calibrate the **SABR volatility surface**. It uses
**TabPFN** (an in-context learning tabular model) as a baseline "oracle" and improves
its **geometric stability and derivative accuracy** through **neural-network stacking**
trained with a **Sobolev loss**, so that the predicted Greeks (e.g. `∂V/∂K`) remain
smooth and usable for hedging.

---

## Key ideas

**1. Hybrid data generation (Sobol + ATM refinement).** SABR parameters are sampled with
**Sobol sequences** for uniform state-space coverage, then refined with a Gaussian mesh
around the at-the-money region (`K ≈ F`) where curvature is highest.

**2. JAX-powered Sobolev loss.** A custom loss built on JAX auto-differentiation penalizes
errors on both the volatility values and the exact SABR gradients, removing the gradient
noise typical of plain regressors.

**3. Stacking (Ray Tune optimized).** TabPFN acts as a feature extractor; an MLP — tuned
with **Ray Tune** — takes the raw SABR parameters plus the TabPFN prediction and outputs
the final smoothed volatility and its derivative.

---

## Repository structure

```
TabPFN-SABR/
├── src/                         # all source code (flat, importable modules)
│   ├── step01_base_sabr.py          # SABR base classes
│   ├── step02_hagan_lognormal.py    # Hagan (2002) lognormal expansion
│   ├── step03_market_data.py        # market-data construction
│   ├── jax_sabr.py                  # JAX SABR engine (values + gradients)
│   ├── hybrid_meshes.py             # Sobol + ATM hybrid mesh generator
│   ├── adaptive_meshes.py           # adaptive dataset generator
│   ├── step06_sobol_dataset.py      # Sobol dataset with derivatives
│   ├── build_dataset.py             # dataset assembly / augmentation
│   ├── scaling.py                   # feature/target scaling
│   ├── sobolev_loss.py              # Sobolev (value + derivative) loss
│   ├── step04_tabpfn_baseline.py    # TabPFN baseline
│   ├── step07_ray_search.py         # Ray Tune hyperparameter search
│   ├── step08_stacking_train.py     # stacking MLP training
│   ├── step09_final_proof.py        # final 3D surface verification
│   ├── compare_models.py            # baseline vs. final comparison
│   └── fig_*.py                     # figure-generation scripts
├── data/                        # datasets (CSV) + scaling params (JSON)
├── models/                      # trained weights (.pth)
├── figures/                     # generated plots
├── reports/                     # report_results.pdf
├── requirements.txt
└── LICENSE
```

> **Large files.** `data/`, `models/` and `figures/` contain large binaries. For a clean
> clone, consider tracking them with [Git LFS](https://git-lfs.com/) or attaching them to a
> GitHub Release rather than committing them directly.

---

## Installation

```bash
git clone https://github.com/<your-account>/TabPFN-SABR.git
cd TabPFN-SABR
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

> `jax` and `ray` are required by the pipeline but were missing from the original
> environment freeze. Install the versions matching your platform/CUDA setup and pin them
> in `requirements.txt`.

All scripts resolve paths relative to the project root, so run them from `src/`:

```bash
cd src
```

---

## Usage

**1. Generate the dataset** (Sobol + ATM hybrid mesh, with JAX-computed derivatives):

```bash
python hybrid_meshes.py
python step06_sobol_dataset.py
python scaling.py
```

**2. Tune and train the stacking model:**

```bash
python step07_ray_search.py        # Ray Tune hyperparameter search
python step08_stacking_train.py    # train the final stacking MLP
```

**3. Evaluate (baseline vs. final) and produce the proof surface:**

```bash
python compare_models.py
python step09_final_proof.py
```

Figures can be regenerated with the `fig_*.py` scripts.

---

## Results

Compared with the baseline TabPFN, the Sobolev-regularized stacking MLP keeps high accuracy
on the volatility values while drastically reducing gradient noise on the skew (`∂V/∂K`),
yielding smooth, financially consistent derivatives suitable for hedging. See
`reports/report_results.pdf` for the full analysis.

---

## Tech stack

- **ML:** PyTorch, TabPFN
- **Scientific computing & gradients:** JAX, NumPy, SciPy
- **Hyperparameter optimization:** Ray Tune
- **Finance:** pysabr (Hagan expansion)
- **Data & viz:** pandas, Matplotlib, seaborn, Plotly

---

## Authors

- Joël Khayat
- Benjamin Benisti

Developed as part of an applied quantitative-finance research project bridging deep learning
and mathematical finance.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
