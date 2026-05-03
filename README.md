# SABR Volatility Surface Calibration via TabPFN & Neural Network Stacking

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![JAX](https://img.shields.io/badge/JAX-Enabled-purple)

---

## 📌 Overview

This repository presents an advanced machine learning approach to calibrate the **SABR volatility surface**. The project leverages **TabPFN** (a powerful In-Context Learning model) as a baseline and significantly enhances its performance—specifically its geometric stability and derivative accuracy—through **Neural Network Stacking** and a **Sobolev Loss function**.

The primary goal is to reconstruct an **arbitrage-free, smooth volatility surface** with highly accurate Greeks (derivatives like $\partial V/\partial K$), overcoming the numerical instability typically associated with tree-based or pure In-Context Learning models.

---

## 🚀 Key Innovations

### 1. Hybrid Data Generation (Sobol + ATM Refinement)

* SABR parameters are generated using **Sobol sequences** for uniform state-space coverage.
* To capture the critical At-The-Money (ATM) singularity where curvature is highest, we implemented a **Gaussian Refinement Mesh** ($K \approx F$).

### 2. JAX-Powered Sobolev Loss

* Standard ML models often produce noisy derivatives, rendering them unusable for risk management (Delta/Vega hedging).
* We implemented a custom **Sobolev Loss** using JAX auto-differentiation, forcing the neural network to minimize errors not only on the volatility values ($V$) but also on the exact SABR gradients.

### 3. Neural Network Stacking (Ray Tune Optimized)

* TabPFN acts as a high-level "Oracle" feature extractor.
* A Multilayer Perceptron (MLP), hyperparameter-tuned via **Ray Tune**, takes both the raw SABR parameters and the TabPFN prediction to output the final, smoothed volatility and its derivatives.

---

## 📁 Repository Structure

```
TabPFN-Project/
├── data/                       
│   ├── sabr_hybrid_mesh_scaled.csv
│   └── scaling_params_derivatives.json
├── script/                     
│   ├── all_the_steps
│   └── ray_results.json
├── ray_results/                
├── graph/                      
└── tabpfn_step9_causal_final.pth
```

---

## 📊 Results & Performance

By comparing the baseline TabPFN with our final Sobolev-regularized Stacking MLP, we achieved significant improvements, particularly regarding the stability of the Greeks:

* **Volatility Prediction:** High accuracy maintained across the surface
* **Skew ($\partial V/\partial K$) Stability:** The Stacking architecture drastically reduced the "gradient noise" inherent to the baseline model, providing smooth and financially consistent derivatives crucial for hedging

*(Tip: add a comparison plot here — ça fait très pro en entretien)*

---

## ⚙️ Usage

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/JoelKt29/TabPFN-Project.git
cd TabPFN-Project
pip install -r requirements.txt
```

---

### 2. Running the Comparison

```bash
python script/comparision.py
```

---

### 3. Training the Stacking Model

```bash
python script/step08_transfer_learning.py
```

---

## 🛠️ Technology Stack

* **Machine Learning:** PyTorch, TabPFN
* **Scientific Computing & Gradients:** JAX, NumPy
* **Hyperparameter Optimization:** Ray Tune
* **Data Processing:** Pandas, SciPy (Sobol sequences)

---

## 🤝 Acknowledgements

This project was developed as part of an applied quantitative finance research initiative, focusing on bridging the gap between state-of-the-art Deep Learning and rigorous mathematical finance.

---

## 👤 Author

**Joël Khayat**

---
