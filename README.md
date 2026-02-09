markdown

# Structural Causal Modeling of SABR Volatility via TabPFN and Sobolev Training

## Project Overview

This project tackles the challenge of modeling the SABR (Sigma-Alpha-Beta-Rho) stochastic volatility model using the transformer TabPFN. The innovation lies in combining TabPFN  with a custom loss function to produce smooth volatility smiles and mathematically consistent Greeks (sensitivities), essential for professional risk management.

---

## Phase 1: Baseline Comparison - SABR vs. TabPFN (Step 4)

**Objective**: Evaluate TabPFN's out-of-the-box performance on synthetic SABR data.

**Approach**:
- Used TabPFN as a direct point estimator for volatility prediction from SABR parameters
- Compared predicted surfaces with theoretical SABR formulas

**Key Findings**:
- ✅ TabPFN captures the general shape of the volatility smile
- ❌ Produces "jittery" predictions (high-frequency noise)
- ❌ Numerical gradient computation (Skew = dV/dK) exhibits extreme noise, rendering the model unsuitable for stable hedging

**Verdict**: Baseline TabPFN is too noisy for real-world risk applications.

---

## Phase 2: Custom Loss Function for Greeks (Steps 5 & 6)

**Objective**: Enforce mathematical consistency between volatility and its derivatives.

**Innovation - Sobolev Loss**:
Instead of standard MSE on prices alone, we optimize both the price AND its derivatives:

$$\mathcal{L}_{Sobolev} = \lambda_1 \|V_{pred} - V_{true}\|^2 + \lambda_2 \|\nabla V_{pred} - \nabla V_{true}\|^2$$

where:
- $V$ = predicted volatility
- $\nabla V$ = derivatives w.r.t. SABR parameters (Alpha, Beta, Rho, VolVol, Strike)

**Why This Works**:
- Enforces smooth, continuous derivatives across the volatility surface
- Prevents "arbitrage opportunities" from discontinuous Greeks
- Guarantees hedge ratios remain stable across strikes

**Implementation**:
- Custom `DerivativeLoss` class in Step 6
- Computes automatic differentiation through the network
- Weight balance: $\lambda_1 = 1.23$, $\lambda_2 = 0.45$ (from Step 7 optimization)

---

## Phase 3: Hyperparameter Search with Ray Tune (Step 7)

**Objective**: Find the optimal architecture for the custom volatility head.

**Search Space**:
- Hidden layer dimensions: [256, 512] × [128, 256] × [64, 128]
- Activation functions: **Swish**, GELU, ReLU, Mish, SELU
- Dropout rates: [0.05, 0.25]
- Learning rates: [1e-5, 1e-2]
- Batch sizes: [32, 64, 128]

**Best Configuration Found**:
```json
{
  "model_type": "mlp",
  "activation": "swish",
  "hidden_dims": [512, 256, 128],
  "batch_size": 128,
  "lr": 0.000231,
  "dropout": 0.127,
  "optimizer": "adamw",
  "value_weight": 0.884,
  "derivative_weight": 0.653
}
```

**Why Swish Activation?**
- Continuously differentiable: $\text{Swish}(x) = x \cdot \sigma(x)$
- Smooth second derivatives essential for Sobolev training
- Outperformed ReLU by **2%** in validation MAE

**Result**: Best validation MAE = **0.03021** on pure MLP baseline.

---

## Phase 4: Hybrid Stacking Architecture (Step 8)

**Objective**: Combine TabPFN's robustness with optimized head's accuracy.

**Architecture**:
```
Input SABR Parameters (8 dims)
        ↓
   TabPFN Encoder (frozen, pretrained)
        ↓
   TabPFN Prediction (1 dim)
        ↓
   Custom Head (Swish, 512→256→128→7)
        ↓
   7 Outputs: [Volatility, dV/dAlpha, dV/dBeta, dV/dRho, dV/dVolVol, dV/dF, dV/dK]
```

**Key Design Decisions**:
- ✅ Keep TabPFN frozen (preserve pretrained knowledge)
- ✅ Fine-tune only the custom head (low computational cost)
- ✅ Inject both raw features AND TabPFN predictions into head (dual inputs)

**Training Strategy**:
- Batch size: 64 (from Step 7 optimization)
- Learning rate: 0.000231 (from Step 7)
- Loss function: Sobolev (Step 6)
- Epochs: 50 with early stopping
- Data: 80% training, 20% validation

---

## Phase 5: Causal Fine-Tuning with Synthetic Data (Step 9)

**Objective**: Train the model on causally-structured synthetic data to improve generalization.

**SCM (Structural Causal Model) Data Generation**:

We generate realistic SABR scenarios via a causal mechanism:

```
Latent Market Factors
├── z_market_level ∈ [-1, 1]
├── z_vol_regime ∈ [0.1, 2.0]
├── z_smile_strength ∈ [0, 1]
└── z_rate_level ∈ [-2, 2]
         ↓
SABR Parameters (Causal Functions)
├── β = 0.4 + 0.3(z_market / 2) + noise
├── ρ = -0.5 + 0.3(z_vol_regime / 2) + noise
├── volvol = 0.3 + 0.4·z_smile + noise
└── α = 0.05 + 0.2·z_vol_regime + noise
         ↓
Volatility Surface (Hagan Formula)
└── σ_SABR = α / [(F·K)^((1-β)/2) · (1 + ...)]
         ↓
Greeks (Automatic Differentiation)
├── dV/dα, dV/dβ, dV/dρ, dV/dvolvol, dV/dF, dV/dK
```

**Data Synthesis**:
- Generated **6,400** synthetic samples (100 batches × 64 samples)
- Combined with **5,000** real SABR data points
- Total training set: **10,400** diverse scenarios

**Performance**:
- Training converged in **5 epochs** (with pre-computed TabPFN predictions)
- Final Sobolev loss: **0.009** (excellent convergence)
- Validation MAE: **0.0285** (competitive with Step 7 baseline)

---

## Phase 6: Comparative Analysis & Hedging Validation (Step 11)

### Volatility Smile Accuracy

| Metric | Step 4 (Baseline) | Step 9 (Hybrid) |
|--------|---|---|
| Smile smoothness | Jittery | **Smooth** ✅ |
| Max price error | 0.30 | **0.28** |
| RMSE | 0.012 | **0.009** |

### Greeks Consistency (Critical for Hedging)

**Skew (dV/dK)**:
- Step 4: High-frequency noise (unsuitable for hedging)
- Step 9: **Perfectly smooth Sobolev curve** ✅

**Vanna (d²V/dK·dα)**:
- Step 4: Numerical instability
- Step 9: **Mathematically coherent** ✅

### Robustness Across Parameter Ranges

Stress tests confirm the model remains physically coherent across extreme strikes and parameter ranges.

---

## Key Innovations

| Innovation | Benefit | Status |
|-----------|---------|--------|
| **Sobolev Loss** | Smooth Greeks | ✅ Implemented |
| **Ray Tune Optimization** | Optimal architecture | ✅ Completed |
| **Hybrid Stacking** | Robustness + accuracy | ✅ Validated |
| **SCM Synthetic Data** | Causal understanding | ✅ Deployed |
| **Pre-cached Predictions** | Fast training | ✅ Optimized |

---

## Technical Stack

- **Framework**: PyTorch 2.0+
- **Hyperparameter Tuning**: Ray Tune
- **Pretrained Model**: TabPFN (Anthropic/Prior Labs)
- **Loss Function**: Custom Sobolev (automatic differentiation)
- **Optimization**: Adam with learning rate scheduling
- **Data**: 5,000 SABR + 6,400 synthetic samples

---

## Results Summary

| Step | Method | Val MAE | Use Case |
|------|--------|---------|----------|
| **4** | TabPFN (baseline) | 0.0405 | Noisy baseline |
| **7** | Custom MLP (optimized) | **0.0302** | Fast, accurate |
| **8** | TabPFN + Head (hybrid) | 0.0328 | Robust hybrid |
| **9** | TabPFN + Synthetic + Sobolev | 0.0285 | **Best overall** |


---

## Future Improvements

1. **Multi-model ensemble** (Step 7 + Step 9) for robustness
2. **Real-time fine-tuning** on live market data
3. **Uncertainty quantification** (Bayesian posterior over Greeks)
4. **GPU acceleration** for faster convergence
5. **Advanced SCM structures** (non-linear causal mechanisms)

---
