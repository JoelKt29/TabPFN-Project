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
-  TabPFN captures the general shape of the volatility smile
-  Produces "jittery" predictions (high-frequency noise)
-  Numerical gradient computation (Skew = dV/dK) exhibits extreme noise, rendering the model unsuitable for stable hedging

**Verdict**: Baseline TabPFN is too noisy for real-world risk applications.

---

## Phase 2: Custom Loss Function for Greeks (Steps 5 & 6)

**Objective**: Enforce mathematical consistency between volatility and its derivatives.

**Innovation - Custom Loss with derivatives**:
Instead of standard MSE on prices alone, we optimize both the price AND its derivatives:

$$\mathcal{L}_{Sobolev} = \lambda_1 \|V_{pred} - V_{true}\| + \lambda_2 \|\nabla V_{pred} - \nabla V_{true}\|

where:
- $V$ = predicted volatility
- $\nabla V$ = derivatives w.r.t. SABR parameters (Alpha, Beta, Rho, VolVol, Strike)

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
-  Keep TabPFN frozen (preserve pretrained knowledge)
-  Fine-tune only the custom head (low computational cost)
-  Inject both raw features AND TabPFN predictions into head (dual inputs)

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



---
