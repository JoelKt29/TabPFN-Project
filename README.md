# SABR TabPFN Fine-Tuning Project - Phase 2

## 📋 Project Overview

This project implements advanced fine-tuning of TabPFN for SABR (Stochastic Alpha Beta Rho) volatility surface modeling in finance. We're now in **Phase 2**, which focuses on:

1. ✅ Computing SABR derivatives (Greeks) - **Priority**
2. ✅ Custom loss functions incorporating derivatives
3. ✅ Modern activation functions (Swish, Mish, GELU, SELU)
4. ✅ Hyperparameter optimization with Ray Tune
5. ✅ Architecture experimentation

### Current Achievement
- **MAE: 5e-5** (Target was 1e-4) ✨

---

## 📁 File Structure

```
├── Phase 1 Files (Your original work)
│   ├── base_sabr.py                    # SABR base classes
│   ├── hagan_2002_lognormal_sabr.py   # Hagan SABR implementation
│   ├── Statap2.py                      # Dataset generation
│   └── test_tabpfn.py                  # TabPFN baseline testing
│
├── Phase 2 Files (New implementation)
│   ├── sabr_derivatives.py             # Compute SABR Greeks (derivatives)
│   ├── custom_losses.py                # Advanced loss functions
│   ├── modified_architectures.py       # Custom architectures with new activations
│   ├── ray_tune_search.py              # Automated hyperparameter search
│   └── train_sabr_model.py             # Complete training pipeline
│
└── Generated Data
    ├── sabr_data_recovery.csv          # Your original dataset
    ├── scaling_params_recovery.json    # Scaling parameters
    └── sabr_data_with_greeks.csv       # New dataset with derivatives
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install numpy pandas scikit-learn torch tqdm matplotlib

# For TabPFN (if not already installed)
pip install tabpfn

# For Ray Tune (optional but recommended)
pip install "ray[tune]" optuna

# For PyTorch with CUDA (if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Generate Dataset with Derivatives (Priority!)

As Peter emphasized, **derivatives are the priority**:

```bash
python sabr_derivatives.py
```

This will:
- Generate SABR data with all Greeks (dV/dF, dV/dK, dV/dBeta, dV/dRho, dV/dVolvol)
- Include optional second-order derivatives (d²V/dF², d²V/dK²)
- Save to `sabr_data_with_greeks.csv`

### Step 3: Train Model with Custom Architecture

```bash
python train_sabr_model.py
```

This will:
- Train a custom transformer with Mish activation
- Use both values AND derivatives in the loss
- Save best model and training plots
- Report comprehensive metrics

### Step 4: Run Hyperparameter Search with Ray Tune

```bash
python ray_tune_search.py \
    --data sabr_data_with_greeks.csv \
    --samples 50 \
    --epochs 50 \
    --cpus 2.0
```

This will:
- Test different activation functions (Swish, Mish, GELU, SELU)
- Explore different architectures
- Automatically find the best configuration
- Save results to `./ray_results/`

---

## 🔬 Key Features Implemented

### 1. SABR Derivatives (Greeks)

The `sabr_derivatives.py` module computes:

**First-order sensitivities:**
- `dV/dF`: Sensitivity to forward rate (delta-like)
- `dV/dK`: Sensitivity to strike
- `dV/dBeta`: Sensitivity to beta parameter
- `dV/dRho`: Sensitivity to rho parameter
- `dV/dVolvol`: Sensitivity to vol-of-vol (vega-like)
- `dV/dVatm`: Sensitivity to ATM volatility

**Second-order sensitivities (optional):**
- `d²V/dF²`: Gamma-like (curvature in forward)
- `d²V/dK²`: Convexity in strike

**Example usage:**
```python
from sabr_derivatives import SABRGreeks, generate_dataset_with_greeks

# Generate data with Greeks
df = generate_dataset_with_greeks(
    num_samples=5000,
    num_strikes=8,
    include_second_order=True
)

# Or compute Greeks for specific parameters
calculator = SABRGreeks()
greeks = calculator.compute_all_greeks(
    f=0.05, k=0.055, t=1.0, v_atm_n=0.01,
    beta=0.5, rho=0.0, volvol=0.2
)
print(greeks)
```

### 2. Advanced Loss Functions

The `custom_losses.py` module provides:

**SABRDerivativeLoss**: Penalizes errors in both values AND derivatives
```python
from custom_losses import SABRDerivativeLoss

loss_fn = SABRDerivativeLoss(
    value_weight=1.0,      # Weight for volatility errors
    derivative_weight=0.5   # Weight for Greek errors
)
```

**WeightedMAELoss**: Puts more weight on ATM strikes
```python
from custom_losses import WeightedMAELoss, atm_weight_function

loss_fn = WeightedMAELoss(moneyness_weight_fn=atm_weight_function)
```

**HuberLossWithDerivatives**: Robust to outliers
```python
from custom_losses import HuberLossWithDerivatives

loss_fn = HuberLossWithDerivatives(delta=0.01, derivative_weight=0.3)
```

### 3. Modern Activation Functions

The `modified_architectures.py` module implements:

- **Swish** (SiLU): `f(x) = x * sigmoid(x)` - Self-gated, smooth
- **Mish**: `f(x) = x * tanh(softplus(x))` - Often outperforms ReLU
- **GELU**: Used in BERT/GPT - Smooth ReLU approximation
- **SELU**: Self-normalizing - Good for deep networks

**Example usage:**
```python
from modified_architectures import CustomTabularTransformer

model = CustomTabularTransformer(
    input_dim=10,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    activation='mish',  # Try: 'swish', 'gelu', 'selu'
    use_mlp_head=True
)
```

### 4. Hyperparameter Search with Ray Tune

The `ray_tune_search.py` module automates:

- Activation function selection
- Architecture depth/width optimization
- Learning rate tuning
- Loss function configuration

**Search space includes:**
```python
{
    'activation': ['swish', 'mish', 'gelu', 'selu'],
    'model_type': ['transformer', 'feedforward'],
    'd_model': [128, 256, 512],
    'num_layers': [2, 3, 4, 6],
    'lr': loguniform(1e-5, 1e-2),
    'loss_type': ['mae', 'huber', 'derivative'],
    ...
}
```

---

## 📊 Expected Results

### Baseline (Your Phase 1)
- Model: TabPFN
- MAE: **5e-5** (excellent!)
- Features: 8
- Data: 5000 samples

### Phase 2 Targets
- **With Derivatives**: MAE < 5e-5 for both values AND Greeks
- **Architecture**: Find activation that beats baseline
- **Generalization**: Better performance on out-of-distribution strikes

---

## 🎯 Next Steps (Following Peter's Guidance)

### Immediate Priority: Derivatives ✅
1. ✅ Generate data with Greeks using `sabr_derivatives.py`
2. ⏳ Train with derivative loss using `train_sabr_model.py`
3. ⏳ Compare MAE on values vs Greeks

### Architecture Experimentation
1. ⏳ Test all activation functions (Swish, Mish, GELU, SELU)
2. ⏳ Use Ray Tune for systematic search
3. ⏳ Compare transformer vs feedforward architectures

### Advanced (Optional)
1. ⏳ Generate synthetic financial data using causal graphs (as in TabPFN paper)
2. ⏳ Implement more complex derivative losses
3. ⏳ Test on real market data

---

## 💡 Tips for Google Colab

If you don't have local GPU access:

1. **Upload files to Colab:**
```python
from google.colab import files
uploaded = files.upload()  # Upload your .py files
```

2. **Install dependencies:**
```python
!pip install torch tabpfn "ray[tune]" optuna
```

3. **Use GPU:**
```python
# Runtime > Change runtime type > GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

4. **Run training:**
```python
!python train_sabr_model.py
```

---

## 📈 Monitoring Training

The training script automatically:
- Saves best model to `./checkpoints/best_model.pt`
- Generates evaluation plots to `./checkpoints/evaluation_plots.png`
- Reports metrics every 10 epochs
- Uses early stopping (patience=20)

**Example output:**
```
Epoch 10/100 | Train Loss: 0.000123 | Val Loss: 0.000089 | Val MAE: 0.000045 | LR: 1.00e-03
Epoch 20/100 | Train Loss: 0.000067 | Val Loss: 0.000051 | Val MAE: 0.000032 | LR: 1.00e-03
...
```

---

## 🔍 Understanding the Code

### Derivative Computation (Finite Differences)

```python
# Example: Computing dV/dF
eps = 1e-6
sabr_base = SABR(f=f, ...)
sabr_perturbed = SABR(f=f+eps, ...)

v_base = sabr_base.normal_vol(k)
v_perturbed = sabr_perturbed.normal_vol(k)

dV_dF = (v_perturbed - v_base) / eps
```

### Custom Loss with Derivatives

```python
# Loss = α * |vol_pred - vol_true| + β * |greeks_pred - greeks_true|
total_loss = value_weight * vol_error + derivative_weight * greek_error
```

### Ray Tune Integration

```python
# Define training function
def train_model(config):
    model = create_model(config)
    for epoch in range(config['num_epochs']):
        ...
        train.report({'val_mae': val_mae})  # Report to Ray

# Run search
tuner = tune.Tuner(train_model, param_space=search_space)
results = tuner.fit()
```

---

## 🐛 Troubleshooting

### Issue: "Ray not installed"
```bash
pip install "ray[tune]" optuna
```

### Issue: "CUDA out of memory"
- Reduce `batch_size` in config
- Reduce `d_model` or `num_layers`
- Use CPU: `device='cpu'`

### Issue: "Greeks not found in data"
- Run `sabr_derivatives.py` first
- Make sure output CSV contains columns like `dV_dF`, `dV_dK`, etc.

### Issue: "Loss not decreasing"
- Try different activation: `activation='gelu'`
- Lower learning rate: `lr=1e-4`
- Check data scaling
- Increase model capacity

---

## 📚 References

1. **SABR Model**: Hagan et al. (2002) - "Managing Smile Risk"
2. **TabPFN**: Hollmann et al. (2022) - "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
3. **Activation Functions**:
   - Swish: Ramachandran et al. (2017)
   - Mish: Misra (2019)
   - GELU: Hendrycks & Gimpel (2016)
4. **Ray Tune**: Liaw et al. (2018) - "Tune: A Research Platform for Distributed Model Selection and Training"

---

## 📧 Questions?

If you have questions:
1. Check Peter's latest guidance
2. Review error messages carefully
3. Test with smaller datasets first
4. Use verbose mode for debugging

---

## ✅ Checklist

Phase 2 Progress:
- [x] Generate SABR data with derivatives
- [ ] Train model with derivative loss
- [ ] Test different activation functions
- [ ] Run Ray Tune hyperparameter search
- [ ] Compare results with baseline
- [ ] Report findings to Peter

---

**Good luck with Phase 2! 🚀**

Remember: Start with derivatives (Peter's priority), then experiment with architectures systematically using Ray Tune.