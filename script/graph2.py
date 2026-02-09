import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabpfn import TabPFNRegressor

# --- PATHS ---
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"
model_path = current_dir / "final_model.pth"
scaling_path = data_dir / "scaling_params_derivatives.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL SKELETON (Must match StackingHead from Step 9) ---
class StackingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = 9 # 8 SABR features + 1 TabPFN prediction
        h_dims = config.get('hidden_dims', [512, 256, 128])
        dropout = config.get('dropout', 0.05)
        
        layers = []
        prev = input_dim
        for h in h_dims:
            layers.extend([nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout)])
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, 7))

    def forward(self, x_raw, x_pfn):
        # Concatenate raw features and TabPFN prior before the head
        return self.net(torch.cat([x_raw, x_pfn], dim=1))

def run_comparison():
    # 1. Load Config & Assets
    with open(config_path, "r") as f: config = json.load(f)
    with open(scaling_path, "r") as f: scaling = json.load(f)
    df = pd.read_csv(data_dir / "sabr_with_derivatives_scaled.csv")

    # 2. Load the Trained Model weights (.pth)
    model = StackingHead(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Successfully loaded Step 9 model: {model_path}")

    # 3. Initialize Baseline TabPFN (Step 4)
    baseline_pfn = TabPFNRegressor(device='cpu')
    # Fit on a small context to define the SABR regime
    ref = df.sample(200)
    baseline_pfn.fit(ref[['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']].values, 
                     ref['volatility_scaled'].values)

    # 4. Prepare Test Smile (Varying K)
    sample = df.iloc[np.random.randint(0, len(df))]
    f_val = sample['F']
    strikes = np.linspace(0.5 * f_val, 1.5 * f_val, 100)
    
    batch_np = np.zeros((100, 8))
    cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    for i, col in enumerate(cols):
        batch_np[:, i] = sample[col]
    
    batch_np[:, 6] = strikes # Update Strikes
    batch_np[:, 7] = np.log(strikes / f_val) # Update Moneyness

    # 5. Model Inference
    # Get TabPFN Prior first
    pfn_prior_scaled = baseline_pfn.predict(batch_np).reshape(-1, 1)
    
    with torch.no_grad():
        x_raw = torch.FloatTensor(batch_np).to(device)
        x_pfn = torch.FloatTensor(pfn_prior_scaled).to(device)
        # Step 9 prediction (Stacking)
        outputs = model(x_raw, x_pfn).cpu().numpy()

    # 6. Descaling
    v_min, v_max = scaling['volatility']['min'], scaling['volatility']['max']
    vol_s4 = pfn_prior_scaled.flatten() * (v_max - v_min) + v_min
    vol_s9 = outputs[:, 0] * (v_max - v_min) + v_min
    
    # Calculate Skew (dV/dK)
    skew_s4 = np.gradient(vol_s4, strikes)
    skew_s9 = np.gradient(vol_s9, strikes)

    # --- PLOTTING ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"SABR Evolution: Baseline (Red) vs Causal Fine-tuned (Cyan)\nParams: Beta={sample['beta']:.2f}, Rho={sample['rho']:.2f}", fontsize=14)

    # Ax1: The Smile
    ax1.plot(strikes, vol_s4, 'r--', label='Step 4 (Baseline)')
    ax1.plot(strikes, vol_s9, 'cyan', lw=2.5, label='Step 9 (Causal SCM)')
    ax1.set_title("Volatility Smile")
    ax1.set_xlabel("Strike (K)")
    ax1.legend()

    # Ax2: The Skew (Greek stability)
    ax2.plot(strikes, skew_s4, 'r--', label='Step 4 Skew (Unstable)')
    ax2.plot(strikes, skew_s9, 'cyan', lw=2.5, label='Step 9 Skew (Smooth)')
    ax2.set_title("Hedging Stability ($dV/dK$)")
    ax2.set_xlabel("Strike (K)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()