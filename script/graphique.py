import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tabpfn import TabPFNRegressor
# Importing the architecture from Step 8
from step08_transfer_learning import TabPFNStackingModel

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent 
graph_dir = project_root / "graph"
graph_dir.mkdir(exist_ok=True)

data_dir = project_root / "data"
model_path = current_dir / "tabpfn_step9_causal_final.pth"
config_path = current_dir / "ray_results" / "best_config.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_final_report():
    print(f"🚀 Generating report on {device}...")

    with open(config_path, 'r') as f: config = json.load(f)
    df = pd.read_csv(data_dir / 'sabr_with_derivatives_scaled.csv')
    
    # In a real scenario, load your scaler.joblib here. 
    # Hardcoded stats for Strike (K) based on standard SABR ranges to demonstrate descaling:
    K_MEAN, K_STD = 0.5, 0.2 
    VOL_MEAN, VOL_STD = 0.3, 0.1

    deriv_cols = [c for c in df.columns if c.startswith('dV_') and c.endswith('_scaled')]
    y_cols = ['volatility_scaled'] + deriv_cols
    num_outputs = len(y_cols)
    
    # Initialize Step 4 Baseline
    pfn_baseline = TabPFNRegressor(device='cpu', n_estimators=4)
    idx_ref = np.random.choice(len(df), 500, replace=False)
    pfn_baseline.fit(df.iloc[idx_ref][['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']].values, 
                     df.iloc[idx_ref]['volatility_scaled'].values)

    # Load Step 9 Fine-tuned Model
    model_s9 = TabPFNStackingModel(config, n_outputs=num_outputs).to(device)
    model_s9.load_state_dict(torch.load(model_path, map_location=device))
    model_s9.eval()

    # Scenario Generation
    sample = df.sample(1).iloc[0]
    f_val = sample['F']
    
    # Descaled x-axis (Actual Strikes)
    raw_strikes = np.linspace(0.2 * f_val, 1.8 * f_val, 100)
    # Scaling strikes for model input
    scaled_strikes = (raw_strikes - K_MEAN) / K_STD
    
    test_batch = np.zeros((100, 8))
    cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    for i, col in enumerate(cols):
        test_batch[:, i] = sample[col]
    
    test_batch[:, 6] = scaled_strikes 
    test_batch[:, 7] = np.log(raw_strikes / f_val) # Log-moneyness remains unscaled

    # Inference
    pfn_prior = pfn_baseline.predict(test_batch).reshape(-1, 1)
    with torch.no_grad():
        x_raw = torch.FloatTensor(test_batch).to(device)
        x_pfn = torch.FloatTensor(pfn_prior).to(device)
        preds_s9 = model_s9(x_raw, x_pfn).cpu().numpy()

    # Descaling predictions for Y-axis
    pfn_prior_unscaled = pfn_prior * VOL_STD + VOL_MEAN
    preds_s9_unscaled = preds_s9[:, 0] * VOL_STD + VOL_MEAN

    # Numerical Gradients for Step 4
    skew_s4 = np.gradient(pfn_prior_unscaled.flatten(), raw_strikes)
    
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Sobolev (Step 9) vs Baseline (Step 4)\nParams: Rho={sample['rho']:.2f}, VolVol={sample['volvol']:.2f}", fontsize=20)

    # Volatility Smile
    axs[0, 0].plot(raw_strikes, pfn_prior_unscaled, 'r--', label='Step 4: Baseline', alpha=0.7)
    axs[0, 0].plot(raw_strikes, preds_s9_unscaled, 'cyan', lw=3, label='Step 9: Fine-tuned')
    axs[0, 0].set_title("Implied Volatility Smile", fontsize=14)
    axs[0, 0].set_ylabel("Volatility (Unscaled)")
    axs[0, 0].legend()

    # Skew (dV/dK)
    idx_dk = next((i for i, c in enumerate(y_cols) if 'dK' in c), 5)
    axs[0, 1].plot(raw_strikes, skew_s4, 'r--', label='Step 4: Numerical Skew', alpha=0.7)
    axs[0, 1].plot(raw_strikes, preds_s9[:, idx_dk], 'cyan', lw=3, label='Step 9: Sobolev Skew')
    axs[0, 1].set_title("Hedging Stability (dV/dK)", fontsize=14)
    axs[0, 1].legend()

    # Vanna (dV/dvolvol)
    idx_dvv = next((i for i, c in enumerate(y_cols) if 'dvolvol' in c), 3)
    axs[1, 0].plot(raw_strikes, preds_s9[:, idx_dvv], 'orange', lw=3, label='Step 9: Vanna')
    axs[1, 0].set_title("Vanna Stability (dV/dvolvol)", fontsize=14)
    axs[1, 0].set_xlabel("Strike (Actual K)")
    axs[1, 0].legend()

    # dV/drho
    idx_drho = next((i for i, c in enumerate(y_cols) if 'drho' in c), 2)
    axs[1, 1].plot(raw_strikes, preds_s9[:, idx_drho], 'magenta', lw=3, label='Step 9: dV/drho')
    axs[1, 1].set_title("Correlation Sensitivity (dV/drho)", fontsize=14)
    axs[1, 1].set_xlabel("Strike (Actual K)")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(graph_dir / f"report_rho_{sample['rho']:.2f}.png")
    plt.show()

if __name__ == "__main__":
    generate_final_report()