import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tabpfn import TabPFNRegressor
from step08_transfer_learning import TabPFNStackingModel

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent 
graph_dir = project_root / "graph"
graph_dir.mkdir(exist_ok=True)

data_dir = project_root / "data"
model_path = current_dir / "tabpfn_step9_causal_final.pth"
config_path = current_dir / "ray_results" / "best_config.json"
scaling_path = data_dir / "scaling_params_derivatives.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_final_graph():
    with open(config_path, 'r') as f: config = json.load(f)
    with open(scaling_path, 'r') as f: scaling_params = json.load(f)
    
    df = pd.read_csv(data_dir / 'sabr_with_derivatives_scaled.csv')
    
    K_MEAN = scaling_params['K']['mean']
    K_STD = scaling_params['K']['std']
    VOL_MEAN = scaling_params['volatility']['mean']
    VOL_STD = scaling_params['volatility']['std']

    deriv_cols = [c for c in df.columns if c.startswith('dV_') and c.endswith('_scaled')]
    y_cols = ['volatility_scaled'] + deriv_cols
    num_outputs = len(y_cols)
    
    pfn_baseline = TabPFNRegressor(device='cpu', n_estimators=4)
    idx_ref = np.random.choice(len(df), 500, replace=False)
    pfn_baseline.fit(df.iloc[idx_ref][['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']].values, 
                     df.iloc[idx_ref]['volatility_scaled'].values)

    model_s9 = TabPFNStackingModel(config, n_outputs=num_outputs).to(device)
    model_s9.load_state_dict(torch.load(model_path, map_location=device))
    model_s9.eval()

    # 3. Scenario Generation (Actual prices)
    sample = df.sample(1).iloc[0]
    f_val_scaled = sample['F']
    # Descale Forward price to generate a consistent strike range
    f_val_raw = f_val_scaled * scaling_params['F']['std'] + scaling_params['F']['mean']
    
    # Generate actual strikes
    raw_strikes = np.linspace(0.5 * f_val_raw, 1.5 * f_val_raw, 100)
    scaled_strikes = (raw_strikes - K_MEAN) / K_STD
    
    test_batch = np.zeros((100, 8))
    cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    for i, col in enumerate(cols):
        test_batch[:, i] = sample[col]
    
    test_batch[:, 6] = scaled_strikes 
    test_batch[:, 7] = np.log(raw_strikes / f_val_raw) # Log-moneyness

    # 4. Inference
    pfn_prior = pfn_baseline.predict(test_batch).reshape(-1, 1)
    with torch.no_grad():
        x_raw = torch.FloatTensor(test_batch).to(device)
        x_pfn = torch.FloatTensor(pfn_prior).to(device)
        preds_s9 = model_s9(x_raw, x_pfn).cpu().numpy()

    # 5. Descaling Output for Plotting
    pfn_vol_raw = pfn_prior * VOL_STD + VOL_MEAN
    s9_vol_raw = preds_s9[:, 0] * VOL_STD + VOL_MEAN

    # Plotting
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Final Model Analysis (Unscaled)\nRho={sample['rho']:.2f}, VolVol={sample['volvol']:.2f}", fontsize=20)

    # Volatility Smile
    axs[0, 0].plot(raw_strikes, pfn_vol_raw, 'r--', label='Step 4 Baseline', alpha=0.7)
    axs[0, 0].plot(raw_strikes, s9_vol_raw, 'cyan', lw=3, label='Step 9 Sobolev')
    axs[0, 0].set_title("Implied Volatility Smile (Actual Prices)", fontsize=14)
    axs[0, 0].set_xlabel("Strike Price (K)")
    axs[0, 0].set_ylabel("Volatility")
    axs[0, 0].legend()

    # Greeks (Keep scaled for sensitivity comparison or descale if needed)
    idx_dk = next((i for i, c in enumerate(y_cols) if 'dK' in c), 5)
    axs[0, 1].plot(raw_strikes, preds_s9[:, idx_dk], 'cyan', lw=3, label='Step 9 Skew')
    axs[0, 1].set_title("Skew Stability (dV/dK)", fontsize=14)
    axs[0, 1].set_xlabel("Strike Price (K)")

    idx_dvv = next((i for i, c in enumerate(y_cols) if 'dvolvol' in c), 3)
    axs[1, 0].plot(raw_strikes, preds_s9[:, idx_dvv], 'orange', lw=3, label='Step 9 Vanna')
    axs[1, 0].set_title("Vanna Sensitivity (dV/dvolvol)", fontsize=14)
    axs[1, 0].set_xlabel("Strike Price (K)")

    idx_drho = next((i for i, c in enumerate(y_cols) if 'drho' in c), 2)
    axs[1, 1].plot(raw_strikes, preds_s9[:, idx_drho], 'magenta', lw=3, label='Step 9 dV/drho')
    axs[1, 1].set_title("Correlation Sensitivity (dV/drho)", fontsize=14)
    axs[1, 1].set_xlabel("Strike Price (K)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    generate_final_graph()