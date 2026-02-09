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

def unscale(z, params):
    return z * (params['max'] - params['min']) + params['min']

def generate_final_report():
    with open(config_path, 'r') as f: config = json.load(f)
    with open(scaling_path, 'r') as f: scaling_params = json.load(f)
    
    df = pd.read_csv(data_dir / 'sabr_with_derivatives_scaled.csv')
    
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    idx_map = {name: i for i, name in enumerate(target_cols)}
    num_outputs = len(target_cols)

    model_s9 = TabPFNStackingModel(config, n_outputs=num_outputs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    fixed_state_dict = {k.replace('net.', 'head.'): v for k, v in state_dict.items()}
    model_s9.load_state_dict(fixed_state_dict)
    model_s9.eval()

    sample = df.sample(1).iloc[0]
    f_raw = unscale(sample['F'], scaling_params['features']['F'])
    
    raw_strikes = np.linspace(0.4 * f_raw, 1.6 * f_raw, 100)
    k_params = scaling_params['features']['K']
    scaled_strikes = (raw_strikes - k_params['min']) / (k_params['max'] - k_params['min'])
    
    test_batch = np.zeros((100, 8))
    feature_names = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    for i, col in enumerate(feature_names):
        test_batch[:, i] = sample[col]
    
    test_batch[:, 6] = scaled_strikes 
    test_batch[:, 7] = np.log(raw_strikes / f_raw)

    pfn_baseline = TabPFNRegressor(device='cpu', n_estimators=4)
    ref_idx = np.random.choice(len(df), 500, replace=False)
    pfn_baseline.fit(df.iloc[ref_idx][feature_names].values, df.iloc[ref_idx]['volatility_scaled'].values)
    
    pfn_prior = pfn_baseline.predict(test_batch).reshape(-1, 1)

    with torch.no_grad():
        x_raw = torch.FloatTensor(test_batch).to(device)
        x_pfn = torch.FloatTensor(pfn_prior).to(device)
        preds_s9 = model_s9(x_raw, x_pfn).cpu().numpy()

    vol_params = scaling_params['volatility']
    pfn_vol_raw = unscale(pfn_prior, vol_params).flatten()
    s9_vol_raw = unscale(preds_s9[:, 0], vol_params)

    baseline_skew_num = np.gradient(pfn_vol_raw, raw_strikes)

    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Top Plots Comparison (Real Units)\nRho: {sample['rho']:.2f} | VolVol: {sample['volvol']:.2f}", fontsize=16)

    # 1. Volatility Smile
    axs[0].plot(raw_strikes, pfn_vol_raw, 'r--', label='Step 4: Baseline (Numerical)')
    axs[0].plot(raw_strikes, s9_vol_raw, 'cyan', lw=3, label='Step 9: Sobolev (Causal)')
    axs[0].axvline(f_raw, color='white', linestyle=':', alpha=0.5, label='ATM')
    axs[0].set_title("Implied Volatility Smile")
    axs[0].set_ylabel("Volatility")
    axs[0].set_xlabel("Strike Price")
    axs[0].legend()

    # 2. Skew 
    axs[1].plot(raw_strikes, baseline_skew_num, 'r--', label='Step 4: Numerical Gradient', alpha=0.6)
    axs[1].plot(raw_strikes, preds_s9[:, idx_map['dV_dK_scaled']], 'cyan', lw=3, label='Step 9: Sobolev Prediction')
    axs[1].set_title("Skew Stability (dV/dK)")
    axs[1].set_xlabel("Strike Price")
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = graph_dir / f"top_comparison_rho_{sample['rho']:.2f}.png"
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    generate_final_report()