import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tabpfn import TabPFNRegressor


current_dir = Path(__file__).resolve().parent
data_dir = Path("data")
csv_path = data_dir / "sabr_with_derivatives_scaled.csv"
scaling_path = data_dir / "scaling_params_derivatives.json"

def show_tabpfn_derivative_noise():
    df = pd.read_csv(csv_path)
    with open(scaling_path, 'r') as f:
        scaling_params = json.load(f)
    
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    pfn_baseline = TabPFNRegressor(device='cpu', n_estimators=4)
    
    train_idx = np.random.choice(len(df), 1000, replace=False)
    pfn_baseline.fit(df.iloc[train_idx][feature_cols].values, 
                     df.iloc[train_idx]['volatility_scaled'].values)

    sample = df.sample(1).iloc[0]
    f_raw = sample['F'] * (scaling_params['features']['F']['max'] - scaling_params['features']['F']['min']) + scaling_params['features']['F']['min']
    k_raw = np.linspace(0.4 * f_raw, 1.6 * f_raw, 200)
    k_min, k_max = scaling_params['features']['K']['min'], scaling_params['features']['K']['max']
    k_scaled = (k_raw - k_min) / (k_max - k_min)
    
    test_batch = np.zeros((200, 8))
    for i, col in enumerate(feature_cols):
        test_batch[:, i] = sample[col]
    test_batch[:, 6] = k_scaled # Update K
    test_batch[:, 7] = np.log(k_raw / f_raw) # Update log-moneyness

    pfn_preds = pfn_baseline.predict(test_batch)
    pfn_derivative = np.gradient(pfn_preds, k_scaled)


    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(k_scaled, pfn_derivative, color='red', lw=1.5, label='TabPFN  Derivative ')
    ax.axhline(sample['dV_dK_scaled'], color='cyan', lw=3, label='True Derivatives')
    ax.set_title("Why TabPFN fails at Greeks (Numerical Gradient Noise)", fontsize=16)
    ax.set_xlabel("Scaled Strike (K)")
    ax.set_ylabel("dV/dK")
    ax.legend()

        
    graph_dir = current_dir.parent / "graph"
    save_path = graph_dir / "derivatives_not_working.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    show_tabpfn_derivative_noise()