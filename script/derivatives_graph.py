import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tabpfn import TabPFNRegressor
import jax.numpy as jnp

# IMPORT DU MOTEUR JAX
from step02_jax_sabr import compute_sabr_with_jax

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
csv_path = data_dir / "sabr_with_derivatives_scaled.csv" 
scaling_path = data_dir / "scaling_params_derivatives.json"

def show_tabpfn_derivative_noise():
    print("⏳ Chargement des données et entraînement de TabPFN...")
    df = pd.read_csv(csv_path)
    with open(scaling_path, 'r') as f:
        scaling_params = json.load(f)
    
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_col = 'volatility_scaled' if 'volatility_scaled' in df.columns else 'SABR_volatility'
    
    pfn_baseline = TabPFNRegressor(device='cpu', n_estimators=4)
    train_idx = np.random.choice(len(df), min(1000, len(df)), replace=False)
    pfn_baseline.fit(df.iloc[train_idx][feature_cols].values, 
                     df.iloc[train_idx][target_col].values)

    print("📈 Génération du scénario de test...")
    
    # =================================================================
    # 1. UTILITAIRES DE MISE À L'ÉCHELLE
    # =================================================================
    def get_min(col):
        if 'X_min' in scaling_params: return scaling_params['X_min'][col]
        return scaling_params['features'][col]['min']

    def get_max(col):
        if 'X_max' in scaling_params: return scaling_params['X_max'][col]
        return scaling_params['features'][col]['max']

    def scale_to_minus1_1(raw_val, col_name):
        c_min, c_max = get_min(col_name), get_max(col_name)
        val_0_1 = (raw_val - c_min) / (c_max - c_min)
        return (val_0_1 * 2.0) - 1.0

    def unscale_from_minus1_1(scaled_val, col_name):
        c_min, c_max = get_min(col_name), get_max(col_name)
        val_0_1 = (scaled_val + 1.0) / 2.0
        return val_0_1 * (c_max - c_min) + c_min

    # =================================================================
    # 2. CAPTURE D'UN SCÉNARIO COHÉRENT (Ligne 50)
    # =================================================================
    sample_row = df.iloc[50]
    f_raw = unscale_from_minus1_1(sample_row['F'], 'F')
    alpha_raw = unscale_from_minus1_1(sample_row['alpha'], 'alpha')
    beta_raw = unscale_from_minus1_1(sample_row['beta'], 'beta')
    rho_raw = unscale_from_minus1_1(sample_row['rho'], 'rho')
    volvol_raw = unscale_from_minus1_1(sample_row['volvol'], 'volvol')
    t_raw = 1.0 

    k_raw = np.linspace(0.5 * f_raw, 1.5 * f_raw, 200)

    # =================================================================
    # 3. VÉRITÉ MATHÉMATIQUE AVEC JAX (Lognormal)
    # =================================================================
    true_vols, true_grads = compute_sabr_with_jax(
        jnp.array(k_raw), 
        jnp.full(200, f_raw), 
        jnp.full(200, t_raw), 
        jnp.full(200, alpha_raw), 
        jnp.full(200, beta_raw), 
        jnp.full(200, rho_raw), 
        jnp.full(200, volvol_raw)
    )
    true_derivative = np.array(true_grads['dV_dK'])

    # =================================================================
    # 4. PRÉDICTIONS TabPFN (Normal) ET CALCUL DES DÉRIVÉES
    # =================================================================
    test_batch = np.zeros((200, 8))
    for i, col in enumerate(feature_cols):
        test_batch[:, i] = sample_row[col] 
        
    test_batch[:, feature_cols.index('K')] = scale_to_minus1_1(k_raw, 'K')
    test_batch[:, feature_cols.index('log_moneyness')] = scale_to_minus1_1(np.log(k_raw / f_raw), 'log_moneyness')

    pfn_preds_scaled = pfn_baseline.predict(test_batch)
    
    y_min = scaling_params.get('y_min', 0.0)
    y_max = scaling_params.get('y_max', 1.0)
    pfn_preds_raw = pfn_preds_scaled * (y_max - y_min) + y_min

    pfn_derivative = np.gradient(pfn_preds_raw, k_raw)

    # =================================================================
    # 5. Z-SCORE ALIGNMENT (La Magie pour comparer Normal et Lognormal)
    # =================================================================
    # On centre et on réduit les deux courbes pour qu'elles se superposent
    true_derivative_z = (true_derivative - np.mean(true_derivative)) / (np.std(true_derivative) + 1e-8)
    pfn_derivative_z = (pfn_derivative - np.mean(pfn_derivative)) / (np.std(pfn_derivative) + 1e-8)

    # =================================================================
    # 6. LE GRAPHIQUE FINAL Z-SCORE
    # =================================================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(k_raw, pfn_derivative_z, color='red', lw=1.5, alpha=0.8, label='TabPFN Derivative')
    ax.plot(k_raw, true_derivative_z, color='cyan', lw=3, label='JAX Skew ')
    
    ax.set_title(" Gradient issue ", fontsize=16)
    ax.set_xlabel(f"Strike K (Forward = {f_raw:.4f})")
    ax.set_ylabel("Sensivity (dV/dK)")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)

    graph_dir = current_dir.parent / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    save_path = graph_dir / "SABR_Derivatives.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    show_tabpfn_derivative_noise()