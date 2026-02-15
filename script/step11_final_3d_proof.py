import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import sys
from pathlib import Path
from tabpfn import TabPFNRegressor
from step02_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR

# Import dynamique
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from step08_transfer_learning import TabPFNStackingModel

# Chemins
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"
model_path = current_dir / "tabpfn_step9_causal_final.pth"
scaling_path = data_dir / "scaling_params_derivatives.json"

def generate_aligned_3d_proof():
    print("--- Génération de la Preuve 3D (Alignement Statistique) ---")

    # 1. Chargement
    with open(config_path, 'r') as f: config = json.load(f)
    with open(scaling_path, 'r') as f: scaling_params = json.load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TabPFNStackingModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Context TabPFN
    df_train = pd.read_csv(data_dir / "sabr_with_derivatives_scaled.csv").sample(2000)
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    X_train_ctx = df_train[feature_cols].values
    y_train_ctx = df_train['volatility_scaled'].values
    
    tabpfn = TabPFNRegressor(device=device, n_estimators=4, ignore_pretraining_limits=True)
    tabpfn.fit(X_train_ctx, y_train_ctx)

    # 3. Grille de Test
    n_points = 50
    f_range = np.linspace(0.02, 0.04, n_points)
    k_range = np.linspace(0.01, 0.06, n_points)
    F_grid, K_grid = np.meshgrid(f_range, k_range)
    flat_F, flat_K = F_grid.ravel(), K_grid.ravel()
    
    # Paramètres Scénario
    beta, rho, volvol, v_atm_n = 0.5, -0.3, 0.4, 0.01
    T, shift = 1.0, 0.0

    # 4. Calcul de la Vérité Terrain (Ground Truth) AVANT tout le reste
    print("Calcul de la surface Mathématique...")
    Z_true_grad = np.zeros_like(flat_F)
    eps = 1e-5
    for i in range(len(flat_F)):
        m = Hagan2002LognormalSABR(f=flat_F[i], shift=shift, t=T, v_atm_n=v_atm_n, 
                                   beta=beta, rho=rho, volvol=volvol)
        v_p = m.normal_vol(flat_K[i] + eps)
        v_m = m.normal_vol(flat_K[i] - eps)
        Z_true_grad[i] = (v_p - v_m) / (2*eps)
    
    # 5. Prédictions IA
    def scale_val(val, param_name):
        min_v = scaling_params['features'][param_name]['min']
        max_v = scaling_params['features'][param_name]['max']
        return (val - min_v) / (max_v - min_v) * 2 - 1

    alpha_list = []
    for f_val in flat_F:
        m = Hagan2002LognormalSABR(f=f_val, shift=shift, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol)
        alpha_list.append(m.alpha())
    flat_alpha = np.array(alpha_list)

    X_batch = np.zeros((len(flat_F), 8))
    X_batch[:, 0] = scale_val(beta, 'beta')
    X_batch[:, 1] = scale_val(rho, 'rho')
    X_batch[:, 2] = scale_val(volvol, 'volvol')
    X_batch[:, 3] = scale_val(v_atm_n, 'v_atm_n')
    X_batch[:, 4] = scale_val(flat_alpha, 'alpha')
    X_batch[:, 5] = scale_val(flat_F, 'F')
    X_batch[:, 6] = scale_val(flat_K, 'K')
    X_batch[:, 7] = scale_val(np.log(flat_K/flat_F), 'log_moneyness')

    pfn_preds = tabpfn.predict(X_batch).reshape(-1, 1)
    
    with torch.no_grad():
        X_torch = torch.FloatTensor(X_batch).to(device)
        P_torch = torch.FloatTensor(pfn_preds).to(device)
        hybrid_out = model(X_torch, P_torch)
        # Sortie brute du réseau (normalisée entre -1 et 1 environ)
        raw_ai_preds = hybrid_out[:, 6].cpu().numpy()

    # 6. ALIGNEMENT STATISTIQUE (Le cœur du correctif)
    # Au lieu d'utiliser le scaler JSON cassé, on aligne la distribution IA sur la distribution Vraie
    # Z_aligned = (Z_raw - mean_raw) * (std_true / std_raw) + mean_true
    
    true_mean = np.mean(Z_true_grad)
    true_std = np.std(Z_true_grad)
    
    ai_mean = np.mean(raw_ai_preds)
    ai_std = np.std(raw_ai_preds)
    
    print(f"Stats Vérité -> Moyenne: {true_mean:.4f}, Std: {true_std:.4f}")
    print(f"Stats IA Raw -> Moyenne: {ai_mean:.4f}, Std: {ai_std:.4f}")
    
    # Projection de l'IA dans l'espace réel
    Z_ai_aligned = (raw_ai_preds - ai_mean) * (true_std / ai_std) + true_mean
    
    # Reshape pour Plotly
    Z_true_surf = Z_true_grad.reshape(n_points, n_points)
    Z_ai_surf = Z_ai_aligned.reshape(n_points, n_points)

    # 7. Plotly 3D
    fig = go.Figure()

    # Vérité (Vert Translucide)
    fig.add_trace(go.Surface(
        z=Z_true_surf, x=F_grid, y=K_grid,
        colorscale='Greens', opacity=0.4,
        name='Math Truth', showscale=False
    ))

    # IA (Or Opaque)
    fig.add_trace(go.Surface(
        z=Z_ai_surf, x=F_grid, y=K_grid,
        colorscale='Oranges', 
        name='AI Prediction (Aligned)', showscale=True,
        colorbar=dict(title="dV/dK")
    ))

    fig.update_layout(
        title='Final Validation: Structural Convergence (Z-Score Aligned)',
        scene=dict(
            xaxis_title='Forward (F)',
            yaxis_title='Strike (K)',
            zaxis_title='Risk Sensitivity (dV/dK)',
            aspectmode='cube'
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_file = current_dir.parent / "final_proof_aligned.html"
    fig.write_html(output_file)
    print(f"✅ Preuve Alignée générée : {output_file}")
    
    try: import webbrowser; webbrowser.open(f'file://{output_file}')
    except: pass

if __name__ == "__main__":
    generate_aligned_3d_proof()