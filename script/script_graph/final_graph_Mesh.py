import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import sys
from pathlib import Path
from tabpfn import TabPFNRegressor

# Import de ta formule mathématique exacte
from script.true_step02_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR

# Chemins
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"
model_path = current_dir / "tabpfn_sabr_step8_stacking.pth"

# Les DEUX scalers
sabr_scaler_path = data_dir / "scaling_params_zscore.json"
tabpfn_scaler_path = current_dir / "tabpfn_scaler.json"

# ==========================================
# DEFINITION DU MODELE (Pour éviter les conflits d'import)
# ==========================================
def get_activation(name: str):
    activations = {
        'swish': nn.SiLU(),
        'mish': nn.Mish(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'relu': nn.ReLU()
    }
    return activations.get(name.lower(), nn.SiLU())

class TabPFNStackingModel(nn.Module):
    def __init__(self, config, n_outputs=7):
        super().__init__()
        # On force à 11 pour correspondre exactement à ta Step 8
        input_dim = 11 
        layers = []
        prev_dim = input_dim

        h_dims = config.get('hidden_dims', [512, 256, 128])
        activation_name = config.get('activation', 'swish')

        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                get_activation(activation_name),
                nn.Dropout(config.get('dropout', 0.1))
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, x_sabr, x_tabpfn):
        x = torch.cat([x_sabr, x_tabpfn], dim=1)
        return self.head(x)

# ==========================================
# GENERATION DE LA PREUVE 3D
# ==========================================
def generate_true_3d_proof():
    print("--- Génération de la Preuve 3D (Échelle Réelle) ---")
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # 1. Chargement des configs et scalers
    with open(config_path, 'r') as f: config = json.load(f)
    with open(sabr_scaler_path, 'r') as f: sabr_scaler = json.load(f)
    with open(tabpfn_scaler_path, 'r') as f: tabpfn_scaler = json.load(f)
    
    # 2. Chargement du Modèle
    model = TabPFNStackingModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Context TabPFN
    df_train = pd.read_csv(data_dir / "sabr_hybrid_mesh_scaled.csv").sample(1000, random_state=42)
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    X_train_ctx = df_train[feature_cols].values
    y_train_ctx = df_train['volatility_scaled'].values
    
    tabpfn = TabPFNRegressor(device=device, n_estimators=4, ignore_pretraining_limits=True)
    tabpfn.fit(X_train_ctx, y_train_ctx)

    # 4. Grille de Test
    n_points = 50
    f_range = np.linspace(0.02, 0.06, n_points)
    k_range = np.linspace(0.01, 0.08, n_points)
    F_grid, K_grid = np.meshgrid(f_range, k_range)
    flat_F, flat_K = F_grid.ravel(), K_grid.ravel()
    
    # Paramètres Scénario (Doivent être dans les bornes du Hybrid Mesh)
    beta, rho, volvol, v_atm_n = 0.62, 0.0, 0.45, 0.027
    T, shift = 1.0, 0.0

    # 5. Calcul de la Vérité Terrain Mathématique
    print("Calcul de la surface Mathématique (Hagan)...")
    Z_true_grad = np.zeros_like(flat_F)
    eps = 1e-5
    for i in range(len(flat_F)):
        m = Hagan2002LognormalSABR(f=flat_F[i], shift=shift, t=T, v_atm_n=v_atm_n, 
                                   beta=beta, rho=rho, volvol=volvol)
        # ---> CORRECTION ICI : On utilise la volatilité Lognormale (Black) <---
        v_p = m.lognormal_vol(flat_K[i] + eps)
        v_m = m.lognormal_vol(flat_K[i] - eps)
        Z_true_grad[i] = (v_p - v_m) / (2*eps)

    # 6. Préparation des Entrées pour l'IA (Z-Score Scaling)
    print("Prédiction IA...")
    alpha_list = []
    for f_val in flat_F:
        m = Hagan2002LognormalSABR(f=f_val, shift=shift, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol)
        alpha_list.append(m.alpha())
    flat_alpha = np.array(alpha_list)

    def z_scale(val, param_name):
        mu = sabr_scaler['features'][param_name]['mean']
        sig = sabr_scaler['features'][param_name]['std']
        return (val - mu) / sig

    X_batch = np.zeros((len(flat_F), 8))
    X_batch[:, 0] = z_scale(beta, 'beta')
    X_batch[:, 1] = z_scale(rho, 'rho')
    X_batch[:, 2] = z_scale(volvol, 'volvol')
    X_batch[:, 3] = z_scale(v_atm_n, 'v_atm_n')
    X_batch[:, 4] = z_scale(flat_alpha, 'alpha')
    X_batch[:, 5] = z_scale(flat_F, 'F')
    X_batch[:, 6] = z_scale(flat_K, 'K')
    X_batch[:, 7] = z_scale(np.log(flat_K/flat_F), 'log_moneyness')

    # 7. Inférence TabPFN + Enrichissement + Normalisation
    pfn_raw = tabpfn.predict(X_batch).reshape(-1, 1)
    
    # Feature Engineering (les 3 features de ta Step 8)
    pfn_enriched = np.hstack([pfn_raw, pfn_raw**2, np.abs(pfn_raw)])
    
    # Normalisation stricte avec le scaler sauvegardé
    pfn_mu = np.array(tabpfn_scaler['mean'])
    pfn_sig = np.array(tabpfn_scaler['std'])
    pfn_scaled = (pfn_enriched - pfn_mu) / pfn_sig

    # 8. Inférence Réseau Hybride
    with torch.no_grad():
        X_torch = torch.FloatTensor(X_batch).to(device)
        P_torch = torch.FloatTensor(pfn_scaled).to(device)
        hybrid_out = model(X_torch, P_torch)
        
        # Le Skew (dV/dK) est la dernière colonne (index 6)
        pred_skew_scaled = hybrid_out[:, 6].cpu().numpy()

    # 9. INVERSION DU SCALING
    target_mu = sabr_scaler['targets']['dV_dK']['mean']
    target_sig = sabr_scaler['targets']['dV_dK']['std']
    
    Z_ai_real = (pred_skew_scaled * target_sig) + target_mu

    # Reshape pour Plotly
    Z_true_surf = Z_true_grad.reshape(n_points, n_points)
    Z_ai_surf = Z_ai_real.reshape(n_points, n_points)

    # 10. Plotly 3D
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z_true_surf, x=F_grid, y=K_grid,
        colorscale='Greens', opacity=0.5,
        name='Math Truth', showscale=False
    ))

    fig.add_trace(go.Surface(
        z=Z_ai_surf, x=F_grid, y=K_grid,
        colorscale='Oranges', opacity=0.9,
        name='AI Prediction (Real Scale)', showscale=True,
        colorbar=dict(title="dV/dK")
    ))

    fig.update_layout(
        title='Final Validation: Absolute Structural Convergence (Real Scale)',
        scene=dict(
            xaxis_title='Forward (F)',
            yaxis_title='Strike (K)',
            zaxis_title='Risk Sensitivity (dV/dK)',
            aspectmode='cube'
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    output_file = current_dir.parent / "final_proof_real_scale.html"
    fig.write_html(output_file)
    print(f"✅ Preuve générée : {output_file}")

    
    # --- LES DEUX LIGNES À AJOUTER ---
    try: import webbrowser; webbrowser.open(f'file://{output_file}')
    except: pass


if __name__ == "__main__":
    generate_true_3d_proof()