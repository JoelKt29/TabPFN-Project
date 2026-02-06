from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from tabpfn import TabPFNRegressor

# --- CONFIGURATION DES CHEMINS ---
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"

try:
    from step6_loss_with_derivatives import create_loss_function
except ImportError:
    print("❌ Erreur : step6_loss_with_derivatives.py introuvable.")

# ==========================================
# 1. ARCHITECTURE SCM (META-LEARNER)
# ==========================================
class FinancialSCM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Input: 8 features SABR + 1 prédiction TabPFN = 9
        input_dim = 9 
        
        h_dims = config.get('hidden_dims', [512, 256, 128])
        layers = []
        prev_dim = input_dim
        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(), # Swish
                nn.Dropout(config.get('dropout', 0.05))
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 7)) # Vol + 6 Greeks
        self.head = nn.Sequential(*layers)

    def forward(self, x_sabr, x_tabpfn):
        # On concatène les données brutes et l'avis du Transformer
        combined = torch.cat([x_sabr, x_tabpfn], dim=1)
        return self.head(combined)

# ==========================================
# 2. RUN STEP 9 (CAUSAL ADAPTATION)
# ==========================================
def run_step9():
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print("📂 Config Step 7 chargée.")
    else:
        config = {"hidden_dims": [512, 256, 128], "lr": 0.001, "dropout": 0.05}

    # Données
    df = pd.read_csv(data_dir / "sabr_with_derivatives_scaled.csv")
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    
    X_raw = df[feature_cols].values
    Y_raw = df[target_cols].values

    # ÉTAPE CRUCIALE : On utilise TabPFN comme un SCM générateur
    print("🔄 TabPFN génère les priors structurels...")
    tabpfn = TabPFNRegressor(device='cpu')
    
    # On lui donne un échantillon de contexte (In-context learning)
    train_idx = np.random.choice(len(df), 500, replace=False)
    tabpfn.fit(X_raw[train_idx], Y_raw[train_idx, 0])
    
    # On récupère sa vision du monde
    tabpfn_preds = tabpfn.predict(X_raw).reshape(-1, 1)

    X_sabr = torch.FloatTensor(X_raw)
    X_tabpfn = torch.FloatTensor(tabpfn_preds)
    Y = torch.FloatTensor(Y_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinancialSCM(config).to(device)

    # Loss Sobolev : On apprend au modèle la causalité (Comment les entrées causent les dérivées)
    criterion = create_loss_function(loss_type='derivative', value_weight=0.5, derivative_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 0.001))

    print("\n🚀 TRAINING SCM : Apprentissage des lois causales du SABR...")
    model.train()
    
    for epoch in range(101):
        indices = torch.randperm(X_sabr.size(0))[:128]
        bx_s, bx_t, by = X_sabr[indices].to(device), X_tabpfn[indices].to(device), Y[indices].to(device)

        optimizer.zero_grad()
        out = model(bx_s, bx_t)
        
        # Sobolev Loss : apprend les relations structurelles (Greeks)
        loss, _ = criterion(out[:, 0:1], by[:, 0:1], 
                           {f'd{j}': out[:, j:j+1] for j in range(1, 7)},
                           {f'd{j}': by[:, j:j+1] for j in range(1, 7)})
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            mae = torch.mean(torch.abs(out[:, 0] - by[:, 0])).item()
            print(f"Époque {epoch:03d} | Loss: {loss.item():.6f} | MAE Vol: {mae:.6f}")

    torch.save(model.state_dict(), current_dir / "tabpfn_sabr_step9_causal_final.pth")
    print(f"✅ Modèle SCM (Stacking) sauvegardé.")

if __name__ == "__main__":
    run_step9()