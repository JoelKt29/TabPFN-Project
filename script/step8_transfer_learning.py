from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from tabpfn import TabPFNRegressor


current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"

from step6_loss_with_derivatives import create_loss_function

# ==========================================
# 1. ARCHITECTURE HYBRIDE (STACKING)
# ==========================================
class TabPFNStackingModel(nn.Module):
    def __init__(self, config, n_outputs=7):
        super().__init__()
        # 8 features SABR + 1 feature (la prédiction brute de TabPFN)
        input_dim = 9 
        
        layers = []
        prev_dim = input_dim
        # On récupère tes hidden_dims optimisés [512, 256, 128]
        h_dims = config.get('hidden_dims', [512, 256, 128])
        
        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(), # Swish activation (Peter-approved)
                nn.Dropout(config.get('dropout', 0.09))
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, n_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, x_sabr, x_tabpfn):
        # On fusionne les données d'entrée avec l'avis de TabPFN
        combined = torch.cat([x_sabr, x_tabpfn], dim=1)
        return self.head(combined)

# ==========================================
# 2. LOGIQUE D'ENTRAÎNEMENT
# ==========================================
def run_step8():
    # A. Chargement de la config Step 7
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"📂 Config Step 7 chargée avec succès depuis : {config_path}")
    else:
        print(f"⚠️ Attention : {config_path} introuvable. Utilisation des défauts.")
        config = {"hidden_dims": [512, 256, 128], "lr": 0.0016, "dropout": 0.09, 
                  "value_weight": 0.5, "derivative_weight": 0.1, "batch_size": 128}

    # B. Chargement des données
    csv_path = data_dir / "sabr_with_derivatives_scaled.csv"
    if not csv_path.exists():
        print(f"❌ Erreur : CSV introuvable à {csv_path}")
        return

    df = pd.read_csv(csv_path)
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    
    X_raw = df[feature_cols].values
    y_raw = df[target_cols].values

    # C. GÉNÉRATION DE LA FEATURE "CERVEAU" (TabPFN)
    print("🔄 Analyse TabPFN en cours (Stacking)...")
    tabpfn = TabPFNRegressor(device='cpu')
    
    # On utilise 500 points pour fixer le contexte In-Context Learning
    train_idx = np.random.choice(len(df), 500, replace=False)
    tabpfn.fit(X_raw[train_idx], y_raw[train_idx, 0])
    
    # On génère la prédiction qui servira de 9ème colonne
    tabpfn_preds = tabpfn.predict(X_raw).reshape(-1, 1)

    # Conversion en tenseurs
    X_sabr = torch.FloatTensor(X_raw)
    X_tabpfn = torch.FloatTensor(tabpfn_preds)
    Y = torch.FloatTensor(y_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNStackingModel(config).to(device)

    # Loss Sobolev (Step 6)
    criterion = create_loss_function(
        loss_type='derivative',
        value_weight=config.get('value_weight', 0.5),
        derivative_weight=config.get('derivative_weight', 0.1)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 0.0016))

    print("\n" + "="*50)
    print("✨ STEP 8 : TRAINING HYBRID (STACKING MODE)")
    print(f"Cible MAE à battre : {config.get('best_mae', 0.00554181)}")
    print("="*50)

    model.train()
    batch_size = config.get('batch_size', 128)
    
    for epoch in range(101): # 100 époques pour bien converger
        permutation = torch.randperm(X_sabr.size(0))
        epoch_loss = 0
        
        for i in range(0, X_sabr.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            bx_s = X_sabr[indices].to(device)
            bx_t = X_tabpfn[indices].to(device)
            by = Y[indices].to(device)

            optimizer.zero_grad()
            out = model(bx_s, bx_t)
            
            # Calcul Sobolev : Vol + 6 Dérivées
            loss, _ = criterion(out[:, 0:1], by[:, 0:1], 
                               {f'd{j}': out[:, j:j+1] for j in range(1, 7)},
                               {f'd{j}': by[:, j:j+1] for j in range(1, 7)})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            current_mae = torch.mean(torch.abs(out[:, 0:1] - by[:, 0:1])).item()
            print(f"Époque {epoch:03d} | Loss Sobolev: {epoch_loss/(X_sabr.size(0)/batch_size):.6f} | MAE Vol: {current_mae:.6f}")

    # D. SAUVEGARDE
    save_path = current_dir / "tabpfn_sabr_step8_stacking.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Modèle hybride sauvegardé ici : {save_path}")

if __name__ == "__main__":
    run_step8()