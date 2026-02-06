from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
import os
from tabpfn import TabPFNRegressor

# Gestion des chemins
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"

try:
    from step6_loss_with_derivatives import create_loss_function
except ImportError:
    print("❌ Erreur : Impossible de trouver step6_loss_with_derivatives.py")

# ==========================================
# 1. ARCHITECTURE TABPFN + INTERCEPTION
# ==========================================
class TabPFNSABRRegressor(nn.Module):
    def __init__(self, config, n_outputs=7):
        super().__init__()
        
        print("🔄 Chargement de TabPFN...")
        self.tabpfn = TabPFNRegressor(device='cpu')
        # Initialisation du modèle interne
        self.tabpfn.fit(np.random.randn(5, 8), np.random.randn(5))
        
        # On récupère le modèle PyTorch sous-jacent
        self.inner_model = self.tabpfn.model_ if hasattr(self.tabpfn, 'model_') else self.tabpfn.model
        
        # On gèle tout le modèle
        for param in self.inner_model.parameters():
            param.requires_grad = False
            
        # --- LE HACK DU HOOK ---
        # On va stocker les caractéristiques ici à chaque passage
        self.captured_features = None
        
        def hook_fn(module, input, output):
            # L'output du transformer est souvent [Seq, Batch, 512]
            # On capture et on enlève la dimension de séquence
            self.captured_features = output[0] if output.dim() == 3 else output

        # On attache le hook à la fin du transformer (juste avant la couche de sortie)
        # Dans TabPFN, c'est généralement le bloc 2 ou le dernier bloc du Sequential
        self.inner_model[2].register_forward_hook(hook_fn)

        # --- TÊTE DE MODÈLE (TRANSFORMER HEAD) ---
        d_model = 512 
        layers = []
        prev_dim = d_model
        
        hidden_dims = config.get('hidden_dims', [512, 256, 128])
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(), # Swish
                nn.Dropout(config.get('dropout', 0.09))
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, n_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # 1. On utilise le passage standard de TabPFN
        # Cela gère toute la plomberie interne (masques, séquences) pour nous
        with torch.no_grad():
            # On passe x. La sortie 'preds' ne nous intéresse pas, 
            # c'est le hook qui va remplir 'self.captured_features'
            _ = self.tabpfn.predict(x.cpu().numpy())
            
        # 2. On récupère les 512 caractéristiques interceptées
        features = torch.FloatTensor(self.captured_features).to(x.device)
        
        # 3. On passe dans ta tête optimisée
        return self.head(features)

# ==========================================
# 2. LOGIQUE D'ENTRAÎNEMENT
# ==========================================
def run_step8():
    config_path = current_dir / "ray_results" / "best_config.json"
    if not config_path.exists():
        config = {"hidden_dims": [512, 256, 128], "lr": 0.0016, "dropout": 0.09, 
                  "batch_size": 128, "value_weight": 0.5, "derivative_weight": 0.1}
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    df = pd.read_csv(data_dir / "sabr_with_derivatives_scaled.csv")
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    
    X = torch.FloatTensor(df[feature_cols].values)
    y = torch.FloatTensor(df[target_cols].values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNSABRRegressor(config, n_outputs=len(target_cols)).to(device)

    criterion = create_loss_function(
        loss_type='derivative',
        value_weight=config.get('value_weight', 0.5),
        derivative_weight=config.get('derivative_weight', 0.1)
    )
    
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config.get('lr', 0.0016))

    print("\n" + "="*50)
    print("✨ STEP 8 : TRAINING HYBRID (HOOK INTERCEPTION)")
    print(f"Objectif : Battre MAE {config.get('best_mae', 0.0055)}")
    print("="*50)

    model.train()
    batch_size = config.get('batch_size', 128)
    
    for epoch in range(51):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0
        
        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices].to(device), y[indices].to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            
            loss, _ = criterion(out[:, 0:1], batch_y[:, 0:1], 
                               {f'd{j}': out[:, j:j+1] for j in range(1, 7)},
                               {f'd{j}': batch_y[:, j:j+1] for j in range(1, 7)})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            current_mae = torch.mean(torch.abs(out[:, 0:1] - batch_y[:, 0:1])).item()
            print(f"Époque {epoch:02d} | Loss: {epoch_loss/(X.size(0)/batch_size):.6f} | MAE: {current_mae:.6f}")

    torch.save(model.state_dict(), current_dir / "tabpfn_sabr_step8_hybrid.pth")
    print("\n✅ Modèle sauvegardé.")

if __name__ == "__main__":
    run_step8()