from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from tabpfn import TabPFNRegressor

# --- GESTION DES CHEMINS ---
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"

try:
    from step6_loss_with_derivatives import create_loss_function
except ImportError:
    print("❌ Erreur : step6_loss_with_derivatives.py introuvable.")

# ==========================================
# 1. ARCHITECTURE STEP 9 (CAUSAL ADAPTATION)
# ==========================================
class TabPFNCausalModel(nn.Module):
    def __init__(self, config, n_outputs=7):
        super().__init__()
        print("🔄 Initialisation de TabPFN pour Fine-tuning Causal...")
        reg = TabPFNRegressor(device='cpu')
        reg.fit(np.random.randn(5, 8), np.random.randn(5))
        
        self.backbone = reg.model_ if hasattr(reg, 'model_') else reg.model
        
        # --- DÉGEL TOTAL (UNFREEZING) ---
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        input_dim = 512 # Sortie directe du Transformer
        layers = []
        prev_dim = input_dim
        h_dims = config.get('hidden_dims', [512, 256, 128])
        
        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(), # Swish
                nn.Dropout(config.get('dropout', 0.05))
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, n_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # Trick de la séquence double (Size 2) pour éviter l'erreur de TabPFN
        x_in = x.unsqueeze(0)
        x_combined = torch.cat([x_in, x_in], dim=0) 
        y_combined = torch.zeros(2, x.size(0), 1).to(x.device)
        
        # Passage dans le backbone unfrozen
        all_features = self.backbone(x_combined, y_combined)
        
        if isinstance(all_features, (list, tuple)):
            all_features = all_features[0]
            
        # Extraction du test (index 1)
        features = all_features[1, :, :]
            
        return self.head(features)

# ==========================================
# 2. LOGIQUE DE FINE-TUNING
# ==========================================
def run_step9():
    # A. Chargement de la config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"📂 Config Step 7 chargée : {config_path}")
    else:
        config = {"hidden_dims": [512, 256, 128], "lr": 0.0001, "value_weight": 0.5, "derivative_weight": 0.1}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNCausalModel(config).to(device)
    
    # B. Préparation des données
    csv_path = data_dir / "sabr_with_derivatives_scaled.csv"
    df = pd.read_csv(csv_path)
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    
    X = torch.FloatTensor(df[feature_cols].values)
    Y = torch.FloatTensor(df[target_cols].values)

    criterion = create_loss_function(
        loss_type='derivative',
        value_weight=config.get('value_weight', 0.5),
        derivative_weight=config.get('derivative_weight', 0.1)
    )
    
    # Optimizer avec LR différentiel pour protéger TabPFN
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-6}, 
        {'params': model.head.parameters(), 'lr': config.get('lr', 1e-4)}
    ])

    print("\n" + "="*50)
    print("🚀 STEP 9 : CAUSAL FINE-TUNING (START)")
    print("="*50)

    model.train()
    batch_size = 16 
    
    for epoch in range(21):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0
        
        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            if len(indices) < 2: continue
            bx, by = X[indices].to(device), Y[indices].to(device)

            optimizer.zero_grad()
            out = model(bx)
            
            loss, _ = criterion(out[:, 0:1], by[:, 0:1], 
                               {f'd{j}': out[:, j:j+1] for j in range(1, 7)},
                               {f'd{j}': by[:, j:j+1] for j in range(1, 7)})
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 5 == 0:
            current_mae = torch.mean(torch.abs(out[:, 0] - by[:, 0])).item()
            print(f"Époque {epoch:02d} | Loss: {epoch_loss/(X.size(0)/batch_size):.6f} | MAE: {current_mae:.6f}")

    # SAUVEGARDE
    final_path = current_dir / "tabpfn_sabr_step9_causal_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Terminé ! Modèle expert sauvegardé : {final_path.name}")

if __name__ == "__main__":
    run_step9()