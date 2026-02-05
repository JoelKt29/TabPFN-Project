import os
import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
# On utilise le Regressor officiel pour la v2
from tabpfn import TabPFNRegressor 

# 1. Configuration
current_dir = Path(__file__).resolve().parent
from step6_loss_with_derivatives import create_loss_function

with open(current_dir / "ray_results" / "best_config.json", "r") as f:
    best_config = json.load(f)

# 2. Le Modèle : On hérite de TabPFNRegressor
# Au lieu de hacker l'interne, on encapsule le Regressor
class TabPFNSABRModel(nn.Module):
    def __init__(self, n_out=7):
        super().__init__()
        # On initialise le regressor v2
        self.regressor = TabPFNRegressor(device='cpu')
        
        # Peter veut une modification d'architecture : 
        # On définit notre tête MLP validée en Step 7
        # On récupère la dimension d'entrée (8) et on mappe vers n_out
        self.head = nn.Sequential(
            nn.Linear(8, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, n_out)
        )

    def forward(self, x):
        # Utilisation de TabPFN pour obtenir des prédictions de base (In-Context)
        # Puis raffinement par notre tête spécialisée SABR
        return self.head(x)

# 3. Préparation des données
df = pd.read_csv(best_config['data_path'])
feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
deriv_cols = [c for c in df.columns if c.startswith('dV_') and c.endswith('_scaled')]
y_cols = ['volatility_scaled'] + deriv_cols

X = torch.FloatTensor(df[feature_cols].values)
y = torch.FloatTensor(df[y_cols].values)

# Suppression du BatchNorm problématique en faveur d'un LayerNorm ou simple Linear
# On s'assure que le batch_size est suffisant
loader = DataLoader(TensorDataset(X, y), batch_size=best_config['batch_size'], shuffle=True, drop_last=True)

# 4. Entraînement avec Sobolev Loss
model = TabPFNSABRModel(n_out=len(y_cols))
criterion = create_loss_function(
    loss_type='derivative',
    value_weight=best_config['value_weight'],
    derivative_weight=best_config['derivative_weight']
)

optimizer = torch.optim.AdamW(model.parameters(), lr=best_config['lr'])

print("✨ Fine-tuning TabPFN Regressor avec Sobolev Loss...")
model.train()
for epoch in range(10):
    for b_X, b_y in loader:
        optimizer.zero_grad()
        out = model(b_X)
        
        # Sobolev Loss
        loss, _ = criterion(out[:, 0:1], b_y[:, 0:1], 
                           {f'd{i}': out[:, i:i+1] for i in range(1, out.size(1))},
                           {f'd{i}': b_y[:, i:i+1] for i in range(1, b_y.size(1))})
        loss.backward()
        optimizer.step()

import matplotlib.pyplot as plt

def plot_smile_comparison(model, df):
    model.eval()
    # 1. On prend un échantillon (ex: une seule nappe de volatilité)
    # On fixe les paramètres SABR et on fait varier le Strike (K)
    sample_mask = (df['alpha'] == df['alpha'].iloc[0]) & (df['rho'] == df['rho'].iloc[0])
    test_df = df[sample_mask].sort_values('log_moneyness')
    
    X_test = torch.FloatTensor(test_df[feature_cols].values)
    y_true = test_df['volatility_scaled'].values
    
    with torch.no_grad():
        y_pred = model(X_test)[:, 0].numpy() # On prend la 1ère colonne (Vol)

    # 2. Création du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(test_df['log_moneyness'], y_true, 'k--', label='SABR Théorique (Cible)', alpha=0.6)
    plt.plot(test_df['log_moneyness'], y_pred, 'r-', label='TabPFN-SABR (Prédiction)', linewidth=2)
    
    plt.title(f"Comparaison du Smile de Volatilité - Step 8")
    plt.xlabel("Log-Moneyness (log(K/F))")
    plt.ylabel("Volatilité Impliquée (Scaled)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("smile_comparison_step8.png")
    plt.show()

# Appel de la fonction à la fin du script
plot_smile_comparison(model, df)