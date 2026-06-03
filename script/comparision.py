import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabpfn import TabPFNRegressor
from pathlib import Path
import sys

# --- Configuration des chemins ---
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

data_dir = current_dir.parent / "data"
data_path = data_dir / "sabr_hybrid_mesh_scaled.csv"
model_path = current_dir / "tabpfn_step9_causal_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ARCHITECTURE DU MODÈLE (Synchronisée avec le .pth)
# ==========================================
class StackingMLP(nn.Module):
    def __init__(self, input_dim=9): # 8 features SABR + 1 prédiction TabPFN
        super(StackingMLP, self).__init__()
        # On utilise "self.head" car c'est le nom enregistré dans ton fichier .pth
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.098),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.098),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.098),
            nn.Linear(128, 2) # Sorties : Volatilité et dV/dK
        )
        
    def forward(self, x):
        return self.head(x)

def run_real_comparison():
    print(f"🚀 Analyse des performances sur : {device}")

    # ==========================================
    # 2. CHARGEMENT DES DONNÉES
    # ==========================================
    if not data_path.exists():
        print(f"❌ Erreur : Fichier introuvable à {data_path}")
        return

    df = pd.read_csv(data_path)
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    
    # On sépare 1000 points pour le contexte et 1000 pour le test
    df_context = df.sample(1000, random_state=42)
    df_test = df.drop(df_context.index).sample(1000, random_state=123)
    
    X_test_raw = df_test[feature_cols].values
    y_true_vol = df_test['volatility_scaled'].values
    y_true_skew = df_test['dV_dK_scaled'].values

    # ==========================================
    # 3. INFÉRENCE TABPFN (ORACLE)
    # ==========================================
    print("🧠 TabPFN : Chargement du contexte In-Context Learning...")
    tabpfn = TabPFNRegressor(device=str(device), n_estimators=4)
    tabpfn.fit(df_context[feature_cols].values, df_context['volatility_scaled'].values)
    
    print("🔮 TabPFN : Prédiction de base...")
    pfn_preds = tabpfn.predict(X_test_raw)

    # ==========================================
    # 4. CHARGEMENT DU MODÈLE .PTH
    # ==========================================
    print(f"📥 Chargement du modèle final : {model_path.name}")
    model = StackingMLP(input_dim=len(feature_cols) + 1).to(device)
    
    # Chargement avec gestion de la correspondance des clés
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict)
        print("✅ Poids chargés avec succès !")
    except RuntimeError:
        print("⚠️ Correspondance partielle, tentative de chargement flexible...")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()

    # ==========================================
    # 5. INFÉRENCE DU STACKING (HYBRIDE)
    # ==========================================
    # Concaténation des entrées : features + oracle
    X_stack = np.column_stack([X_test_raw, pfn_preds])
    X_stack_tensor = torch.tensor(X_stack, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(X_stack_tensor).cpu().numpy()
        final_vol = output[:, 0]
        final_skew = output[:, 1]

    # ==========================================
    # 6. CALCUL DES MÉTRIQUES RÉELLES
    # ==========================================
    results = {
        "Metric": ["MAE Volatilité", "RMSE Volatilité", "MAE Skew (dV/dK)"],
        "TabPFN (Base)": [
            mean_absolute_error(y_true_vol, pfn_preds),
            np.sqrt(mean_squared_error(y_true_vol, pfn_preds)),
            "N/A (Bruit Numérique)"
        ],
        "Modèle Final (Step 9)": [
            mean_absolute_error(y_true_vol, final_vol),
            np.sqrt(mean_squared_error(y_true_vol, final_vol)),
            mean_absolute_error(y_true_skew, final_skew)
        ]
    }

    print("\n" + "="*50)
    print("📊 RÉSULTATS DE LA COMPARAISON")
    print("="*50)
    print(pd.DataFrame(results).to_string(index=False))
    print("="*50)

    # ==========================================
    # 7. GRAPHIQUES DE VALIDATION
    # ==========================================
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 5))

    # Volatilité
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_vol, final_vol, alpha=0.3, color='cyan')
    plt.plot([y_true_vol.min(), y_true_vol.max()], [y_true_vol.min(), y_true_vol.max()], '--w')
    plt.title("Précision Volatilité (Final)")
    plt.xlabel("Vérité JAX")
    plt.ylabel("Prédiction MLP")

    # Skew
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_skew, final_skew, alpha=0.3, color='orange')
    plt.plot([y_true_skew.min(), y_true_skew.max()], [y_true_skew.min(), y_true_skew.max()], '--w')
    plt.title("Précision Skew (Sobolev)")
    plt.xlabel("Vérité dV/dK")
    plt.ylabel("Prédiction MLP")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_real_comparison()