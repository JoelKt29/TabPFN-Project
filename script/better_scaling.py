import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configuration des chemins
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"

def standardize_hybrid_dataset():
    print("--- Z-Score Standardization (StandardScaler) ---")
    
    # 1. Chargement du dataset contenant tes prix et dérivées JAX
    # (Remplace par le nom exact de ton fichier issu de l'étape JAX)
    input_path = data_dir / "sabr_hybrid_mesh_with_derivatives.csv"
    
    if not input_path.exists():
        print(f"❌ Erreur : Fichier {input_path.name} introuvable.")
        return

    df = pd.read_csv(input_path)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # 2. Définition des colonnes
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility', 'dV_dbeta', 'dV_drho', 'dV_dvolvol', 'dV_dalpha', 'dV_dF', 'dV_dK']

    # Dictionnaire pour stocker les paramètres de scaling (pour l'inférence Step 9)
    scaling_params = {
        'type': 'z_score',
        'features': {},
        'targets': {}
    }

    df_scaled = df.copy()

    # 3. Standardisation des Features
    print("Standardisation des Features (Moyenne = 0, Std = 1)...")
    for col in feature_cols:
        mean_val = float(df[col].mean())
        std_val = float(df[col].std())
        
        # Sécurité : éviter la division par zéro si une feature est constante
        if std_val < 1e-8:
            std_val = 1.0 
            
        df_scaled[col] = (df[col] - mean_val) / std_val
        scaling_params['features'][col] = {'mean': mean_val, 'std': std_val}

    # 4. Standardisation des Targets (Prix et Dérivées)
    print("Standardisation des Targets (Prix et Greeks)...")
    for col in target_cols:
        mean_val = float(df[col].mean())
        std_val = float(df[col].std())
        
        if std_val < 1e-8:
            std_val = 1.0
            
        # On crée de nouvelles colonnes avec le suffixe '_scaled' pour que ton Step 8 fonctionne sans modifs
        scaled_col_name = f"{col}_scaled"
        df_scaled[scaled_col_name] = (df[col] - mean_val) / std_val
        scaling_params['targets'][col] = {'mean': mean_val, 'std': std_val}

    # 5. Sauvegarde du Dataset Scalé
    output_csv = data_dir / "sabr_hybrid_mesh_scaled.csv"
    df_scaled.to_csv(output_csv, index=False)
    print(f"✅ Dataset standardisé sauvegardé sous : {output_csv.name}")

    # 6. Sauvegarde des Paramètres (Le "Cerveau" du Scaler)
    output_json = data_dir / "scaling_params_zscore.json"
    with open(output_json, 'w') as f:
        json.dump(scaling_params, f, indent=4)
    print(f"✅ Paramètres de scaling sauvegardés sous : {output_json.name}")

if __name__ == "__main__":
    standardize_hybrid_dataset()