import numpy as np
import pandas as pd
from scipy.stats import qmc
from pathlib import Path

# Configuration
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

def generate_hybrid_mesh(n_total=6000, atm_ratio=0.5):
    """
    Génère un dataset Hybrid Mesh pour le modèle SABR.
    Combine une couverture globale (Sobol) et une densification ATM.
    """
    print(f"--- Génération du Hybrid Mesh SABR ({n_total} points) ---")
    
    n_global = int(n_total * (1 - atm_ratio))
    n_atm = n_total - n_global

    # Bornes de l'espace des paramètres (à ajuster selon tes besoins)
    bounds = {
        'beta': (0.25, 0.99),
        'rho': (-0.8, 0.8),
        'volvol': (0.1, 0.8),
        'v_atm_n': (0.005, 0.05),
        'alpha': (0.05, 0.40),
        'F': (0.01, 0.07),
        'K': (0.005, 0.10)
    }

    # Liste pour garantir l'ordre des colonnes
    cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K']
    l_bounds = np.array([bounds[c][0] for c in cols])
    u_bounds = np.array([bounds[c][1] for c in cols])

    # ==========================================
    # PARTIE 1 : GLOBAL MESH (Sobol Uniforme)
    # Garantit que les bords ne sont pas "affamés"
    # ==========================================
    print(f"1. Génération de {n_global} points globaux (Sobol)...")
    sampler_global = qmc.Sobol(d=len(cols), scramble=True)
    sample_global = sampler_global.random(n=n_global)
    # Mise à l'échelle des points Sobol [0,1] vers nos bornes réelles
    scaled_global = qmc.scale(sample_global, l_bounds, u_bounds)
    df_global = pd.DataFrame(scaled_global, columns=cols)

    # ==========================================
    # PARTIE 2 : ATM REFINEMENT (Localisé)
    # Résout la singularité au centre
    # ==========================================
    print(f"2. Génération de {n_atm} points ATM Refined...")
    # On utilise Sobol pour tous les paramètres SAUF K (donc dimension 6)
    cols_no_k = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F']
    l_bounds_no_k = np.array([bounds[c][0] for c in cols_no_k])
    u_bounds_no_k = np.array([bounds[c][1] for c in cols_no_k])

    sampler_atm = qmc.Sobol(d=len(cols_no_k), scramble=True)
    sample_atm = sampler_atm.random(n=n_atm)
    scaled_atm = qmc.scale(sample_atm, l_bounds_no_k, u_bounds_no_k)
    df_atm = pd.DataFrame(scaled_atm, columns=cols_no_k)

    # Stratégie de raffinement : K est très proche de F
    # On tire un log-moneyness d'une loi normale centrée sur 0 avec écart-type serré (ex: 0.1)
    np.random.seed(42)
    log_moneyness_atm = np.random.normal(loc=0.0, scale=0.10, size=n_atm)
    
    # Calcul de K et clipping pour s'assurer qu'il ne sort pas des bornes absolues
    df_atm['K'] = df_atm['F'] * np.exp(log_moneyness_atm)
    df_atm['K'] = np.clip(df_atm['K'], bounds['K'][0], bounds['K'][1])

    # ==========================================
    # PARTIE 3 : FUSION ET POST-PROCESSING
    # ==========================================
    print("3. Fusion et calcul des features enrichies...")
    df_hybrid = pd.concat([df_global, df_atm], ignore_index=True)
    
    # Mélange aléatoire (très important pour que les batchs PyTorch soient équilibrés)
    df_hybrid = df_hybrid.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calcul de la feature enrichie (Log Moneyness)
    df_hybrid['log_moneyness'] = np.log(df_hybrid['K'] / df_hybrid['F'])

    # Sauvegarde
    output_path = data_dir / "sabr_hybrid_mesh_features.csv"
    df_hybrid.to_csv(output_path, index=False)
    
    print(f"✅ Hybrid Mesh généré avec succès !")
    print(f"   Shape: {df_hybrid.shape}")
    print(f"   Sauvegardé dans : {output_path}")
    
    # Petit check de santé
    atm_count = len(df_hybrid[np.abs(df_hybrid['log_moneyness']) < 0.05])
    print(f"   Points très proches de l'ATM (|log(K/F)| < 0.05) : {atm_count} ({atm_count/n_total*100:.1f}%)")

if __name__ == "__main__":
    generate_hybrid_mesh()