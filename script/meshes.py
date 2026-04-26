import numpy as np
from scipy.stats import qmc
import pandas as pd

# Importe ton moteur JAX (Ajuste le nom du fichier et de la fonction si besoin)
from step02_jax_sabr import compute_sabr_with_jax

def generate_adaptive_mesh(n_coarse=10000, n_fine=40000, refinement_ratio=0.10):
    """
    Génère un dataset SABR avec un maillage densifié sur les zones à fort gradient.
    
    n_coarse : Nombre de points pour le repérage global (Sobol).
    n_fine   : Nombre de points générés UNIQUEMENT dans les zones difficiles.
    refinement_ratio : % des points Coarse considérés comme "zones de danger".
    """
    print("🚀 Démarrage de l'Adaptive Mesh Refinement...")

    # 1. Définition des bornes (Bounds) restrictives (Suite aux conseils de l'encadrant)
    # Paramètres : [K, F, T, Alpha, Beta, Rho, Volvol]
    # On fixe F à 0.05 et T à 1.0 (ou tu peux les faire légèrement varier)
    bounds_lower = [0.01, 0.05, 1.0, 0.01, 0.5, -0.9, 0.01]
    bounds_upper = [0.15, 0.05, 1.0, 0.10, 0.5,  0.9, 0.80]

    # ==========================================
    # PHASE 1 : Coarse Grid (Grille de Repérage)
    # ==========================================
    print(f"🔍 Phase 1 : Génération de {n_coarse} points de repérage (Sobol)...")
    sampler_coarse = qmc.Sobol(d=7, scramble=True)
    coarse_sample = sampler_coarse.random(n=n_coarse)
    coarse_scaled = qmc.scale(coarse_sample, bounds_lower, bounds_upper)

    # ==========================================
    # PHASE 2 : Détection via les Gradients JAX
    # ==========================================
    print("⚙️ Phase 2 : Calcul des sensibilités via JAX pour détecter les zones complexes...")
    K, F, T, Alpha, Beta, Rho, Volvol = coarse_scaled.T
    
    # On fait appel à JAX pour obtenir les dérivées exactes
    vols, grads = compute_sabr_with_jax(K, F, T, Alpha, Beta, Rho, Volvol)
    
    # Calcul du "Score de Complexité"
    # On cible les zones où la pente (Skew) ou la sensibilité au Volvol est extrême
    complexity_score = np.abs(grads['dV_dK']) + np.abs(grads['dV_dvolvol'])
    
    # On isole les X% des points les plus instables
    n_top = int(n_coarse * refinement_ratio)
    top_indices = np.argsort(complexity_score)[-n_top:]
    critical_points = coarse_scaled[top_indices]

    # ==========================================
    # PHASE 3 : Mesh Refinement (Densification)
    # ==========================================
    print(f"🎯 Phase 3 : Densification du maillage autour de {n_top} sous-domaines critiques...")
    fine_samples = []
    points_per_subdomain = n_fine // n_top
    
    # On initialise un nouveau générateur pour les micro-domaines
    sampler_fine = qmc.Sobol(d=7, scramble=True)
    
    for point in critical_points:
        # Création d'un micro-domaine de +/- 5% autour du point de danger
        micro_lower = point * 0.95
        micro_upper = point * 1.05
        
        # On génère des points ultra-concentrés dans ce petit carré
        micro_points = sampler_fine.random(n=points_per_subdomain)
        micro_scaled = qmc.scale(micro_points, micro_lower, micro_upper)
        fine_samples.append(micro_scaled)
        
    fine_scaled = np.vstack(fine_samples)

    # ==========================================
    # PHASE 4 : Fusion et Export
    # ==========================================
    final_dataset = np.vstack((coarse_scaled, fine_scaled))
    
    print(f"✅ Terminé ! Dataset hybride généré : {len(final_dataset)} points.")
    
    # Conversion en DataFrame Pandas pour l'envoi vers TabPFN
    columns = ['K', 'F', 'T', 'Alpha', 'Beta', 'Rho', 'Volvol']
    df = pd.DataFrame(final_dataset, columns=columns)
    
    return df

# --- Pour tester le script tout de suite ---
if __name__ == "__main__":
    df_dataset = generate_adaptive_mesh(n_coarse=10000, n_fine=40000, refinement_ratio=0.10)
    # Tu peux ensuite sauvegarder df_dataset en CSV ou l'utiliser directement.
    # df_dataset.to_csv('adaptive_sabr_dataset.csv', index=False)