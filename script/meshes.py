import numpy as np
from scipy.stats import qmc
import pandas as pd

from step05_jax_sabr import compute_sabr_with_jax


def generate_adaptive_mesh(n_coarse=10000, n_fine=40000, refinement_ratio=0.10):
    """
    Génère un dataset SABR avec Adaptive Mesh Refinement basé sur les gradients.

    n_coarse : nombre de points globaux (Sobol)
    n_fine   : nombre total de points raffinés
    refinement_ratio : proportion des points critiques
    """

    print("🚀 Adaptive Mesh Refinement lancé...")

    # ==============================
    # PARAMÈTRES (4D effectif)
    # ==============================
    bounds_lower = [0.01, 0.01, -0.9, 0.01]   # K, Alpha, Rho, Volvol
    bounds_upper = [0.15, 0.10,  0.9, 0.80]

    # ==============================
    # PHASE 1 : Coarse sampling
    # ==============================
    print(f"🔍 Génération de {n_coarse} points Sobol...")

    sampler = qmc.Sobol(d=4, scramble=True)
    sample = sampler.random(n=n_coarse)
    scaled = qmc.scale(sample, bounds_lower, bounds_upper)

    # Variables
    K, Alpha, Rho, Volvol = scaled.T
    F = np.full_like(K, 0.05)
    T = np.full_like(K, 1.0)
    Beta = np.full_like(K, 0.5)

    coarse_scaled = np.column_stack([K, F, T, Alpha, Beta, Rho, Volvol])

    # ==============================
    # PHASE 2 : Gradients JAX
    # ==============================
    print("⚙️ Calcul des gradients SABR...")

    vols, grads = compute_sabr_with_jax(K, F, T, Alpha, Beta, Rho, Volvol)

    # Score de complexité
    complexity_score = np.abs(grads['dV_dK']) + np.abs(grads['dV_dvolvol'])

    # Sélection zones critiques
    n_top = max(1, int(n_coarse * refinement_ratio))
    top_indices = np.argsort(complexity_score)[-n_top:]
    critical_points = coarse_scaled[top_indices]

    # ==============================
    # PHASE 3 : Raffinement local
    # ==============================
    print(f"🎯 Raffinement autour de {n_top} zones critiques...")

    sampler_fine = qmc.Sobol(d=7, scramble=True)

    points_per_subdomain = max(1, n_fine // n_top)
    fine_samples = []

    for point in critical_points:
        # Micro-domaine autour du point
        lower = np.minimum(point * 0.95, point * 1.05)
        upper = np.maximum(point * 0.95, point * 1.05)

        micro_sample = sampler_fine.random(n=points_per_subdomain)
        micro_scaled = qmc.scale(micro_sample, lower, upper)

        fine_samples.append(micro_scaled)

    fine_scaled = np.vstack(fine_samples)

    # ==============================
    # PHASE 4 : Fusion
    # ==============================
    final_dataset = np.vstack((coarse_scaled, fine_scaled))

    print(f"✅ Dataset final : {len(final_dataset)} points")

    columns = ['K', 'F', 'T', 'Alpha', 'Beta', 'Rho', 'Volvol']
    df = pd.DataFrame(final_dataset, columns=columns)

    return df


# ==============================
# TEST
# ==============================
if __name__ == "__main__":
    df_dataset = generate_adaptive_mesh(
        n_coarse=10000,
        n_fine=40000,
        refinement_ratio=0.10
    )

    df_dataset.to_csv("adaptive_sabr_dataset.csv", index=False)
    print("📁 Dataset sauvegardé")