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
    Generate a Hybrid Mesh dataset for the SABR model.
    Combine une couverture globale (Sobol) et une densification ATM.
    """
    print(f"--- Generating SABR Hybrid Mesh ({n_total} points) ---")
    
    n_global = int(n_total * (1 - atm_ratio))
    n_atm = n_total - n_global

    # Parameter-space bounds (adjust as needed)
    bounds = {
        'beta': (0.25, 0.99),
        'rho': (-0.8, 0.8),
        'volvol': (0.1, 0.8),
        'v_atm_n': (0.005, 0.05),
        'alpha': (0.05, 0.40),
        'F': (0.01, 0.07),
        'K': (0.005, 0.10)
    }

    # List to keep a fixed column order
    cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K']
    l_bounds = np.array([bounds[c][0] for c in cols])
    u_bounds = np.array([bounds[c][1] for c in cols])

    # ==========================================
    # PART 1: GLOBAL MESH (uniform Sobol)
    # Ensures the edges are not under-sampled
    # ==========================================
    print(f"1. Generating {n_global} global points (Sobol)...")
    sampler_global = qmc.Sobol(d=len(cols), scramble=True)
    sample_global = sampler_global.random(n=n_global)
    # Scale Sobol points from [0,1] to the real bounds
    scaled_global = qmc.scale(sample_global, l_bounds, u_bounds)
    df_global = pd.DataFrame(scaled_global, columns=cols)

    # ==========================================
    # PART 2: ATM REFINEMENT (localized)
    # Resolves the singularity at the center
    # ==========================================
    print(f"2. Generating {n_atm} ATM-refined points...")
    # Use Sobol for all parameters EXCEPT K (so dimension 6)
    cols_no_k = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F']
    l_bounds_no_k = np.array([bounds[c][0] for c in cols_no_k])
    u_bounds_no_k = np.array([bounds[c][1] for c in cols_no_k])

    sampler_atm = qmc.Sobol(d=len(cols_no_k), scramble=True)
    sample_atm = sampler_atm.random(n=n_atm)
    scaled_atm = qmc.scale(sample_atm, l_bounds_no_k, u_bounds_no_k)
    df_atm = pd.DataFrame(scaled_atm, columns=cols_no_k)

    # Refinement strategy: K stays very close to F
    # Draw log-moneyness from a normal centered at 0 with a tight std (e.g. 0.1)
    np.random.seed(42)
    log_moneyness_atm = np.random.normal(loc=0.0, scale=0.10, size=n_atm)
    
    # Compute K and clip to keep it within the absolute bounds
    df_atm['K'] = df_atm['F'] * np.exp(log_moneyness_atm)
    df_atm['K'] = np.clip(df_atm['K'], bounds['K'][0], bounds['K'][1])

    # ==========================================
    # PART 3: MERGE AND POST-PROCESSING
    # ==========================================
    print("3. Merging and computing enriched features...")
    df_hybrid = pd.concat([df_global, df_atm], ignore_index=True)
    
    # Random shuffle (important so PyTorch batches stay balanced)
    df_hybrid = df_hybrid.sample(frac=1, random_state=42).reset_index(drop=True)

    # Compute the enriched feature (log-moneyness)
    df_hybrid['log_moneyness'] = np.log(df_hybrid['K'] / df_hybrid['F'])

    # Save
    output_path = data_dir / "sabr_hybrid_mesh_features.csv"
    df_hybrid.to_csv(output_path, index=False)
    
    print(f"Hybrid Mesh generated successfully.")
    print(f"   Shape: {df_hybrid.shape}")
    print(f"   Saved to: {output_path}")
    
    # Quick sanity check
    atm_count = len(df_hybrid[np.abs(df_hybrid['log_moneyness']) < 0.05])
    print(f"   Points very close to ATM (|log(K/F)| < 0.05): {atm_count} ({atm_count/n_total*100:.1f}%)")

if __name__ == "__main__":
    generate_hybrid_mesh()