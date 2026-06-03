import jax
import jax.numpy as jnp

def sabr_vol_hagan(K, F, T, alpha, beta, rho, volvol):
    """
    Formule de Hagan (2002) pour le modèle SABR (Lognormal).
    Écrite en fonctions mathématiques pures jax.numpy (jnp) pour la différenciation.
    """
    # 1. Gestion de la singularité mathématique ATM (At-The-Money)
    # En JAX, les deux branches d'un 'where' sont évaluées. Il faut empêcher la 
    # division par zéro (0/0) si K == F, sinon le gradient renverra NaN.
    eps = 1e-7
    is_atm = jnp.abs(F - K) < eps
    
    # On décale très légèrement K juste pour les calculs de la branche non-ATM
    K_safe = jnp.where(is_atm, F + eps, K)
    
    # 2. Calculs de la branche générale (Out-of-the-Money / In-the-Money)
    logFK = jnp.log(F / K_safe)
    FK_beta = (F * K_safe)**((1 - beta) / 2)
    
    z = (volvol / alpha) * FK_beta * logFK
    
    # Terme x : ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
    sqrt_term = jnp.sqrt(1 - 2 * rho * z + z**2)
    x = jnp.log((sqrt_term + z - rho) / (1 - rho))
    
    # Prévention du 0/0 sur z/x (sécurité supplémentaire)
    x_safe = jnp.where(jnp.abs(x) < eps, eps, x)
    z_over_x = z / x_safe
    
    # Les 3 grands termes de la formule de Hagan
    term1 = alpha / (FK_beta * (1 + ((1 - beta)**2 / 24) * logFK**2 + ((1 - beta)**4 / 1920) * logFK**4))
    
    term3_inner = (((1 - beta)**2 / 24) * (alpha**2 / ((F * K_safe)**(1 - beta))) +
                   (1 / 4) * ((rho * beta * volvol * alpha) / FK_beta) +
                   ((2 - 3 * rho**2) / 24) * volvol**2)
    term3 = 1 + term3_inner * T
    
    vol_not_atm = term1 * z_over_x * term3
    
    # 3. Calculs de la branche spécifique ATM (K = F)
    term1_atm = alpha / (F**(1 - beta))
    term3_inner_atm = (((1 - beta)**2 / 24) * (alpha**2 / (F**(2 - 2 * beta))) +
                       (1 / 4) * ((rho * beta * volvol * alpha) / (F**(1 - beta))) +
                       ((2 - 3 * rho**2) / 24) * volvol**2)
    term3_atm = 1 + term3_inner_atm * T
    
    vol_atm = term1_atm * term3_atm
    
    # 4. Choix final
    return jnp.where(is_atm, vol_atm, vol_not_atm)

# =====================================================================
# WRAPPER JAX : VMAP (Vectorisation) et JIT (Compilation) + GRADIENTS
# =====================================================================

# On vectorise sur TOUS les axes (0, 0, 0, 0, 0, 0, 0) pour traiter chaque ligne du CSV
# 1. On s'assure que la vectorisation accepte bien des tableaux (arrays) sur tous les axes
sabr_vectorized = jax.vmap(sabr_vol_hagan, in_axes=(0, 0, 0, 0, 0, 0, 0))

def _sabr_sum(K, F, T, alpha, beta, rho, volvol):
    return jnp.sum(sabr_vectorized(K, F, T, alpha, beta, rho, volvol))

@jax.jit
def compute_sabr_with_jax(K, F, T, alpha, beta, rho, volvol):
    K = jnp.atleast_1d(jnp.asarray(K, dtype=jnp.float32))
    
    # 1. Calcul de la volatilité pure
    vol = sabr_vectorized(K, F, T, alpha, beta, rho, volvol)
    
    # 2. Calcul des gradients exacts avec la différenciation automatique
    grads = jax.grad(_sabr_sum, argnums=(0, 1, 2, 3, 4, 5, 6))(K, F, T, alpha, beta, rho, volvol)

    grad_dict = {
        'dV_dK':     grads[0],
        'dV_dF':     grads[1],
        'dV_dT':     grads[2],
        'dV_dalpha': grads[3],
        'dV_dbeta':  grads[4],
        'dV_drho':   grads[5],
        'dV_dvolvol':grads[6],
    }
    
    return vol, grad_dict



if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # Chemins
    data_dir = Path(__file__).resolve().parent.parent / "data"
    input_path = data_dir / "sabr_hybrid_mesh_features.csv"
    output_path = data_dir / "sabr_hybrid_mesh_with_derivatives.csv"

    print(f"📂 Chargement du mesh hybride : {input_path.name}")
    df = pd.read_csv(input_path)

    # Conversion des colonnes en tableaux JAX (float32 pour la rapidité)
    K = jnp.array(df['K'].values, dtype=jnp.float32)
    F = jnp.array(df['F'].values, dtype=jnp.float32)
    T = jnp.ones_like(K) * 1.0  # On fixe T=1.0 si pas dans le CSV
    alpha = jnp.array(df['alpha'].values, dtype=jnp.float32)
    beta = jnp.array(df['beta'].values, dtype=jnp.float32)
    rho = jnp.array(df['rho'].values, dtype=jnp.float32)
    volvol = jnp.array(df['volvol'].values, dtype=jnp.float32)

    print(f"🚀 Calcul JAX sur {len(df)} points (Vols + 6 Gradients)...")
    
    # Calcul de la Volatilité
    vols = sabr_vectorized(K, F, T, alpha, beta, rho, volvol)
    
    # Calcul des Gradients (Greeks)
    # Note : jax.grad de la somme renvoie les gradients de chaque élément
    grads = jax.grad(_sabr_sum, argnums=(0, 1, 3, 4, 5, 6))(K, F, T, alpha, beta, rho, volvol)

    # 3. On ajoute les résultats au DataFrame
    df['volatility'] = np.array(vols)
    df['dV_dK']      = np.array(grads[0])
    df['dV_dF']      = np.array(grads[1])
    df['dV_dalpha']  = np.array(grads[2])
    df['dV_dbeta']   = np.array(grads[3])
    df['dV_drho']    = np.array(grads[4])
    df['dV_dvolvol'] = np.array(grads[5])

    # Sauvegarde pour l'étape suivante (Standardization)
    df.to_csv(output_path, index=False)
    print(f"✅ Terminé ! Fichier sauvegardé : {output_path.name}")
    print(df[['volatility', 'dV_dK', 'dV_drho']].head())