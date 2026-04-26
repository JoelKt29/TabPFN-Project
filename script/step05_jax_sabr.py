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

# 1. On crée d'abord l'opérateur mathématique pur (pour un seul point scalaire)

# 2. On vectorise (vmap) cet opérateur pour qu'il accepte des tableaux de 200 points
sabr_vectorized = jax.vmap(sabr_vol_hagan, in_axes=(0, None, None, None, None, None, None))


def _sabr_sum(K, F, T, alpha, beta, rho, volvol):
    return jnp.sum(sabr_vectorized(K, F, T, alpha, beta, rho, volvol))

sabr_val_and_grad_vectorized = jax.value_and_grad(_sabr_sum, argnums=(0, 1, 2, 3, 4, 5, 6))
# 3. On compile le tout (jit) pour que ça s'exécute à la vitesse de la lumière
@jax.jit
def compute_sabr_with_jax(K, F, T, alpha, beta, rho, volvol):
    K = jnp.atleast_1d(jnp.asarray(K, dtype=jnp.float32))
    _, grads = sabr_val_and_grad_vectorized(K, F, T, alpha, beta, rho, volvol)
    vol = sabr_vectorized(K, F, T, alpha, beta, rho, volvol)  # vols réels

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

# --- Petit bloc de test rapide ---
if __name__ == "__main__":
    import numpy as np
    
    # Simulation de 3 scénarios de marché
    K = jnp.array([0.04, 0.05, 0.06])
    F = jnp.array([0.05, 0.05, 0.05])
    T = jnp.array([1.0, 1.0, 1.0])
    alpha = jnp.array([0.03, 0.03, 0.03])
    beta = jnp.array([0.5, 0.5, 0.5])
    rho = jnp.array([-0.2, -0.2, -0.2])
    volvol = jnp.array([0.4, 0.4, 0.4])
    
    print("🚀 Test du moteur JAX vectorisé...")
    vols, grads = compute_sabr_with_jax(K, F, T, alpha, beta, rho, volvol)
    
    print(f"\n✅ Volatilités calculées : {np.array(vols)}")
    print(f"✅ Skew (dV/dK) calculé : {np.array(grads['dV_dK'])}")
    print("\nSi des chiffres s'affichent, le moteur est opérationnel pour le graphique !")