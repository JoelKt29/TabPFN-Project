import jax
import jax.numpy as jnp

def sabr_vol_hagan(K, F, T, alpha, beta, rho, volvol):
    """
    Hagan (2002) formula for the SABR model (lognormal).
    Written with pure jax.numpy (jnp) math functions to allow differentiation.
    """
    # 1. Handle the ATM (At-The-Money) mathematical singularity
    # In JAX both branches of a 'where' are evaluated, so we must avoid
    # division by zero (0/0) when K == F, otherwise the gradient becomes NaN.
    eps = 1e-7
    is_atm = jnp.abs(F - K) < eps
    
    # Nudge K very slightly, only for the non-ATM branch computations
    K_safe = jnp.where(is_atm, F + eps, K)
    
    # 2. General branch computations (Out-of-the-Money / In-the-Money)
    logFK = jnp.log(F / K_safe)
    FK_beta = (F * K_safe)**((1 - beta) / 2)
    
    z = (volvol / alpha) * FK_beta * logFK
    
    # x term: ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
    sqrt_term = jnp.sqrt(1 - 2 * rho * z + z**2)
    x = jnp.log((sqrt_term + z - rho) / (1 - rho))
    
    # Prevent 0/0 on z/x (extra safeguard)
    x_safe = jnp.where(jnp.abs(x) < eps, eps, x)
    z_over_x = z / x_safe
    
    # The three main terms of the Hagan formula
    term1 = alpha / (FK_beta * (1 + ((1 - beta)**2 / 24) * logFK**2 + ((1 - beta)**4 / 1920) * logFK**4))
    
    term3_inner = (((1 - beta)**2 / 24) * (alpha**2 / ((F * K_safe)**(1 - beta))) +
                   (1 / 4) * ((rho * beta * volvol * alpha) / FK_beta) +
                   ((2 - 3 * rho**2) / 24) * volvol**2)
    term3 = 1 + term3_inner * T
    
    vol_not_atm = term1 * z_over_x * term3
    
    # 3. Dedicated ATM branch computation (K = F)
    term1_atm = alpha / (F**(1 - beta))
    term3_inner_atm = (((1 - beta)**2 / 24) * (alpha**2 / (F**(2 - 2 * beta))) +
                       (1 / 4) * ((rho * beta * volvol * alpha) / (F**(1 - beta))) +
                       ((2 - 3 * rho**2) / 24) * volvol**2)
    term3_atm = 1 + term3_inner_atm * T
    
    vol_atm = term1_atm * term3_atm
    
    # 4. Final selection
    return jnp.where(is_atm, vol_atm, vol_not_atm)

# =====================================================================
# JAX WRAPPER: VMAP (vectorization) and JIT (compilation) + GRADIENTS
# =====================================================================

# Vectorize over ALL axes (0, 0, 0, 0, 0, 0, 0) to process each CSV row
# 1. Ensure the vectorization maps over arrays on every axis
sabr_vectorized = jax.vmap(sabr_vol_hagan, in_axes=(0, 0, 0, 0, 0, 0, 0))

def _sabr_sum(K, F, T, alpha, beta, rho, volvol):
    return jnp.sum(sabr_vectorized(K, F, T, alpha, beta, rho, volvol))

@jax.jit
def compute_sabr_with_jax(K, F, T, alpha, beta, rho, volvol):
    K = jnp.atleast_1d(jnp.asarray(K, dtype=jnp.float32))
    
    # 1. Compute the volatility itself
    vol = sabr_vectorized(K, F, T, alpha, beta, rho, volvol)
    
    # 2. Compute exact gradients via automatic differentiation
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

    # Paths
    data_dir = Path(__file__).resolve().parent.parent / "data"
    input_path = data_dir / "sabr_hybrid_mesh_features.csv"
    output_path = data_dir / "sabr_hybrid_mesh_with_derivatives.csv"

    print(f"Loading hybrid mesh: {input_path.name}")
    df = pd.read_csv(input_path)

    # Convert columns to JAX arrays (float32 for speed)
    K = jnp.array(df['K'].values, dtype=jnp.float32)
    F = jnp.array(df['F'].values, dtype=jnp.float32)
    T = jnp.ones_like(K) * 1.0  # On fixe T=1.0 si pas dans le CSV
    alpha = jnp.array(df['alpha'].values, dtype=jnp.float32)
    beta = jnp.array(df['beta'].values, dtype=jnp.float32)
    rho = jnp.array(df['rho'].values, dtype=jnp.float32)
    volvol = jnp.array(df['volvol'].values, dtype=jnp.float32)

    print(f"JAX computation over {len(df)} points (vols + 6 gradients)...")
    
    # Compute volatility
    vols = sabr_vectorized(K, F, T, alpha, beta, rho, volvol)
    
    # Compute gradients (Greeks)
    # Note: jax.grad of the sum returns the per-element gradients
    grads = jax.grad(_sabr_sum, argnums=(0, 1, 3, 4, 5, 6))(K, F, T, alpha, beta, rho, volvol)

    # 3. Append the results to the DataFrame
    df['volatility'] = np.array(vols)
    df['dV_dK']      = np.array(grads[0])
    df['dV_dF']      = np.array(grads[1])
    df['dV_dalpha']  = np.array(grads[2])
    df['dV_dbeta']   = np.array(grads[3])
    df['dV_drho']    = np.array(grads[4])
    df['dV_dvolvol'] = np.array(grads[5])

    # Save for the next step (standardization)
    df.to_csv(output_path, index=False)
    print(f"Done. File saved: {output_path.name}")
    print(df[['volatility', 'dV_dK', 'dV_drho']].head())