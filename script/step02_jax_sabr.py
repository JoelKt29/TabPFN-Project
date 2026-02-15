import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad

# Enable 64-bit precision (Critical for Finance)
jax.config.update("jax_enable_x64", True)

@jit
def _hagan_vol_jit(k, f, t, alpha, beta, rho, volvol):
    """
    Hagan 2002 Lognormal SABR formula implemented in JAX.
    Handles singular cases (ATM) via jnp.where for differentiable branching.
    """
    # Constants
    eps = 1e-07
    
    # Pre-calculations
    # We use jnp.maximum for safety inside log, though Sobol bounds should prevent k<=0
    logfk = jnp.log(f / jnp.maximum(k, 1e-8))
    fkbeta = (f * k)**(1 - beta)
    
    # Main terms
    a = (1 - beta)**2 * alpha**2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / (fkbeta**0.5)
    c = (2 - 3 * rho**2) * volvol**2 / 24
    d = fkbeta**0.5
    v = (1 - beta)**2 * logfk**2 / 24
    w = (1 - beta)**4 * logfk**4 / 1920
    z = volvol * (fkbeta**0.5) * logfk / alpha
    
    # x(z) function
    # We use a safe computation for sqrt to avoid NaNs if argument is slightly negative due to float errors
    arg_sqrt = jnp.maximum(1 - 2*rho*z + z**2, 1e-10)
    x = jnp.log((jnp.sqrt(arg_sqrt) + z - rho) / (1 - rho))
    
    # Branching logic (ATM vs OTM)
    # If z is very close to 0, x(z) ~ z, and we must avoid division by zero
    numerator = alpha * (1 + (a + b + c) * t)
    denominator_atm = d * (1 + v + w)
    
    # Standard Case (|z| > eps)
    vol_standard = (numerator * z) / (denominator_atm * x)
    
    # ATM Case (|z| <= eps)
    vol_atm = numerator / denominator_atm
    
    return jnp.where(jnp.abs(z) > eps, vol_standard, vol_atm)

# --- Vectorized & Differentiable Interface ---

# We transform the function to compute value AND gradients simultaneously.
# argnums corresponds to: 0:k, 1:f, 2:t, 3:alpha, 4:beta, 5:rho, 6:volvol
# We want gradients w.r.t: K, F, Alpha, Beta, Rho, Volvol
_sabr_value_and_grads = value_and_grad(_hagan_vol_jit, argnums=(0, 1, 3, 4, 5, 6))

# We vectorize it to process multiple strikes (K) at once for a single SABR config
# signature: (k array, scalar, scalar...) -> (vol array, (dk array, df array...))
vectorized_sabr = jax.vmap(_sabr_value_and_grads, in_axes=(0, None, None, None, None, None, None))

def compute_sabr_with_jax(k_array, f, t, alpha, beta, rho, volvol):
    """
    Wrapper to call JAX optimized function.
    Returns: (volatilities, dictionary_of_derivatives)
    """
    # JAX inputs
    k_jax = jnp.array(k_array)
    
    # Compute
    vols, grads = vectorized_sabr(k_jax, f, t, alpha, beta, rho, volvol)
    
    # Unpack gradients tuple
    d_k, d_f, d_alpha, d_beta, d_rho, d_volvol = grads
    
    # Convert to dictionary of numpy arrays (for compatibility)
    return np.array(vols), {
        'dV_dK': np.array(d_k),
        'dV_dF': np.array(d_f),
        'dV_dalpha': np.array(d_alpha),
        'dV_dbeta': np.array(d_beta),
        'dV_drho': np.array(d_rho),
        'dV_dvolvol': np.array(d_volvol)
    }

import numpy as np # standard numpy for the wrapper return