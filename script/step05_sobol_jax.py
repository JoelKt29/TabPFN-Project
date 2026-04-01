import numpy as np
import pandas as pd
from step02_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR # Keep for alpha root finding
from step02_jax_sabr import compute_sabr_with_jax # New JAX engine
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm
from pathlib import Path
from scipy.stats import qmc # Sobol

# --- Configuration ---
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Parameter Bounds (Sobol)
L_BOUNDS = [0.01, 0.25, -0.50, 0.15, 0.005]
U_BOUNDS = [0.06, 0.99,  0.50, 0.45, 0.030]

def generate_data_jax_sobol(num_samples=3000, num_strikes=8):
    """
    Generates SABR data using:
    1. Sobol Sequences for parameter coverage (Optimization 1)
    2. JAX AutoDiff for exact derivatives (Optimization 2)
    """
    print(f"Initializing Sobol JAX generation for {num_samples} samples...")
    
    # 1. Sobol Setup
    target_configs = int(np.ceil(num_samples / num_strikes))
    m = int(np.ceil(np.log2(target_configs)))
    real_configs = 2**m
    print(f" -> Generating {real_configs} market configs via Sobol.")

    sampler = qmc.Sobol(d=5, scramble=True)
    sample_raw = sampler.random_base2(m=m)
    params = qmc.scale(sample_raw, L_BOUNDS, U_BOUNDS)
    
    # 2. Main Loop (Sequential is fast enough thanks to JAX)
    data_rows = []
    
    # T and Shift are constant here
    T = 1.0
    SHIFT = 0.0
    
    print("Starting JAX computation...")
    for i in tqdm(range(real_configs)):
        f, beta, rho, volvol, v_atm_n = params[i]
        
        # A. Find Alpha (Root Finding - Classical Numpy)
        # We use the legacy class because creating a differentiable root finder in JAX is complex 
        # and unnecessary for just one parameter.
        try:
            temp_model = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, 
                                                beta=beta, rho=rho, volvol=volvol)
            alpha = temp_model.alpha()
        except Exception:
            continue # Skip invalid configs
            
        # B. Define Strikes (Vectorized)
        strikes = np.linspace(0.75 * f, 1.5 * f, num_strikes)
        
        # C. Compute Vol and Derivatives (JAX - Exact & Fast)
        # This single call computes all strikes and all Greeks at once
        vols, grads = compute_sabr_with_jax(strikes, f, T, alpha, beta, rho, volvol)
        
        # D. Chain Rule for dV/dV_atm
        # JAX gives us dV/dAlpha. The dataset expects dV/dV_atm.
        # Analytic approximation: Sigma_ATM ~= Alpha * f^(beta-1)
        # Therefore: Alpha ~= Sigma_ATM * f^(1-beta)
        # derivative dAlpha/dSigma_ATM ~= f^(1-beta)
        d_alpha_d_vatm = f**(1 - beta)
        dv_dvatm = grads['dV_dalpha'] * d_alpha_d_vatm
        
        # E. Store Data
        for j, k in enumerate(strikes):
            # Sanity check
            if np.isnan(vols[j]) or vols[j] < 0: continue
            
            data_rows.append({
                # Features
                'beta': beta, 'rho': rho, 'volvol': volvol, 'v_atm_n': v_atm_n,
                'alpha': alpha, 'F': f, 'K': k, 'log_moneyness': np.log(k/f),
                'T': T, 'Shift': SHIFT, 
                
                # Target
                'volatility': float(vols[j]),
                
                # Derivatives (Exact from JAX)
                'dV_dbeta': float(grads['dV_dbeta'][j]), 
                'dV_drho': float(grads['dV_drho'][j]), 
                'dV_dvolvol': float(grads['dV_dvolvol'][j]),
                'dV_dF': float(grads['dV_dF'][j]), 
                'dV_dK': float(grads['dV_dK'][j]),
                
                # Derivative (via Chain Rule)
                'dV_dvatm': float(dv_dvatm[j]) 
            })

    return pd.DataFrame(data_rows)

def scale_and_save(df):
    """Identical to previous scaling logic"""
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    derivative_cols = ['dV_dbeta', 'dV_drho', 'dV_dvolvol', 'dV_dvatm', 'dV_dF', 'dV_dK']
    constant_cols = ['T', 'Shift']
    
    print("Scaling data...")
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(df[feature_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    vol_min, vol_max = df['volatility'].min(), df['volatility'].max()
    vol_scaled = (df['volatility'] - vol_min) / (vol_max - vol_min)
    
    scaler_derivs = {}
    derivs_scaled = pd.DataFrame(index=df.index)
    
    for col in derivative_cols:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df[[col]])
        derivs_scaled[col + '_scaled'] = scaled.flatten()
        scaler_derivs[col] = {'min': float(scaler.data_min_[0]), 'max': float(scaler.data_max_[0])}
    
    df_scaled = pd.concat([X_scaled_df, df[constant_cols],
        pd.DataFrame({'volatility_scaled': vol_scaled}), derivs_scaled], axis=1)
    
    scaling_params = {
        'volatility': {'min': float(vol_min), 'max': float(vol_max)},
        'features': {c: {'min': float(scaler_X.data_min_[i]), 'max': float(scaler_X.data_max_[i])} 
                     for i, c in enumerate(feature_cols)},
        'derivatives': scaler_derivs}
    
    df_scaled.to_csv(data_dir / 'sabr_with_derivatives_scaled.csv', index=False)
    with open(data_dir / 'scaling_params_derivatives.json', 'w') as f:
        json.dump(scaling_params, f, indent=2)
    print("Success: JAX+Sobol Dataset generated.")

if __name__ == "__main__":
    # Faster generation thanks to JAX
    df = generate_data_jax_sobol(3000, 8)
    if not df.empty:
        scale_and_save(df)
    else:
        print("Error: No data generated.")