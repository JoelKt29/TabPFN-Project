import numpy as np
import pandas as pd
from step02_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from scipy.stats import qmc # Import Sobol

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Bounds for step 05 (F, beta, rho, volvol, v_atm_n)
L_BOUNDS = [0.01, 0.25, -0.50, 0.15, 0.005]
U_BOUNDS = [0.06, 0.99,  0.50, 0.45, 0.030]

def compute_config_samples(args):
    """
    Computes volatility and derivatives (Greeks) for a given SABR configuration.
    """
    f, beta, rho, volvol, v_atm_n, T, SHIFT, num_strikes, eps = args
    results = []
    
    try:
        # Base model for derivatives
        base = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol)
        
        # Beta gradients (Finite Differences)
        b_up, b_dn = min(beta + eps, 0.9999), max(beta - eps, 0.01)
        m_b_up = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=b_up, rho=rho, volvol=volvol)
        m_b_dn = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=b_dn, rho=rho, volvol=volvol)
        
        # Rho gradients
        r_up, r_dn = min(rho + eps, 0.9999), max(rho - eps, -0.9999)
        m_r_up = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=beta, rho=r_up, volvol=volvol)
        m_r_dn = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=beta, rho=r_dn, volvol=volvol)
        
        # Volvol gradients
        vv_up, vv_dn = volvol + eps, max(volvol - eps, 0.01)
        m_vv_up = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=vv_up)
        m_vv_dn = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=vv_dn)
        
        # V_atm gradients
        va_eps = eps * max(abs(v_atm_n), 0.001)
        va_up, va_dn = v_atm_n + va_eps, max(v_atm_n - va_eps, 0.0001)
        m_va_up = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=va_up, beta=beta, rho=rho, volvol=volvol)
        m_va_dn = Hagan2002LognormalSABR(f=f, shift=SHIFT, t=T, v_atm_n=va_dn, beta=beta, rho=rho, volvol=volvol)
        
        # F gradients
        f_eps = eps * max(abs(f), 0.01)
        f_up, f_dn = f + f_eps, max(f - f_eps, 0.001)
        m_f_up = Hagan2002LognormalSABR(f=f_up, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol)
        m_f_dn = Hagan2002LognormalSABR(f=f_dn, shift=SHIFT, t=T, v_atm_n=v_atm_n, beta=beta, rho=rho, volvol=volvol)

        alpha_val = base.alpha()
        
        # Generating Strikes: Covering a wide range around At-The-Money (75% to 150%)
        strikes = np.linspace(0.75 * f, 1.5 * f, num_strikes)

        for k in strikes:
            v_base = base.normal_vol(k)
            # Sanity check
            if np.isnan(v_base) or v_base < 0: continue

            k_eps = eps * max(abs(k), 0.01)

            # Finite difference calculations for all Greeks
            dv_db = (m_b_up.normal_vol(k) - m_b_dn.normal_vol(k)) / (b_up - b_dn)
            dv_dr = (m_r_up.normal_vol(k) - m_r_dn.normal_vol(k)) / (r_up - r_dn)
            dv_dvv = (m_vv_up.normal_vol(k) - m_vv_dn.normal_vol(k)) / (vv_up - vv_dn)
            dv_dva = (m_va_up.normal_vol(k) - m_va_dn.normal_vol(k)) / (va_up - va_dn)
            dv_df = (m_f_up.normal_vol(k) - m_f_dn.normal_vol(k)) / (f_up - f_dn)
            dv_dk = (base.normal_vol(k + k_eps) - base.normal_vol(k - k_eps)) / (2 * k_eps)

            results.append({
                'beta': beta, 'rho': rho, 'volvol': volvol, 'v_atm_n': v_atm_n,
                'alpha': alpha_val, 'F': f, 'K': k, 'log_moneyness': np.log(k/f),
                'T': T, 'Shift': SHIFT, 'volatility': v_base,
                'dV_dbeta': dv_db, 'dV_drho': dv_dr, 'dV_dvolvol': dv_dvv,
                'dV_dvatm': dv_dva, 'dV_dF': dv_df, 'dV_dK': dv_dk})
                
    except Exception as e:
        return [] # Ignore bad configs to maintain dataset integrity
        
    return results

def fast_generate_sabr_sobol(num_samples=3000, num_strikes=8):
    """
    Optimized generation using Sobol Sequences.
    """
    print(f"Initializing Sobol for {num_samples} configurations...")
    
    # 1. Power of 2 Adjustment
    # num_samples is total rows (volatilities), but sampler generates MARKET CONFIGS.
    # We divide by num_strikes to get approx number of configs needed.
    target_configs = int(np.ceil(num_samples / num_strikes))
    m = int(np.ceil(np.log2(target_configs)))
    real_configs = 2**m
    
    print(f" -> Generating {real_configs} market configs (total approx {real_configs * num_strikes} rows)")

    # 2. Sobol Sampler (5 dims: F, beta, rho, volvol, v_atm)
    sampler = qmc.Sobol(d=5, scramble=True)
    sample_raw = sampler.random_base2(m=m)
    params = qmc.scale(sample_raw, L_BOUNDS, U_BOUNDS)
    
    configs = []
    for i in range(real_configs):
        f, beta, rho, volvol, v_atm_n = params[i]
        configs.append((
            f, beta, rho, volvol, v_atm_n,
            1.0, 0.0, num_strikes, 1e-6    # T, Shift, strikes, eps
        ))

    final_data = []
    # Using ProcessPoolExecutor to speed up derivative calculations
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_config_samples, configs), total=len(configs)))
    
    for res_list in results:
        final_data.extend(res_list)
    
    return pd.DataFrame(final_data)


def scale_data_with_derivatives(df):
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    derivative_cols = ['dV_dbeta', 'dV_drho', 'dV_dvolvol', 'dV_dvatm', 'dV_dF', 'dV_dK']
    constant_cols = ['T', 'Shift']
    
    # Scale Features
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(df[feature_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    # Scale Target (Volatility)
    vol_min = df['volatility'].min()
    vol_max = df['volatility'].max()
    vol_scaled = (df['volatility'] - vol_min) / (vol_max - vol_min)
    
    scaler_derivs = {}
    derivs_scaled = pd.DataFrame(index=df.index)
    
    # Scale Derivatives individually
    for col in derivative_cols:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df[[col]])
        derivs_scaled[col + '_scaled'] = scaled.flatten()
        scaler_derivs[col] = {
            'min': float(scaler.data_min_[0]),
            'max': float(scaler.data_max_[0])
        }
    
    df_scaled = pd.concat([X_scaled_df, df[constant_cols],
        pd.DataFrame({'volatility_scaled': vol_scaled}),derivs_scaled], axis=1)
    
    scaling_params = {
        'volatility': {'min': float(vol_min), 'max': float(vol_max)},
        'features': {
            col: {'min': float(scaler_X.data_min_[i]),'max': float(scaler_X.data_max_[i])}
            for i, col in enumerate(feature_cols)},
        'derivatives': scaler_derivs}
    return df_scaled, scaling_params

if __name__ == "__main__":
    # Generating approx 3000 rows (sufficient with Sobol)
    df = fast_generate_sabr_sobol(3000, 8)
    
    if len(df) > 0:
        df_scaled, scaling_params = scale_data_with_derivatives(df)
        df_scaled.to_csv(data_dir / 'sabr_with_derivatives_scaled.csv', index=False)
        
        with open(data_dir / 'scaling_params_derivatives.json', 'w') as f:
            json.dump(scaling_params, f, indent=2)
        print("Generation and saving completed successfully.")
    else:
        print("Error: No data generated.")
