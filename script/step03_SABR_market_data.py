import numpy as np
import pandas as pd
from script.step02_jax_sabr import Hagan2002LognormalSABR
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm 
from pathlib import Path
from scipy.stats import qmc # Crucial import for Sobol

# --- Configuration ---
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# Global Parameters
T = 1.0
SHIFT = 0.0
TARGET_SAMPLES = 5000 

# Definition of Sampling Bounds
# Order: [beta, rho, volvol, v_atm_n, f, moneyness_factor]
# moneyness_factor is used to calculate K = F * factor
PARAM_NAMES = ['beta', 'rho', 'volvol', 'v_atm_n', 'f', 'moneyness']
L_BOUNDS = [0.25, -0.50, 0.15, 0.005, 0.01, 0.75] # Lower bounds
U_BOUNDS = [0.99,  0.50, 0.45, 0.030, 0.06, 1.50] # Upper bounds

def generate_sobol_data():
    print(f"Generating data via Sobol (Target: {TARGET_SAMPLES} points)...")
    
    # 1. Calculate required power of 2 (Sobol is optimal with 2^m)
    m = int(np.ceil(np.log2(TARGET_SAMPLES)))
    num_samples_sobol = 2**m
    print(f" -> Adjusting to {num_samples_sobol} points (next power of 2)")

    # 2. Initialize Sobol engine (6 dimensions)
    sampler = qmc.Sobol(d=6, scramble=True)
    sample_raw = sampler.random_base2(m=m)
    
    # 3. Scale to real bounds
    params = qmc.scale(sample_raw, L_BOUNDS, U_BOUNDS)
    
    data_list = []
    
    # 4. Calculation loop
    for i in tqdm(range(num_samples_sobol)):
        # Extract parameters
        beta, rho, volvol, v_atm_n, f, moneyness = params[i]
        
        # Calculate Strike K based on generated moneyness
        k = f * moneyness

        # SABR Model
        try:
            sabr = Hagan2002LognormalSABR(
                f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n,
                beta=beta, rho=rho, volvol=volvol)
            
            alpha = sabr.alpha()
            v_normal = sabr.normal_vol(k)
            
            # Basic sanity check to avoid NaN or absurd values
            if np.isnan(v_normal) or v_normal < 0 or v_normal > 1.0:
                continue

            log_moneyness = np.log(k / f)

            data_list.append({
                'beta': beta,
                'rho': rho,
                'volvol': volvol,
                'v_atm_n': v_atm_n,
                'alpha': alpha,             
                'F': f,
                'K': k,
                'log_moneyness': log_moneyness,
                'T': T, 
                'Shift': SHIFT,
                'volatility_output': v_normal
            })
        except Exception as e:
            continue

    df_sabr = pd.DataFrame(data_list)
    print(f"Valid data generated: {len(df_sabr)}")
    return df_sabr

def process_and_save(df_sabr):
    # Scaling preparation 
    X_raw = df_sabr.drop(columns=['volatility_output'])
    y_raw = df_sabr['volatility_output']

    non_constant_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    X_variable = X_raw[non_constant_cols]
    X_constant = X_raw.drop(columns=non_constant_cols)

    # Scaling data between -1 and 1
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X_variable)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_variable.columns, index=X_raw.index)

    # Scaling volatility between 0 and 1
    y_min = y_raw.min()
    y_max = y_raw.max()
    y_scaled = (y_raw - y_min) / (y_max - y_min)

    # Re-adding constant variables
    df_final = pd.concat([X_scaled_df, X_constant], axis=1)
    df_final['SABR_volatility'] = y_scaled

    # Saving scaling parameters
    scaling_params = {
        'y_min': float(y_min),
        'y_max': float(y_max),
        'X_min': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_min_)},
        'X_max': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_max_)}
    }
    
    with open(data_dir / "scaling_params_recovery.json", 'w') as f:
        json.dump(scaling_params, f, indent=4)

    df_final.to_csv(data_dir / "sabr_market_data.csv", index=False)
    print("Data saved successfully.")

if __name__ == "__main__":
    df = generate_sobol_data()
    process_and_save(df)