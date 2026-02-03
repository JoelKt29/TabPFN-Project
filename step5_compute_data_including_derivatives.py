import numpy as np
import pandas as pd
from pysabr.models.hagan_2002_lognormal_sabr import Hagan2002LognormalSABR
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm

class SABRDerivativesCalculator:
    
    def __init__(self, epsilon=1e-6):
        
        self.eps = epsilon #Step size for finite differences
    
    def compute_all_derivatives(self, f, k, t, v_atm_n, beta, rho, volvol, shift=0.0):
        
        sabr_base = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol)
        v_base = sabr_base.normal_vol(k)
        alpha_base = sabr_base.alpha()
        derivatives = {'volatility': v_base, 'alpha': alpha_base}
        
        # 1. dV/dbeta 
        eps_beta = self.eps
        beta_up = min(beta + eps_beta, 0.9999)  # Keep beta < 1
        beta_down = max(beta - eps_beta, 0.01)  # Keep beta > 0
        
        sabr_beta_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta_up, rho=rho, volvol=volvol)
        
        sabr_beta_down = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta_down, rho=rho, volvol=volvol)
        
        v_beta_up = sabr_beta_up.normal_vol(k)
        v_beta_down = sabr_beta_down.normal_vol(k)
        derivatives['dV_dbeta'] = (v_beta_up - v_beta_down) / (beta_up - beta_down)
        
        # 2. dV/drho
        eps_rho = self.eps
        rho_up = min(rho + eps_rho, 0.9999)
        rho_down = max(rho - eps_rho, -0.9999)
        
        sabr_rho_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho_up, volvol=volvol
        )
        sabr_rho_down = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho_down, volvol=volvol
        )
        
        v_rho_up = sabr_rho_up.normal_vol(k)
        v_rho_down = sabr_rho_down.normal_vol(k)
        derivatives['dV_drho'] = (v_rho_up - v_rho_down) / (rho_up - rho_down)
        
        # 3. dV/dvolvol
        eps_volvol = self.eps
        volvol_up = volvol + eps_volvol
        volvol_down = max(volvol - eps_volvol, 0.01)
        
        sabr_volvol_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol_up
        )
        sabr_volvol_down = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol_down
        )
        
        v_volvol_up = sabr_volvol_up.normal_vol(k)
        v_volvol_down = sabr_volvol_down.normal_vol(k)
        derivatives['dV_dvolvol'] = (v_volvol_up - v_volvol_down) / (volvol_up - volvol_down)
        
        # 4. dV/dv_atm_n
        eps_vatm = self.eps * max(abs(v_atm_n), 0.001)
        vatm_up = v_atm_n + eps_vatm
        vatm_down = max(v_atm_n - eps_vatm, 0.0001)
        
        sabr_vatm_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=vatm_up,
            beta=beta, rho=rho, volvol=volvol
        )
        sabr_vatm_down = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=vatm_down,
            beta=beta, rho=rho, volvol=volvol
        )
        
        v_vatm_up = sabr_vatm_up.normal_vol(k)
        v_vatm_down = sabr_vatm_down.normal_vol(k)
        derivatives['dV_dvatm'] = (v_vatm_up - v_vatm_down) / (vatm_up - vatm_down)
        
        # 5. dV/dF 
        eps_f = self.eps * max(abs(f), 0.01)
        f_up = f + eps_f
        f_down = max(f - eps_f, 0.001)
        
        sabr_f_up = Hagan2002LognormalSABR(
            f=f_up, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol)
        sabr_f_down = Hagan2002LognormalSABR(
            f=f_down, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol)
        
        v_f_up = sabr_f_up.normal_vol(k)
        v_f_down = sabr_f_down.normal_vol(k)
        derivatives['dV_dF'] = (v_f_up - v_f_down) / (f_up - f_down)
        
        # 6. ∂V/∂K
        eps_k = self.eps * max(abs(k), 0.01)
        k_up = k + eps_k
        k_down = max(k - eps_k, 0.001)
        
        v_k_up = sabr_base.normal_vol(k_up)
        v_k_down = sabr_base.normal_vol(k_down)
        derivatives['dV_dK'] = (v_k_up - v_k_down) / (k_up - k_down)
        
        return derivatives


def generate_sabr_with_derivatives(num_samples=5000, num_strikes=8, epsilon=1e-6):
   
 
    np.random.seed(42)
    
    NUM_POINTS = 6
    BETAS = np.linspace(0.25, 0.99, NUM_POINTS)
    RHOS = np.linspace(-0.25, 0.25, NUM_POINTS)
    VOLVOLS = np.linspace(0.15, 0.25, NUM_POINTS)
    ATM_VOLS = np.linspace(0.005, 0.02, NUM_POINTS)
    FORWARDS = np.linspace(0.01, 0.50, NUM_POINTS)
    
    T = 1.0
    SHIFT = 0.0
    
    calculator = SABRDerivativesCalculator(epsilon=epsilon)
    data_list = []
    count = 0
    
    
    for beta in tqdm(BETAS, desc="Generating SABR and Derivatives"):
        for rho in RHOS:
            for volvol in VOLVOLS:
                for v_atm_n in ATM_VOLS:
                    for f in FORWARDS:
                        
                        if count >= num_samples:
                            break
                        
                        try:
                            # Strikes: 0.75f to 1.5f (as Peter specified)
                            strikes = np.linspace(0.75 * f, 1.5 * f, num_strikes)
                            
                            for k in strikes:
                                if count >= num_samples:
                                    break
                                
                                # Compute volatility + all derivatives
                                result = calculator.compute_all_derivatives(
                                    f=f, k=k, t=T, v_atm_n=v_atm_n,
                                    beta=beta, rho=rho, volvol=volvol, shift=SHIFT
                                )
                                
                                log_moneyness = np.log(k / f)
                                
                                row = {
                                    'beta': beta,
                                    'rho': rho,
                                    'volvol': volvol,
                                    'v_atm_n': v_atm_n,
                                    'alpha': result['alpha'],
                                    'F': f,
                                    'K': k,
                                    'log_moneyness': log_moneyness,
                                    'T': T,
                                    'Shift': SHIFT,
                                    'volatility': result['volatility'],
                                    'dV_dbeta': result['dV_dbeta'],
                                    'dV_drho': result['dV_drho'],
                                    'dV_dvolvol': result['dV_dvolvol'],
                                    'dV_dvatm': result['dV_dvatm'],
                                    'dV_dF': result['dV_dF'],
                                    'dV_dK': result['dV_dK'],
                                }
                                
                                data_list.append(row)
                                count += 1
                        
                        except Exception as e:
                            print(f"Warning: Skipped config due to {e}")
                            continue
    
    df = pd.DataFrame(data_list)    
    return df


def scale_data_with_derivatives(df):
    """
    Scale features and outputs (including derivatives)
    
    Returns:
        df_scaled, scaling_params
    """
    
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    derivative_cols = ['dV_dbeta', 'dV_drho', 'dV_dvolvol', 'dV_dvatm', 'dV_dF', 'dV_dK']
    constant_cols = ['T', 'Shift']
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(df[feature_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    vol_min = df['volatility'].min()
    vol_max = df['volatility'].max()
    vol_scaled = (df['volatility'] - vol_min) / (vol_max - vol_min)
    scaler_derivs = {}
    derivs_scaled = pd.DataFrame(index=df.index)
    
    for col in derivative_cols:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df[[col]])
        derivs_scaled[col + '_scaled'] = scaled.flatten()
        scaler_derivs[col] = {
            'min': float(scaler.data_min_[0]),
            'max': float(scaler.data_max_[0])
        }
    
    df_scaled = pd.concat([
        X_scaled_df,
        df[constant_cols],
        pd.DataFrame({'volatility_scaled': vol_scaled}),
        derivs_scaled
    ], axis=1)
    
    scaling_params = {
        'volatility': {'min': float(vol_min), 'max': float(vol_max)},
        'features': {
            col: {
                'min': float(scaler_X.data_min_[i]),
                'max': float(scaler_X.data_max_[i])
            }
            for i, col in enumerate(feature_cols)
        },
        'derivatives': scaler_derivs
    }
    return df_scaled, scaling_params


if __name__ == "__main__":
    df = generate_sabr_with_derivatives(
        num_samples=5000,
        num_strikes=8,
        epsilon=1e-6)
    
    df_scaled, scaling_params = scale_data_with_derivatives(df)
    df.to_csv('sabr_with_derivatives_raw.csv', index=False)
    df_scaled.to_csv('sabr_with_derivatives_scaled.csv', index=False)
    
    with open('scaling_params_derivatives.json', 'w') as f:
        json.dump(scaling_params, f, indent=2)
