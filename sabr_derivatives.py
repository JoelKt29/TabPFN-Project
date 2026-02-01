"""
SABR Derivatives Computation Module
Calculates analytical derivatives (Greeks) for SABR model
These will be used to train TabPFN on both values AND gradients
"""

import numpy as np
from typing import Tuple, Dict
import pandas as pd
from hagan_2002_lognormal_sabr import Hagan2002LognormalSABR


class SABRGreeks:
    """
    Compute SABR sensitivities (Greeks) using finite differences.
    
    Greeks computed:
    - dV/dF (delta-like sensitivity to forward)
    - dV/dK (sensitivity to strike) 
    - dV/dBeta (sensitivity to beta parameter)
    - dV/dRho (sensitivity to rho parameter)
    - dV/dVolvol (vega-like sensitivity to vol-of-vol)
    - dV/dAlpha (sensitivity to alpha parameter)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small perturbation for finite differences
        """
        self.epsilon = epsilon
    
    def compute_all_greeks(
        self, 
        f: float,
        k: float, 
        t: float,
        v_atm_n: float,
        beta: float,
        rho: float,
        volvol: float,
        shift: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute all SABR Greeks using finite differences.
        
        Returns:
            Dictionary with all sensitivities
        """
        
        # Base volatility
        sabr_base = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol
        )
        v_base = sabr_base.normal_vol(k)
        alpha_base = sabr_base.alpha()
        
        greeks = {}
        
        # 1. Sensitivity to Forward (dV/dF)
        eps_f = self.epsilon * max(abs(f), 0.01)
        sabr_f_up = Hagan2002LognormalSABR(
            f=f + eps_f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol
        )
        v_f_up = sabr_f_up.normal_vol(k)
        greeks['dV_dF'] = (v_f_up - v_base) / eps_f
        
        # 2. Sensitivity to Strike (dV/dK)
        eps_k = self.epsilon * max(abs(k), 0.01)
        v_k_up = sabr_base.normal_vol(k + eps_k)
        greeks['dV_dK'] = (v_k_up - v_base) / eps_k
        
        # 3. Sensitivity to Beta (dV/dBeta)
        eps_beta = self.epsilon
        beta_perturbed = min(beta + eps_beta, 0.9999)  # Keep beta < 1
        sabr_beta_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta_perturbed, rho=rho, volvol=volvol
        )
        v_beta_up = sabr_beta_up.normal_vol(k)
        greeks['dV_dBeta'] = (v_beta_up - v_base) / eps_beta
        
        # 4. Sensitivity to Rho (dV/dRho)
        eps_rho = self.epsilon
        rho_perturbed = np.clip(rho + eps_rho, -0.9999, 0.9999)
        sabr_rho_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho_perturbed, volvol=volvol
        )
        v_rho_up = sabr_rho_up.normal_vol(k)
        greeks['dV_dRho'] = (v_rho_up - v_base) / eps_rho
        
        # 5. Sensitivity to Volvol (dV/dVolvol)
        eps_volvol = self.epsilon
        sabr_volvol_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol + eps_volvol
        )
        v_volvol_up = sabr_volvol_up.normal_vol(k)
        greeks['dV_dVolvol'] = (v_volvol_up - v_base) / eps_volvol
        
        # 6. Sensitivity to ATM Vol (dV/dV_ATM)
        eps_vatm = self.epsilon * max(abs(v_atm_n), 0.001)
        sabr_vatm_up = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n + eps_vatm,
            beta=beta, rho=rho, volvol=volvol
        )
        v_vatm_up = sabr_vatm_up.normal_vol(k)
        greeks['dV_dVatm'] = (v_vatm_up - v_base) / eps_vatm
        
        # Store base values too
        greeks['volatility'] = v_base
        greeks['alpha'] = alpha_base
        
        return greeks
    
    def compute_second_order_greeks(
        self,
        f: float,
        k: float, 
        t: float,
        v_atm_n: float,
        beta: float,
        rho: float,
        volvol: float,
        shift: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute second-order derivatives (gamma-like terms).
        Useful for understanding curvature of the volatility surface.
        """
        
        sabr_base = Hagan2002LognormalSABR(
            f=f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol
        )
        v_base = sabr_base.normal_vol(k)
        
        greeks_2nd = {}
        
        # d²V/dF² (Gamma equivalent)
        eps_f = self.epsilon * max(abs(f), 0.01)
        
        sabr_f_up = Hagan2002LognormalSABR(
            f=f + eps_f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol
        )
        sabr_f_down = Hagan2002LognormalSABR(
            f=f - eps_f, shift=shift, t=t, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol
        )
        
        v_f_up = sabr_f_up.normal_vol(k)
        v_f_down = sabr_f_down.normal_vol(k)
        
        greeks_2nd['d2V_dF2'] = (v_f_up - 2*v_base + v_f_down) / (eps_f**2)
        
        # d²V/dK² (Convexity in strike)
        eps_k = self.epsilon * max(abs(k), 0.01)
        v_k_up = sabr_base.normal_vol(k + eps_k)
        v_k_down = sabr_base.normal_vol(k - eps_k)
        
        greeks_2nd['d2V_dK2'] = (v_k_up - 2*v_base + v_k_down) / (eps_k**2)
        
        return greeks_2nd


def generate_dataset_with_greeks(
    num_samples: int = 5000,
    num_strikes: int = 8,
    include_second_order: bool = False,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate SABR dataset with both values AND derivatives.
    
    Args:
        num_samples: Number of samples to generate
        num_strikes: Number of strikes per configuration
        include_second_order: Whether to include second derivatives
        seed: Random seed
        
    Returns:
        DataFrame with features, target volatility, and all Greeks
    """
    
    np.random.seed(seed)
    
    # Parameter grids (same as your Statap2.py)
    NUM_POINTS = 6
    BETAS = np.linspace(0.25, 0.99, NUM_POINTS)
    RHOS = np.linspace(-0.25, 0.25, NUM_POINTS)
    VOLVOLS = np.linspace(0.15, 0.25, NUM_POINTS)
    ATM_VOLS = np.linspace(0.005, 0.02, NUM_POINTS)
    FORWARDS = np.linspace(0.01, 0.50, NUM_POINTS)
    
    T = 1.0
    SHIFT = 0.0
    
    greek_calculator = SABRGreeks(epsilon=1e-6)
    data_list = []
    count = 0
    
    print(f"Generating {num_samples} samples with Greeks...")
    
    from tqdm import tqdm
    
    for beta in tqdm(BETAS, desc="Generating SABR + Greeks"):
        for rho in RHOS:
            for volvol in VOLVOLS:
                for v_atm_n in ATM_VOLS:
                    for f in FORWARDS:
                        
                        if count >= num_samples:
                            break
                        
                        try:
                            strikes = np.linspace(0.75 * f, 1.25 * f, num_strikes)
                            
                            for k in strikes:
                                if count >= num_samples:
                                    break
                                
                                # Compute all Greeks
                                greeks = greek_calculator.compute_all_greeks(
                                    f=f, k=k, t=T, v_atm_n=v_atm_n,
                                    beta=beta, rho=rho, volvol=volvol, shift=SHIFT
                                )
                                
                                # Optionally compute second order
                                if include_second_order:
                                    greeks_2nd = greek_calculator.compute_second_order_greeks(
                                        f=f, k=k, t=T, v_atm_n=v_atm_n,
                                        beta=beta, rho=rho, volvol=volvol, shift=SHIFT
                                    )
                                    greeks.update(greeks_2nd)
                                
                                log_moneyness = np.log(k / f)
                                
                                # Combine features and targets
                                row = {
                                    'beta': beta,
                                    'rho': rho,
                                    'volvol': volvol,
                                    'v_atm_n': v_atm_n,
                                    'alpha': greeks['alpha'],
                                    'F': f,
                                    'K': k,
                                    'log_moneyness': log_moneyness,
                                    'T': T,
                                    'Shift': SHIFT,
                                }
                                
                                # Add all Greeks
                                row.update(greeks)
                                
                                data_list.append(row)
                                count += 1
                        
                        except Exception as e:
                            print(f"Warning: Skipped configuration due to {e}")
                            continue
    
    df = pd.DataFrame(data_list)
    print(f"\n✅ Generated {len(df)} samples with Greeks")
    
    return df


if __name__ == "__main__":
    # Example usage
    df_with_greeks = generate_dataset_with_greeks(
        num_samples=5000,
        num_strikes=8,
        include_second_order=True
    )
    
    print("\nDataFrame columns:")
    print(df_with_greeks.columns.tolist())
    
    print("\nFirst few rows:")
    print(df_with_greeks.head())
    
    print("\nGreeks statistics:")
    greek_cols = [col for col in df_with_greeks.columns if 'dV_' in col or 'd2V_' in col]
    print(df_with_greeks[greek_cols].describe())
    
    # Save
    df_with_greeks.to_csv('sabr_data_with_greeks.csv', index=False)
    print("\n✅ Saved to 'sabr_data_with_greeks.csv'")
