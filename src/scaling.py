import pandas as pd
import numpy as np
import json
from pathlib import Path

# Path configuration
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"

def standardize_hybrid_dataset():
    print("--- Z-Score Standardization (StandardScaler) ---")
    
    # 1. Load the dataset containing the prices and JAX derivatives
    # (Replace with the exact filename produced by the JAX step)
    input_path = data_dir / "sabr_hybrid_mesh_with_derivatives.csv"
    
    if not input_path.exists():
        print(f"Error: file {input_path.name} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Column definitions
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility', 'dV_dbeta', 'dV_drho', 'dV_dvolvol', 'dV_dalpha', 'dV_dF', 'dV_dK']

    # Dict to store scaling parameters (for Step 9 inference)
    scaling_params = {
        'type': 'z_score',
        'features': {},
        'targets': {}
    }

    df_scaled = df.copy()

    # 3. Standardize the features
    print("Standardizing features (mean = 0, std = 1)...")
    for col in feature_cols:
        mean_val = float(df[col].mean())
        std_val = float(df[col].std())
        
        # Safeguard: avoid division by zero if a feature is constant
        if std_val < 1e-8:
            std_val = 1.0 
            
        df_scaled[col] = (df[col] - mean_val) / std_val
        scaling_params['features'][col] = {'mean': mean_val, 'std': std_val}

    # 4. Standardize the targets (prices and derivatives)
    print("Standardizing targets (prices and Greeks)...")
    for col in target_cols:
        mean_val = float(df[col].mean())
        std_val = float(df[col].std())
        
        if std_val < 1e-8:
            std_val = 1.0
            
        # Create new '_scaled' columns so Step 8 works without changes
        scaled_col_name = f"{col}_scaled"
        df_scaled[scaled_col_name] = (df[col] - mean_val) / std_val
        scaling_params['targets'][col] = {'mean': mean_val, 'std': std_val}

    # 5. Save the scaled dataset
    output_csv = data_dir / "sabr_hybrid_mesh_scaled.csv"
    df_scaled.to_csv(output_csv, index=False)
    print(f"Standardized dataset saved to: {output_csv.name}")

    # 6. Save the scaler parameters
    output_json = data_dir / "scaling_params_zscore.json"
    with open(output_json, 'w') as f:
        json.dump(scaling_params, f, indent=4)
    print(f"Scaling parameters saved to: {output_json.name}")

if __name__ == "__main__":
    standardize_hybrid_dataset()