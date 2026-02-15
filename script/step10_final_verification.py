import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from tabpfn import TabPFNRegressor

# Ensure we can import from the current directory
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from step08_transfer_learning import TabPFNStackingModel

# --- Configuration Paths ---
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"
model_path = current_dir / "tabpfn_step9_causal_final.pth"
csv_path = data_dir / "sabr_with_derivatives_scaled.csv"

def verify_model():
    # 1. Load Configuration and Model
    print("Loading model and configuration...")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the hybrid model architecture
    model = TabPFNStackingModel(config).to(device)
    
    # Load the trained weights from Step 9
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare TabPFN (The Prior)
    # We set n_estimators=4 to make the test fast (inference only)
    # In production, you might want to increase this for slightly better accuracy
    print("Initializing TabPFN...")
    tabpfn = TabPFNRegressor(device=device, n_estimators=4, ignore_pretraining_limits=True)
    
    # 3. Load Data for TabPFN Context
    # The TabPFN needs a small set of examples to understand the current task
    print("Loading data...")
    df = pd.read_csv(csv_path)
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    
    # We train the TabPFN on a random subset (2000 samples) to give it context
    train_idx = np.random.choice(len(df), 2000, replace=False)
    X_train = df.iloc[train_idx][feature_cols].values
    y_train = df.iloc[train_idx]['volatility_scaled'].values
    tabpfn.fit(X_train, y_train)

    # 4. Select a Single Scenario to Plot the Curve (The "Smile")
    # We pick one random market configuration
    sample = df.sample(1).iloc[0]
    
    # We generate a range of Strikes (K) around the Forward price to visualize the smile
    # This simulates "sweeping" the strike price while keeping other SABR params constant
    K_range = np.linspace(sample['K'] * 0.5, sample['K'] * 1.5, 100)
    
    # Prepare the test batch
    X_test_list = []
    for k_val in K_range:
        row = sample[feature_cols].copy()
        row['K'] = k_val  # We vary K
        # Recalculate log_moneyness (simplified for this visualization)
        row['log_moneyness'] = np.log(k_val / sample['F']) 
        X_test_list.append(row.values)
    
    X_test = np.array(X_test_list)
    
    # 5. Generate Predictions
    print("Generating predictions...")
    
    # A. TabPFN Raw Predictions (The "Noisy" Baseline)
    pfn_preds = tabpfn.predict(X_test).reshape(-1, 1)
    
    # B. Hybrid Model Predictions (The "Smooth" Final Result)
    # Convert numpy arrays to PyTorch tensors
    X_sabr_torch = torch.FloatTensor(X_test).to(device)
    X_pfn_torch = torch.FloatTensor(pfn_preds).to(device)
    
    with torch.no_grad():
        hybrid_preds = model(X_sabr_torch, X_pfn_torch)
        
        # Based on Step 8/9 architecture:
        # Column 0 is the predicted Volatility
        # Column 6 is the predicted Derivative dV/dK (Greek)
        vol_hybrid = hybrid_preds[:, 0].cpu().numpy()
        deriv_hybrid = hybrid_preds[:, 6].cpu().numpy() 

    # 6. Compute Numerical Derivatives (Slopes) for Comparison
    # We differentiate the volatility curve numerically to see how "bumpy" it is.
    # Note: simple gradient for visualization purposes.
    grad_pfn = np.gradient(pfn_preds.flatten())
    grad_hybrid_vol = np.gradient(vol_hybrid) 

    # 7. Visualization
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Volatility Smile
    # Shows that both models predict roughly the same values (height)
    ax1.plot(K_range, pfn_preds, color='red', alpha=0.5, label='TabPFN Raw (Baseline)', linestyle='--')
    ax1.plot(K_range, vol_hybrid, color='lime', lw=2, label='Hybrid Model (Final)')
    ax1.set_title("Volatility Smile Reconstruction")
    ax1.set_xlabel("Strike (K)")
    ax1.set_ylabel("Volatility")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Plot 2: The Derivative (The Crucial Test)
    # Shows the smoothness of the slope. 
    # We normalize values for display because scaling factors differ.
    def norm(x): 
        return (x - np.mean(x)) / (np.std(x) + 1e-6)
    
    ax2.plot(K_range, norm(grad_pfn), color='red', alpha=0.5, label='TabPFN Gradient (Noisy)')
    ax2.plot(K_range, norm(deriv_hybrid), color='lime', lw=2, label='Hybrid Gradient (Learned)')
    ax2.set_title("Derivative Comparison (Greeks Stability)")
    ax2.set_xlabel("Strike (K)")
    ax2.set_ylabel("Normalized dV/dK")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Save and Show
    graph_dir = current_dir.parent / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    save_path = graph_dir / "step10_final_proof.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Proof graph saved at: {save_path}")
    plt.show()

if __name__ == "__main__":
    verify_model()