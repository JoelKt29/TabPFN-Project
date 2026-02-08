import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from pathlib import Path
from tabpfn import TabPFNRegressor
from step6_loss_with_derivatives import DerivativeLoss

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"

# --- 2. CAUSAL MODEL ARCHITECTURE (SCM) ---
class FinancialSCM(nn.Module):
    """
    Structural Causal Model (SCM) Stacking:
    Combines 8 SABR features + 1 TabPFN prior to predict 
    the Volatility and its 6 Causal Derivatives (Greeks).
    """
    def __init__(self, config):
        super().__init__()
        input_dim = 9 # 8 SABR features + 1 TabPFN prediction
        h_dims = config.get('hidden_dims', [512, 256, 128])
        dropout = config.get('dropout', 0.05)
        
        layers = []
        prev_dim = input_dim
        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(), # Smooth activation for better gradients
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        # Output: 1 Volatility + 6 Causal Gradients (Greeks)
        self.head = nn.Sequential(*layers, nn.Linear(prev_dim, 7))

    def forward(self, x_sabr, x_tabpfn):
        # Concatenate raw features with TabPFN "expert" prior
        combined = torch.cat([x_sabr, x_tabpfn], dim=1)
        return self.head(combined)

# --- 3. TRAINING ENGINE ---
def run_step9():
    # Load optimal hyperparameters from Step 7
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"📂 Loaded optimal config from Step 7 (Seed {SEED}).")
    else:
        config = {"hidden_dims": [512, 256, 128], "lr": 0.0016, "dropout": 0.09}

    # Load Dataset
    df = pd.read_csv(data_dir / "sabr_with_derivatives_scaled.csv")
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    
    X_raw = df[feature_cols].values
    Y_raw = df[target_cols].values

    # Generate TabPFN Structural Priors
    print("🔄 TabPFN extracting structural causal patterns...")
    tabpfn = TabPFNRegressor(device='cpu')
    # Context sample for In-Context Learning (consistent with SEED)
    train_idx = np.random.choice(len(df), 500, replace=False)
    tabpfn.fit(X_raw[train_idx], Y_raw[train_idx, 0])
    
    # Get TabPFN's view of the world
    tabpfn_preds = tabpfn.predict(X_raw).reshape(-1, 1)

    # Convert to Tensors
    X_s = torch.FloatTensor(X_raw)
    X_t = torch.FloatTensor(tabpfn_preds)
    Y = torch.FloatTensor(Y_raw)

    # Initialize SCM
    model = FinancialSCM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 0.001))
    
    # CAUSAL SOBOLEV LOSS: 
    # value_weight = Accuracy on price
    # derivative_weight = Consistency with the causal graph (Greeks)
    criterion = DerivativeLoss(value_weight=0.6, derivative_weight=0.15)

    print("\n🚀 TRAINING SCM: Learning SABR Causal Laws...")
    model.train()
    
    for epoch in range(101):
        # Batching
        indices = torch.randperm(X_s.size(0))[:128]
        bx_s, bx_t, by = X_s[indices].to(device), X_t[indices].to(device), Y[indices].to(device)

        optimizer.zero_grad()
        out = model(bx_s, bx_t)
        
        # Structure the output: index 0 is Value, 1 to 6 are Causal Gradients
        pred_val = out[:, 0:1]
        true_val = by[:, 0:1]
        
        # Mapping predicted Greeks to targets
        pred_greeks = {f'd{j}': out[:, j:j+1] for j in range(1, 7)}
        true_greeks = {f'd{j}': by[:, j:j+1] for j in range(1, 7)}

        # Apply Sobolev/Causal Loss
        loss, _ = criterion(pred_val, true_val, pred_greeks, true_greeks)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            mae = torch.mean(torch.abs(pred_val - true_val)).item()
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.6f} | Vol MAE: {mae:.6f}")

    # Save the Final Causal Brain
    save_path = current_dir / "tabpfn_sabr_step9_causal_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ SCM Model (Causal Stacking) saved to: {save_path}")

if __name__ == "__main__":
    run_step9()