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

class FinancialSCM(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = 9 # 8 SABR features + 1 TabPFN prediction
        h_dims = config.get('hidden_dims')
        dropout = config.get('dropout')
        layers = []
        prev_dim = input_dim
        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(), # Smooth activation for better gradients
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        self.head = nn.Sequential(*layers, nn.Linear(prev_dim, 7))

    def forward(self, x_sabr, x_tabpfn):
        combined = torch.cat([x_sabr, x_tabpfn], dim=1)
        return self.head(combined)

def run_step9():
    with open(config_path, "r") as f:
        config = json.load(f)
    df = pd.read_csv(data_dir / "sabr_with_derivatives_scaled.csv")
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    X_raw = df[feature_cols].values
    Y_raw = df[target_cols].values

    tabpfn = TabPFNRegressor(device=device)
    train_idx = np.random.choice(len(df), 500, replace=False)
    tabpfn.fit(X_raw[train_idx], Y_raw[train_idx, 0])
    tabpfn_preds = tabpfn.predict(X_raw).reshape(-1, 1)
    X_s = torch.FloatTensor(X_raw)
    X_t = torch.FloatTensor(tabpfn_preds)
    Y = torch.FloatTensor(Y_raw)

    # Initialize SCM
    model = FinancialSCM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 0.001))
    criterion = DerivativeLoss(value_weight=config.get('value_weight'), 
                derivative_weight=config.get('derivative_weight'))
    model.train()
    
    for epoch in range(101):
        indices = torch.randperm(X_s.size(0))[:128]
        bx_s, bx_t, by = X_s[indices].to(device), X_t[indices].to(device), Y[indices].to(device)
        optimizer.zero_grad()
        out = model(bx_s, bx_t)
        pred_val = out[:, 0:1]
        true_val = by[:, 0:1]
        
        # Mapping predicted Greeks to targets
        pred_greeks = {f'd{j}': out[:, j:j+1] for j in range(1, 7)}
        true_greeks = {f'd{j}': by[:, j:j+1] for j in range(1, 7)}
        loss, _ = criterion(pred_val, true_val, pred_greeks, true_greeks)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            mae = torch.mean(torch.abs(pred_val - true_val)).item()
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.6f} | Vol MAE: {mae:.6f}")


    save_path = current_dir / "tabpfn_sabr_step9_causal_final.pth"
    torch.save(model.state_dict(), save_path)
if __name__ == "__main__":
    run_step9()