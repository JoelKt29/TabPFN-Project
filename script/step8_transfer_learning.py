from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from tabpfn import TabPFNRegressor
from step6_loss_with_derivatives import DerivativeLoss

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


class TabPFNStackingModel(nn.Module):
    def __init__(self, config, n_outputs=7):
        super().__init__()
        input_dim = 9  # 8 SABR features + 1 TabPFN raw predictio
        layers = []
        prev_dim = input_dim
        h_dims = config.get('hidden_dims', [512, 256, 128])
        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),
                nn.Dropout(config.get('dropout', 0.09))
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, n_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, x_sabr, x_tabpfn):
        combined = torch.cat([x_sabr, x_tabpfn], dim=1)
        return self.head(combined)


def run_step8():
    with open(config_path, "r") as f:
        config = json.load(f)
    csv_path = data_dir / "sabr_with_derivatives_scaled.csv"
    df = pd.read_csv(csv_path)    
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = ['volatility_scaled', 'dV_dbeta_scaled', 'dV_drho_scaled', 
                   'dV_dvolvol_scaled', 'dV_dvatm_scaled', 'dV_dF_scaled', 'dV_dK_scaled']
    X_raw = df[feature_cols].values
    y_raw = df[target_cols].values

    tabpfn = TabPFNRegressor(device=device)
    
    train_idx = np.random.choice(len(df), 500, replace=False)
    tabpfn.fit(X_raw[train_idx], y_raw[train_idx, 0])
    tabpfn_preds = tabpfn.predict(X_raw).reshape(-1, 1)

    X_sabr = torch.FloatTensor(X_raw)
    X_tabpfn = torch.FloatTensor(tabpfn_preds)
    Y = torch.FloatTensor(y_raw)

    model = TabPFNStackingModel(config).to(device)
    criterion = DerivativeLoss(value_weight=config.get('value_weight'), 
                derivative_weight=config.get('derivative_weight'))
    opt_type = config.get('optimizer', 'adamw')
    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr'))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr'), weight_decay=config.get('weight_decay'))
    model.train()
    batch_size = config.get('batch_size')
    

    for epoch in range(101):
        permutation = torch.randperm(X_sabr.size(0))
        epoch_loss = 0
        for i in range(0, X_sabr.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            bx_s, bx_t = X_sabr[indices].to(device), X_tabpfn[indices].to(device)
            by = Y[indices].to(device)

            optimizer.zero_grad()
            out = model(bx_s, bx_t)
            
            # d1 to d6 correspond to the 6 derivatives 
            pred_d = {f'd{j}': out[:, j:j+1] for j in range(1, 7)}
            true_d = {f'd{j}': by[:, j:j+1] for j in range(1, 7)}
            
            loss, _ = criterion(out[:, 0:1], by[:, 0:1], pred_d, true_d)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            current_mae = torch.mean(torch.abs(out[:, 0:1] - by[:, 0:1])).item()
            avg_loss = epoch_loss / (len(X_sabr) / batch_size)
            print(f"Epoch {epoch:03d} | Sobolev Loss: {avg_loss:.6f} | Vol MAE: {current_mae:.6f}")



    save_path = current_dir / "tabpfn_sabr_step8_stacking.pth"
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    run_step8()