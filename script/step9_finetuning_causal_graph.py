import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor 
from step06_loss_with_derivatives import DerivativeLoss
from tqdm import tqdm
from step08_transfer_learning import TabPFNStackingModel

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():

    with open(config_path, 'r') as f: 
        config = json.load(f)
    df = pd.read_csv(data_dir / 'sabr_with_derivatives_scaled.csv')
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    deriv_cols = [c for c in df.columns if c.startswith('dV_') and c.endswith('_scaled')]
    y_cols = ['volatility_scaled'] + deriv_cols
    X_r = df[feature_cols].values
    Y_r = df[y_cols].values
    num_outputs = len(y_cols)
    tabpfn = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', 
                             n_estimators=32, ignore_pretraining_limits=True)
    ctx_size = min(len(X_r), 10000)
    idx = np.random.choice(len(X_r), ctx_size, replace=False)
    tabpfn.fit(X_r[idx], Y_r[idx, 0])

    pfn_priors = []
    chunk_size = 1000 
    for i in tqdm(range(0, len(X_r), chunk_size), desc="Inference Prior"):
        chunk = X_r[i:i+chunk_size]
        pfn_priors.append(tabpfn.predict(chunk).reshape(-1, 1))
    X_pfn = np.vstack(pfn_priors)

    X_train_raw, X_val_raw, X_train_pfn, X_val_pfn, Y_train, Y_val = train_test_split(
        X_r, X_pfn, Y_r, test_size=0.15, random_state=42)

    train_ds = TensorDataset(torch.FloatTensor(X_train_raw), torch.FloatTensor(X_train_pfn), torch.FloatTensor(Y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val_raw), torch.FloatTensor(X_val_pfn), torch.FloatTensor(Y_val))
    
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size'), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size'), shuffle=False)

    model = TabPFNStackingModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr'), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    criterion = DerivativeLoss(
        value_weight=config.get('value_weight'), 
        derivative_weight=config.get('derivative_weight'))

    best_mae = float('inf')

    for epoch in range(100):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/100", leave=False)
        
        for b_raw, b_pfn, b_y in pbar:
            b_raw, b_pfn, b_y = b_raw.to(device), b_pfn.to(device), b_y.to(device)
            
            out = model(b_raw, b_pfn)
            
            pred_vol, true_vol = out[:, 0:1], b_y[:, 0:1]
            pred_der = {f'd{i}': out[:, i:i+1] for i in range(1, num_outputs)}
            true_der = {f'd{i}': b_y[:, i:i+1] for i in range(1, num_outputs)}
            
            loss, _ = criterion(pred_vol, true_vol, pred_der, true_der)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        model.eval()
        val_mae = 0
        val_loss = 0
        with torch.no_grad():
            for v_raw, v_pfn, v_y in val_loader:
                v_raw, v_pfn, v_y = v_raw.to(device), v_pfn.to(device), v_y.to(device)
                preds = model(v_raw, v_pfn)
                
                v_loss, _ = criterion(preds[:, 0:1], v_y[:, 0:1], 
                                     {f'd{i}': preds[:, i:i+1] for i in range(1, num_outputs)}, 
                                     {f'd{i}': v_y[:, i:i+1] for i in range(1, num_outputs)})
                val_loss += v_loss.item()
                val_mae += torch.mean(torch.abs(preds[:, 0] - v_y[:, 0])).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save(model.state_dict(), current_dir/"tabpfn_step9_causal_final.pth")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f" Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.5f} | Val MAE: {avg_val_mae:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"\nBest MAE: {best_mae:.6f}")

if __name__ == "__main__":
    train()