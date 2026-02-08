import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tabpfn import TabPFNRegressor
from step8_transfer_learning import TabPFNStackingModel 

# --- 1. SETUP & REPRODUCIBILITY ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path management
current_dir = Path(__file__).resolve().parent
csv_path = current_dir.parent / "data" / "sabr_with_derivatives_scaled.csv"
model_path = current_dir / "tabpfn_sabr_step8_stacking.pth"
graph_dir = current_dir.parent / "graph"
graph_dir.mkdir(parents=True, exist_ok=True)

# --- 2. DATA LOAD ---
df = pd.read_csv(csv_path)
feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
sample_data = df.iloc[100].copy()

# Generate strike range for testing
strikes = np.linspace(df['K'].min(), df['K'].max(), 200)
test_batch = np.array([
    np.concatenate([sample_data[feature_cols[:6]].values, [k, sample_data['log_moneyness']]]) 
    for k in strikes
])

# --- 3. PREDICTIONS ---
# Step 4 Baseline
tabpfn = TabPFNRegressor(device='cpu')
train_idx = np.random.choice(len(df), 500, replace=False)
tabpfn.fit(df.iloc[train_idx][feature_cols].values, df.iloc[train_idx]['volatility_scaled'].values)
preds4 = tabpfn.predict(test_batch)

# Step 8 Sobolev
config = {"hidden_dims": [512, 256, 128], "dropout": 0.09} 
model = TabPFNStackingModel(config).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    x_sabr = torch.FloatTensor(test_batch).to(device)
    x_tab = torch.FloatTensor(preds4).reshape(-1, 1).to(device)
    preds8 = model(x_sabr, x_tab).cpu().numpy()[:, 0]

# --- 4. GRADIENT & VOG METRICS ---
dk = np.diff(strikes)
slope4 = np.diff(preds4) / dk
slope8 = np.diff(preds8) / dk

# Volatility of Gradient (VoG)
vog4 = np.std(np.diff(slope4))
vog8 = np.std(np.diff(slope8))

print(f"VoG Step 4: {vog4:.6f}")
print(f"VoG Step 8: {vog8:.6f}")
print(f"Stability Gain: {((vog4-vog8)/vog4)*100:.1f}%")

# --- 5. PLOTTING ---
# Use plt.subplots to keep track of the figure object
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

ax1.plot(strikes, preds4, 'r--', label='Step 4 (Baseline)', alpha=0.6)
ax1.plot(strikes, preds8, 'b-', label='Step 8 (Sobolev)', linewidth=2)
ax1.set_title(f"SABR Smile Comparison (Seed {SEED})")
ax1.set_ylabel("Volatility")
ax1.legend()
ax1.grid(True, alpha=0.2)

ax2.plot(strikes[:-1], slope4, 'r--', alpha=0.5, label=f'Step 4 Grad (VoG: {vog4:.4f})')
ax2.plot(strikes[:-1], slope8, 'b-', linewidth=2, label=f'Step 8 Grad (VoG: {vog8:.4f})')
ax2.set_title("Gradient Stability Analysis (dV/dK)")
ax2.set_xlabel("Strike (K)")
ax2.set_ylabel("Slope")
ax2.legend()
ax2.grid(True, alpha=0.2)

# --- 6. SAVE AND SHOW ---
plt.tight_layout()

# Absolute path for saving to avoid directory issues
save_name = "step4_vs_step8_stability.png"
save_path = graph_dir / save_name

# Clear any existing plots before saving to avoid Tkinter loops
plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
print(f"Successfully saved to: {save_path}")

plt.show()