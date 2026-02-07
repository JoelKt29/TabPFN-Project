import numpy as np
import pandas as pd
import json
import torch
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path


current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
DATA_FILE = data_dir / "sabr_market_data.csv"
SCALING_PARAMS_FILE = data_dir / "scaling_params_recovery.json"
df = pd.read_csv(DATA_FILE)
with open(SCALING_PARAMS_FILE, 'r') as f:
    scaling_params = json.load(f)


# Regression
X = df.drop(columns=['SABR_volatility'])
y = df['SABR_volatility'].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
regressor = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
regressor.fit(X_train, y_train)
predictions_scaled = regressor.predict(X_test)


# Descaling 
y_min = scaling_params['y_min']
y_max = scaling_params['y_max']
predictions_real = predictions_scaled * (y_max - y_min) + y_min
y_test_real = y_test * (y_max - y_min) + y_min


# MAE Computing
mae = mean_absolute_error(y_test_real, predictions_real)
print(f"MAE  : {mae:.2e}")


# 7. Vizualisation
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, predictions_real, alpha=0.4, color='blue', s=10, label='Predictions')
plt.xlabel("SABR Volatility")
plt.ylabel(" TabPFN Prediction")
plt.title(f" TabPFN - MAE: {mae:.2e}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


graph_dir = current_dir.parent / "graph"
save_path = graph_dir / "step4_performance_scatter.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
