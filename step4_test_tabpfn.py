import numpy as np
import pandas as pd
import json
import torch
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_FILE = 'sabr_market_data.csv'
SCALING_PARAMS_FILE = 'scaling_params_recovery.json'

# 1. Chargement des données et des paramètres de scaling
df = pd.read_csv(DATA_FILE)
with open(SCALING_PARAMS_FILE, 'r') as f:
    scaling_params = json.load(f)

# 2. Préparation des données
# On retire la cible pour isoler les features X
X = df.drop(columns=['SABR_volatility'])
y = df['SABR_volatility'].values.flatten()

# Split Train/Test (70% entraînement, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Initialisation du modèle avec détection automatique du GPU
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🖥️ Processeur utilisé : {device}")

# On augmente N_ensemble_configurations pour plus de précision (max 32 préconisé)
regressor = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
# 4. Entraînement et Prédiction
print("⏳ TabPFN analyse les données SABR...")
regressor.fit(X_train, y_train)
predictions_scaled = regressor.predict(X_test)
# 5. Descaling (Retour aux vraies valeurs de volatilité)
y_min = scaling_params['y_min']
y_max = scaling_params['y_max']

predictions_real = predictions_scaled * (y_max - y_min) + y_min
y_test_real = y_test * (y_max - y_min) + y_min

# 6. Calcul des erreurs
mae = mean_absolute_error(y_test_real, predictions_real)

print(f"MAE  : {mae:.2e}")

# 7. Visualisation des résultats
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, predictions_real, alpha=0.4, color='blue', s=10, label='Prédictions')
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', label='Parfaite identité')
plt.xlabel("Vraie Volatilité SABR")
plt.ylabel("Prédiction TabPFN")
plt.title(f"Validation TabPFN - MAE: {mae:.2e}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()