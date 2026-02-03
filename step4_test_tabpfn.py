import numpy as np
import pandas as pd
import json
import torch
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- 1. CHARGEMENT DES DONNÉES ET DES PARAMÈTRES ---
DATA_FILE = 'sabr_data_recovery.csv'
SCALING_PARAMS_FILE = 'scaling_params_recovery.json'

try:
    df = pd.read_csv(DATA_FILE)
    with open(SCALING_PARAMS_FILE, 'r') as f:
        scaling_params = json.load(f)
    print(f" Données chargées : {len(df)} échantillons.")
except FileNotFoundError:
    print(" ERREUR : Fichier de données ou de scaling introuvable.")
    exit()

# Extraction des features (X) et de la cible (y)
X_full_scaled = df.drop(columns=['y_scaled'])
y_full_scaled = df['y_scaled'].values.flatten()

# --- 2. SPLIT TRAIN / TEST ---
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_full_scaled, y_full_scaled, test_size=0.3, random_state=42
)

# --- 3. CHARGEMENT DES PARAMÈTRES DE SCALING ---
y_min = scaling_params['y_min']
y_max = scaling_params['y_max']

# --- 4. TABPFN REGRESSION ---
print("\n Démarrage de l'entraînement TabPFN...")

try:
    regressor = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)
    
    regressor.fit(X_train_s, y_train_s)
    print(" Entraînement TabPFN terminé avec succès.")

    # Prédiction sur le jeu de test
    predictions_s = regressor.predict(X_test_s)

    # --- 5. DÉSCALING des prédictions ---
    predictions_real = predictions_s * (y_max - y_min) + y_min
    y_test_real = y_test_s * (y_max - y_min) + y_min

    # --- 6. CALCUL DES MÉTRIQUES ---
    mae_real = mean_absolute_error(y_test_real, predictions_real)

    print(f" MAE  : {mae_real:.6f} (Target = 0.0001)")

    comp_df = pd.DataFrame({
        'SABR ': y_test_real[:5],
        'TabPFN': predictions_real[:5]
    })
    print("\nComparision of the volatility SABR  vs TabPFN for the first five predictions :")
    print(comp_df)

except Exception as e:
    print(f"Détail : {e}")
