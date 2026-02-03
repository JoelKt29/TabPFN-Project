import numpy as np
import pandas as pd
from step2_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm 

# --- 1. DÉFINITION DES CONSTANTES ET DE LA TAILLE OPTIMISÉE ---
np.random.seed(42)

T = 1.0
SHIFT = 0.0
NUM_STRIKES = 8     
MAX_CONFIGS = 625   
MAX_SAMPLES = 5000  

print(f"Génération de {MAX_SAMPLES} échantillons avec Grille Structurée (8 strikes)...")

# --- 2. DÉFINITION DE LA GRILLE DE PARAMÈTRES (Linspace) ---
# Selon Peter : beta .25 .99, rho -.25 .25, volvol .15 .25, atm vol .005 .02, f .01 .5
NUM_POINTS = 6

BETAS = np.linspace(0.25, 0.99, NUM_POINTS) 
RHOS = np.linspace(-0.25, 0.25, NUM_POINTS) 
VOLVOLS = np.linspace(0.15, 0.25, NUM_POINTS) 
ATM_VOLS = np.linspace(0.005, 0.02, NUM_POINTS) 
FORWARDS = np.linspace(0.01, 0.50, NUM_POINTS) 

# --- 3. GÉNÉRATION DE LA GRILLE ET CALCUL SABR ---
data_list = []
count = 0

for beta in tqdm(BETAS, desc="Génération SABR"):
    for rho in RHOS:
        for volvol in VOLVOLS:
            for v_atm_n in ATM_VOLS:
                for f in FORWARDS:
                    
                    if count >= MAX_SAMPLES:
                        break
                    
                    try:
                        sabr = Hagan2002LognormalSABR(
                            f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n,
                            beta=beta, rho=rho, volvol=volvol
                        )
                        alpha = sabr.alpha()

                        strikes = np.linspace(0.75 * f, 1.5 * f, NUM_STRIKES)  

                        for k in strikes:
                            if count >= MAX_SAMPLES:
                                break

                            # CIBLE Y : Volatilité Normale
                            v_normal = sabr.normal_vol(k)
                            log_moneyness = np.log(k / f)

                            data_list.append({
                                'beta': beta,
                                'rho': rho,
                                'volvol': volvol,
                                'v_atm_n': v_atm_n,
                                'alpha': alpha,             
                                'F': f,
                                'K': k,
                                'log_moneyness': log_moneyness,
                                'T': T, 
                                'Shift': SHIFT,
                                'volatility_output': v_normal
                            })
                            count += 1

                    except Exception:
                        continue

df_sabr = pd.DataFrame(data_list)
print(f"\nDataFrame SABR généré et limité à {len(df_sabr)} échantillons valides.")

# --- 4. NORMALISATION (SCALING) ET ENREGISTREMENT ---
X_raw = df_sabr.drop(columns=['volatility_output'])
y_raw = df_sabr['volatility_output']

# Identification des colonnes variables (exclut les constantes comme T, Shift)
non_constant_cols = X_raw.columns[X_raw.nunique() > 1].tolist()
X_variable = X_raw[non_constant_cols]
X_constant = X_raw.drop(columns=non_constant_cols)

# Scaling des Inputs Variables : [-1, 1]
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X_variable)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_variable.columns, index=X_raw.index)

# Scaling de l'Output (Y) : [0, 1]
y_min = y_raw.min()
y_max = y_raw.max()
y_scaled = (y_raw - y_min) / (y_max - y_min)

# Assemblage final
df_final = pd.concat([X_scaled_df, X_constant], axis=1)
df_final['y_scaled'] = y_scaled

# --- Sauvegarde des paramètres de scaling ---
scaling_params = {
    'y_min': float(y_min),
    'y_max': float(y_max),
    'X_min': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_min_)},
    'X_max': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_max_)}
}

with open('scaling_params_recovery.json', 'w') as f:
    json.dump(scaling_params, f, indent=4)

df_final.to_csv('sabr_data_recovery.csv', index=False)

print("\n✅ Jeu de données de récupération généré avec scaling stable.")
print("✅ Strikes range corrigée : 0.75f à 1.5f (selon Peter)")
print("Utilisez 'sabr_data_recovery.csv' pour le test TabPFN.")
