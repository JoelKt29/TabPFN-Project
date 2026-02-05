import numpy as np
import pandas as pd
from step2_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm 

# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Configuration
T = 1.0
SHIFT = 0.0
MAX_SAMPLES = 5000 

data_list = []
count = 0


# Utilisation d'une barre de progression
pbar = tqdm(total=MAX_SAMPLES)

while count < MAX_SAMPLES:
    try:
        # Échantillonnage aléatoire dans les plages spécifiées par Peter
        beta = np.random.uniform(0.25, 0.99)
        rho = np.random.uniform(-0.25, 0.25)
        volvol = np.random.uniform(0.15, 0.25)
        v_atm_n = np.random.uniform(0.005, 0.02)
        f = np.random.uniform(0.01, 0.50)
        
        # Initialisation du modèle SABR
        sabr = Hagan2002LognormalSABR(
            f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n,
            beta=beta, rho=rho, volvol=volvol
        )
        
        alpha = sabr.alpha()
        
        # Tirage d'un strike aléatoire entre 0.75*f et 1.5*f (recommandation Peter)
        k = f * np.random.uniform(0.75, 1.5)

        # Calcul de la volatilité normale
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
        pbar.update(1)


pbar.close()
df_sabr = pd.DataFrame(data_list)

# --- PRÉPARATION DES DONNÉES (SCALING) ---

X_raw = df_sabr.drop(columns=['volatility_output'])
y_raw = df_sabr['volatility_output']

# Identifier les colonnes variables (exclure T et Shift qui sont constants)
non_constant_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
X_variable = X_raw[non_constant_cols]
X_constant = X_raw.drop(columns=non_constant_cols)

# Scaling des entrées entre -1 et 1
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X_variable)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_variable.columns, index=X_raw.index)

# Scaling de la sortie (volatilité) entre 0 et 1
y_min = y_raw.min()
y_max = y_raw.max()
y_scaled = (y_raw - y_min) / (y_max - y_min)

# Assemblage final
df_final = pd.concat([X_scaled_df, X_constant], axis=1)
df_final['y_scaled'] = y_scaled

# --- SAUVEGARDE ---

# Paramètres de scaling pour le descaling dans la step4
scaling_params = {
    'y_min': float(y_min),
    'y_max': float(y_max),
    'X_min': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_min_)},
    'X_max': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_max_)}
}

# Correction des noms de fichiers pour correspondre à la step4
with open('scaling_params_recovery.json', 'w') as f:
    json.dump(scaling_params, f, indent=4)

df_final.to_csv('sabr_market_data.csv', index=False)

print("\n✅ Terminé !")
print(f"Fichier généré : sabr_market_data.csv ({len(df_final)} lignes)")
print("Fichier de scaling : scaling_params_recovery.json")