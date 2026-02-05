import numpy as np
import pandas as pd
from step2_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm 
from pathlib import Path
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"

np.random.seed(42)

T = 1.0
SHIFT = 0.0
MAX_SAMPLES = 5000 
data_list = []
count = 0
pbar = tqdm(total=MAX_SAMPLES)

while count < MAX_SAMPLES:
    # Échantillonnage aléatoire dans les plages spécifiées par Peter
    beta = np.random.uniform(0.25, 0.99)
    rho = np.random.uniform(-0.25, 0.25)
    volvol = np.random.uniform(0.15, 0.25)
    v_atm_n = np.random.uniform(0.005, 0.02)
    f = np.random.uniform(0.01, 0.50)
    k = f * np.random.uniform(0.75, 1.5)

    sabr = Hagan2002LognormalSABR(
        f=f, shift=SHIFT, t=T, v_atm_n=v_atm_n,
        beta=beta, rho=rho, volvol=volvol)
    alpha = sabr.alpha()
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



# Scaling preparation 

X_raw = df_sabr.drop(columns=['volatility_output'])
y_raw = df_sabr['volatility_output']

#Cant scale constant variable
non_constant_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
X_variable = X_raw[non_constant_cols]
X_constant = X_raw.drop(columns=non_constant_cols)

# Scaling data between -1 and 1
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X_variable)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_variable.columns, index=X_raw.index)

# Scaling volatility between 0 and 1
y_min = y_raw.min()
y_max = y_raw.max()
y_scaled = (y_raw - y_min) / (y_max - y_min)

#Re adding constant variables to the rest of the variables
df_final = pd.concat([X_scaled_df, X_constant], axis=1)
df_final['SABR_volatility'] = y_scaled


# Saving scaling parameters for future descalingn a
scaling_params = {
    'y_min': float(y_min),
    'y_max': float(y_max),
    'X_min': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_min_)},
    'X_max': {col: float(v) for col, v in zip(X_variable.columns, scaler_X.data_max_)}}
with open(data_dir / "scaling_params_recovery.json", 'w') as f:
    json.dump(scaling_params, f, indent=4)

# Saving data
df_final.to_csv(data_dir / "sabr_market_data.csv", index=False)

print("\nGenerating OK")
