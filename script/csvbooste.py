import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler

from step02_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR
from step05_jax_sabr import compute_sabr_with_jax


# ==============================
# PATH
# ==============================
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)


# ==============================
# BOUNDS
# ==============================
L_BOUNDS = [0.01, 0.25, -0.50, 0.15, 0.005]
U_BOUNDS = [0.06, 0.99,  0.50, 0.45, 0.030]


# ==============================
# ADAPTIVE STRIKES
# ==============================
def generate_adaptive_strikes(f, T, alpha, beta, rho, volvol, base_n=8):

    coarse_K = np.linspace(0.75 * f, 1.5 * f, base_n)

    vols_c, grads_c = compute_sabr_with_jax(
    np.atleast_1d(coarse_K), f, T, alpha, beta, rho, volvol  # ← add atleast_1d
)

    score = np.abs(grads_c['dV_dK']) + np.abs(grads_c['dV_dvolvol'])

    n_refine = max(2, base_n // 3)
    top_idx = np.argsort(score)[-n_refine:]

    refined_K = []

    for idx in top_idx:
        k0 = coarse_K[idx]
        refined_K.append(np.linspace(0.97 * k0, 1.03 * k0, 5))

    refined_K = np.concatenate(refined_K)

    strikes = np.unique(np.concatenate([coarse_K, refined_K]))

    return strikes


# ==============================
# MAIN
# ==============================
def generate_data_adaptive(num_samples=3000, base_strikes=8):

    print(f"🚀 Génération Sobol + Adaptive Mesh")

    target = int(np.ceil(num_samples / base_strikes))
    m = int(np.ceil(np.log2(target)))
    n_configs = 2**m

    sampler = qmc.Sobol(d=5, scramble=True)
    params = qmc.scale(sampler.random_base2(m), L_BOUNDS, U_BOUNDS)

    data = []

    T = 1.0
    SHIFT = 0.0

    for i in tqdm(range(n_configs)):

        f, beta, rho, volvol, v_atm_n = params[i]

        # alpha calibration
        try:
            model = Hagan2002LognormalSABR(
                f=f, shift=SHIFT, t=T,
                v_atm_n=v_atm_n,
                beta=beta, rho=rho, volvol=volvol
            )
            alpha = model.alpha()
        except:
            continue

        # adaptive strikes
        strikes = generate_adaptive_strikes(
            f, T, alpha, beta, rho, volvol, base_strikes
        )
        n = len(strikes)

        strikes = np.atleast_1d(strikes)

        

        # JAX
        vols, grads = compute_sabr_with_jax(
            strikes, f, T, alpha, beta, rho, volvol
        )

        # chain rule
        d_alpha_d_vatm = f**(1 - beta)
        dv_dvatm = np.full(n, float(grads['dV_dalpha']) * d_alpha_d_vatm)


        dV_dbeta   = np.full(n, float(grads['dV_dbeta']))
        dV_drho    = np.full(n, float(grads['dV_drho']))
        dV_dvolvol = np.full(n, float(grads['dV_dvolvol']))
        dV_dF      = np.full(n, float(grads['dV_dF']))

        for j, k in enumerate(strikes):
            if np.isnan(vols[j]) or vols[j] < 0:
                continue
            
            data.append({
                'beta': beta,
                'rho': rho,
                'volvol': volvol,
                'v_atm_n': v_atm_n,
                'alpha': alpha,
                'F': f,
                'K': k,
                'log_moneyness': np.log(k/f),
                'T': T,
                'Shift': SHIFT,

                'volatility': float(vols[j]),

                 'dV_dbeta':   dV_dbeta[j],
        'dV_drho':    dV_drho[j],
        'dV_dvolvol': dV_dvolvol[j],
        'dV_dF':      dV_dF[j],
        'dV_dK':      float(grads['dV_dK'][j]),   # shape (n,) → ok
        'dV_dvatm':   dv_dvatm[j]
    })

    return pd.DataFrame(data)


# ==============================
# SCALE + SAVE
# ==============================
def scale_and_save(df):

    print("📊 Scaling...")

    feature_cols = ['beta','rho','volvol','v_atm_n','alpha','F','K','log_moneyness']
    deriv_cols = ['dV_dbeta','dV_drho','dV_dvolvol','dV_dvatm','dV_dF','dV_dK']

    scaler = MinMaxScaler((-1,1))
    X_scaled = scaler.fit_transform(df[feature_cols])

    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    df_scaled['volatility_scaled'] = (
        (df['volatility'] - df['volatility'].min()) /
        (df['volatility'].max() - df['volatility'].min())
    )

    for col in deriv_cols:
        s = MinMaxScaler((-1,1))
        df_scaled[col+'_scaled'] = s.fit_transform(df[[col]])

    df_scaled['T'] = df['T']
    df_scaled['Shift'] = df['Shift']

    path = data_dir / "sabr_adaptive_dataset.csv"
    df_scaled.to_csv(path, index=False)

    print(f"✅ Saved → {path}")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":

    df = generate_data_adaptive(3000, 8)

    if not df.empty:
        scale_and_save(df)
    else:
        print("❌ No data generated")