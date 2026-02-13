import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tabpfn import TabPFNRegressor
from step02_hagan_2002_lognormal_sabr import Hagan2002LognormalSABR
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def generate_3d_comparison_fast():
    print("--- 3D Risk Surface Generation (Vectorized & Fast) ---")
    
    # 1. Market Scenario Setup
    beta = 0.5
    rho = -0.3
    volvol = 0.4
    v_atm_n = 0.01
    T = 1.0
    shift = 0.0
    
    # Grid resolution (50x50 = 2500 points to plot)
    n_points = 50 
    
    print("1. Training TabPFN (Context Injection)...")
    # Generating context data (Training set)
    # We use a random distribution to simulate "historical" data
    np.random.seed(42)
    n_train = 2000
    train_F = np.random.uniform(0.02, 0.04, n_train)
    train_K = train_F * np.random.uniform(0.5, 1.5, n_train)
    
    train_X = []
    train_y = []
    
    # Fast generation of training data
    for i in range(n_train):
        model = Hagan2002LognormalSABR(f=train_F[i], shift=shift, t=T, v_atm_n=v_atm_n, 
                                       beta=beta, rho=rho, volvol=volvol)
        vol = model.normal_vol(train_K[i])
        if not np.isnan(vol) and vol > 0:
            # Features: F, K, LogMoneyness
            train_X.append([train_F[i], train_K[i], np.log(train_K[i]/train_F[i])])
            train_y.append(vol)
            
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    
    # Fit TabPFN (The "Brain")
    # CORRECTION HERE: added ignore_pretraining_limits=True to allow >1000 samples on CPU
    tabpfn = TabPFNRegressor(device='cpu', n_estimators=4, ignore_pretraining_limits=True) 
    tabpfn.fit(train_X, train_y)

    print("2. Batch Prediction (The Speedup)...")
    
    # Create the Meshgrid (The Map)
    f_range = np.linspace(0.02, 0.04, n_points)
    k_range = np.linspace(0.01, 0.06, n_points)
    F_grid, K_grid = np.meshgrid(f_range, k_range)
    
    # Flatten the grid to send it to TabPFN in one go
    # We turn the matrix into a long list of points
    flat_F = F_grid.ravel()
    flat_K = K_grid.ravel()
    flat_Moneyness = np.log(flat_K / flat_F)
    
    # Stack features into a (N_points, 3) matrix
    X_test_batch = np.column_stack((flat_F, flat_K, flat_Moneyness))
    
    # --- VECTORIZED INFERENCE ---
    # Instead of looping 2500 times, we call predict ONCE.
    flat_preds = tabpfn.predict(X_test_batch)
    
    # Reshape back to 3D surface format (50x50)
    Z_pfn_vol = flat_preds.reshape(n_points, n_points)

    print("3. Calculating Derivatives...")
    
    # A. Calculate True Gradient (Analytical/Numerical Reference)
    Z_true_grad = np.zeros_like(F_grid)
    eps = 1e-5
    
    # We still loop here for the "True" value calculation (Python math is fast enough)
    for i in range(n_points):
        for j in range(n_points):
            f_val = F_grid[i, j]
            k_val = K_grid[i, j]
            m = Hagan2002LognormalSABR(f=f_val, shift=shift, t=T, v_atm_n=v_atm_n, 
                                       beta=beta, rho=rho, volvol=volvol)
            v_p = m.normal_vol(k_val + eps)
            v_m = m.normal_vol(k_val - eps)
            Z_true_grad[i, j] = (v_p - v_m) / (2*eps)

    # B. Calculate TabPFN Gradient (Numerical from surface)
    # K varies along axis 0 (rows)
    # We calculate the step size dK for the gradient
    dk_step = (k_range[-1] - k_range[0]) / n_points
    grad_k_pfn = np.gradient(Z_pfn_vol, axis=0) / dk_step

    print("4. Rendering 3D Plot...")
    
    fig = go.Figure()

    # Layer 1: The True Model (Reference) - Green/Transparent
    fig.add_trace(go.Surface(
        z=Z_true_grad, x=F_grid, y=K_grid,
        colorscale='Viridis', opacity=0.4,
        name='True Greek (Reference)', showscale=False,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    ))

    # Layer 2: The TabPFN Model (Noisy) - Red/Blue
    fig.add_trace(go.Surface(
        z=grad_k_pfn, x=F_grid, y=K_grid,
        colorscale='RdBu', 
        name='TabPFN Greek (Noisy)', showscale=True,
        colorbar=dict(title="dV/dK Intensity")
    ))

    fig.update_layout(
        title='Risk Surface Analysis: Why Raw TabPFN Fails on Greeks',
        scene=dict(
            xaxis_title='Forward (F)',
            yaxis_title='Strike (K)',
            zaxis_title='Sensitivity (dV/dK)',
            aspectmode='cube'
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_file = "interactive_risk_fast.html"
    fig.write_html(output_file)
    print(f"✅ Done! Open '{output_file}' in your browser.")
    
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(output_file))
    except:
        pass

if __name__ == "__main__":
    generate_3d_comparison_fast()