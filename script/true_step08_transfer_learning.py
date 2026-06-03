from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from tabpfn import TabPFNRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from step06_loss_with_derivatives import DerivativeLoss

# ==============================
# PATHS
# ==============================
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
config_path = current_dir / "ray_results" / "best_config.json"

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# ==============================
# ACTIVATIONS
# ==============================
def get_activation(name: str):
    activations = {
        'swish': nn.SiLU(),
        'mish': nn.Mish(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'relu': nn.ReLU()
    }
    return activations.get(name.lower(), nn.SiLU())

# ==============================
# MODEL
# ==============================
class TabPFNStackingModel(nn.Module):
    def __init__(self, config, n_outputs=7):
        super().__init__()

        # 8 features SABR + 3 features TabPFN enrichies (pred, pred^2, abs(pred))
        input_dim = 11 
        layers = []
        prev_dim = input_dim

        h_dims = config.get('hidden_dims', [512, 256, 128])
        activation_name = config.get('activation', 'swish')

        for h_dim in h_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                get_activation(activation_name),
                nn.Dropout(config.get('dropout', 0.1))
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, x_sabr, x_tabpfn):
        x = torch.cat([x_sabr, x_tabpfn], dim=1)
        return self.head(x)

# ==============================
# MAIN
# ==============================
def run_step8():
    # ---- Load config ----
    with open(config_path, "r") as f:
        config = json.load(f)

    # 1. Chargement du nouveau dataset Hybrid Mesh
    csv_path = data_dir / "sabr_hybrid_mesh_scaled.csv"
    print(f"📂 Chargement des données : {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"📊 Taille dataset : {len(df)} points")

    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    target_cols = [
        'volatility_scaled',
        'dV_dbeta_scaled',
        'dV_drho_scaled',
        'dV_dvolvol_scaled',
        'dV_dalpha_scaled', # <-- Correction vitale appliquée ici
        'dV_dF_scaled',
        'dV_dK_scaled'
    ]

    X = df[feature_cols].values
    y = df[target_cols].values

    # ==============================
    # 2. SPLIT (ANTI-LEAKAGE)
    # ==============================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==============================
    # 3. TABPFN ORACLE (TRAINING)
    # ==============================
    print("🧠 Entraînement de l'Oracle TabPFN (sur le train set)...")
    tabpfn = TabPFNRegressor(
        device=device,
        n_estimators=4,
        ignore_pretraining_limits=True
    )

    # Contexte parfait de 1000 points maximum
    ctx_size = min(1000, len(X_train))
    idx = np.random.choice(len(X_train), ctx_size, replace=False)
    tabpfn.fit(X_train[idx], y_train[idx, 0])

    # ==============================
    # 4. TABPFN INFERENCE (CHUNKING)
    # ==============================
    print("🔮 Inférence TabPFN (Chunking)...")
    def predict_chunks(X_data, chunk_size=2000):
        preds = []
        for i in tqdm(range(0, len(X_data), chunk_size), desc="Prediction"):
            chunk = X_data[i:i+chunk_size]
            preds.append(tabpfn.predict(chunk).reshape(-1, 1))
        return np.vstack(preds)

    pfn_train = predict_chunks(X_train)
    pfn_val = predict_chunks(X_val)

    # ==============================
    # 5. FEATURE ENGINEERING TABPFN
    # ==============================
    def enrich(pred):
        return np.hstack([
            pred,
            pred**2,
            np.abs(pred)
        ])

    pfn_train = enrich(pfn_train)
    pfn_val = enrich(pfn_val)

    # ==============================
    # 6. NORMALISATION & SAUVEGARDE MLOPS
    # ==============================
    mean = pfn_train.mean(axis=0)
    std = pfn_train.std(axis=0) + 1e-8 # Sécurité division par zéro

    pfn_train = (pfn_train - mean) / std
    pfn_val = (pfn_val - mean) / std

    # ---> SAUVEGARDE VITALE POUR L'ÉTAPE 9 <---
    tabpfn_stats = {'mean': mean.tolist(), 'std': std.tolist()}
    scaler_path = current_dir / "tabpfn_scaler.json"
    with open(scaler_path, "w") as f:
        json.dump(tabpfn_stats, f, indent=4)
    print(f"💾 Scaler TabPFN sauvegardé : {scaler_path.name}")

    # ==============================
    # 7. CONVERSION PYTORCH
    # ==============================
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    Xp_train_t = torch.FloatTensor(pfn_train)
    Xp_val_t = torch.FloatTensor(pfn_val)
    y_train_t = torch.FloatTensor(y_train)
    y_val_t = torch.FloatTensor(y_val)

    # ==============================
    # 8. INITIALISATION MLP
    # ==============================
    print("🏗️ Initialisation du réseau hybride...")
    model = TabPFNStackingModel(config).to(device)

    criterion = DerivativeLoss(
        value_weight=config.get('value_weight', 1.0),
        derivative_weight=config.get('derivative_weight', 0.5)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )

    batch_size = config.get('batch_size', 64)
    epochs = config.get('num_epochs', 100)

    # ==============================
    # 9. BOUCLE D'ENTRAÎNEMENT SOBOLEV
    # ==============================
    print("🚀 Début de l'entraînement...")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train_t.size(0))
        train_loss = 0
        num_batches_train = 0

        for i in range(0, X_train_t.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xs = X_train_t[idx].to(device)
            xp = Xp_train_t[idx].to(device)
            yt = y_train_t[idx].to(device)

            optimizer.zero_grad()
            out = model(xs, xp)

            pred_d = {f'd{j}': out[:, j:j+1] for j in range(1, 7)}
            true_d = {f'd{j}': yt[:, j:j+1] for j in range(1, 7)}

            loss, _ = criterion(out[:, 0:1], yt[:, 0:1], pred_d, true_d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches_train += 1

        # ==============================
        # VALIDATION
        # ==============================
        model.eval()
        val_loss = 0
        val_mae = 0
        num_batches_val = 0

        with torch.no_grad():
            for i in range(0, X_val_t.size(0), batch_size):
                xs = X_val_t[i:i+batch_size].to(device)
                xp = Xp_val_t[i:i+batch_size].to(device)
                yt = y_val_t[i:i+batch_size].to(device)

                out = model(xs, xp)

                pred_d = {f'd{j}': out[:, j:j+1] for j in range(1, 7)}
                true_d = {f'd{j}': yt[:, j:j+1] for j in range(1, 7)}

                loss, _ = criterion(out[:, 0:1], yt[:, 0:1], pred_d, true_d)

                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(out[:, 0:1] - yt[:, 0:1])).item()
                num_batches_val += 1

        # Affichage propre et moyenné
        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_train_loss = train_loss / num_batches_train
            avg_val_loss = val_loss / num_batches_val
            avg_val_mae = val_mae / num_batches_val
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.6f}")

    # ==============================
    # 10. SAUVEGARDE
    # ==============================
    save_path = current_dir / "tabpfn_sabr_step8_stacking.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modèle sauvegardé avec succès : {save_path.name}")

if __name__ == "__main__":
    run_step8()