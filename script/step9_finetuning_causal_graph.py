import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabpfn import TabPFNClassifier

# ==========================================
# 1. LA STRUCTURE DU MODÈLE (Intégrée)
# ==========================================
class TabPFNSABRRegressor(nn.Module):
    """Modèle hybride : Encodeur TabPFN + Tête MLP spécialisée SABR."""
    def __init__(self, n_out=7, hidden_dims=[512, 256, 128]):
        super().__init__()
        # Initialisation du moteur TabPFN
        base_model = TabPFNClassifier(device='cpu')
        # Dummy fit pour forcer le chargement des poids internes (model_)
        base_model.fit(np.random.randn(5, 8), np.random.randint(0, 2, 5))
        
        # Accès sécurisé à l'encodeur Transformer
        if hasattr(base_model, 'model_'):
            full_model = base_model.model_
        else:
            full_model = base_model.model
            
        self.encoder = full_model[2] # Le Transformer est le 3ème bloc
        self.d_model = getattr(self.encoder, 'emsize', 512)
        
        # Construction de la tête MLP (basée sur ton optimisation Step 7)
        layers = []
        prev_dim = self.d_model
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(), # Activation Swish
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_out))
        self.custom_head = nn.Sequential(*layers)

    def forward(self, x):
        # Format [Seq=1, Batch, Feats]
        x_in = x.unsqueeze(0)
        dummy_y = torch.zeros(1, x.size(0), 1).to(x.device)
        
        # Extraction des features (on garde les gradients pour le fine-tuning)
        features = self.encoder(x_in, dummy_y)
        return self.custom_head(features.squeeze(0))

# ==========================================
# 2. LE GÉNÉRATEUR CAUSAL (Simulation SABR)
# ==========================================
class SABRCausalGenerator:
    """Moteur physique générant des relations causes -> effets."""
    def __init__(self, beta=0.5):
        self.beta = beta

    def generate(self, batch_size=32):
        # Échantillonnage des paramètres (Causes)
        alpha = np.random.uniform(0.1, 0.4, (batch_size, 1))
        rho = np.random.uniform(-0.8, -0.2, (batch_size, 1))
        volvol = np.random.uniform(0.2, 0.6, (batch_size, 1))
        F = np.random.uniform(95, 105, (batch_size, 1))
        K = np.random.uniform(85, 115, (batch_size, 1))
        T = np.random.uniform(0.1, 1.5, (batch_size, 1))
        v_atm_n = alpha * (F ** (self.beta - 1))
        log_moneyness = np.log(K / F)

        # Calcul de la Volatilité (Effet) - Formule Hagan simplifiée
        vol = alpha * (1 + (volvol**2 * (2-3*rho**2)/24) * T)
        
        # Dérivées (Greeks) pour la Sobolev Loss
        dV_dalpha = np.ones_like(vol) * 1.1
        dV_drho = np.ones_like(vol) * 0.4
        dV_dvolvol = np.ones_like(vol) * 0.7
        dV_dF = (vol / F) * -0.05
        dV_dK = (vol / K) * 0.05
        dV_dT = np.ones_like(vol) * 0.02

        X = np.hstack([np.full((batch_size, 1), self.beta), rho, volvol, v_atm_n, alpha, F, K, log_moneyness])
        y = np.hstack([vol, dV_dalpha, dV_drho, dV_dvolvol, dV_dF, dV_dK, dV_dT])
        
        return torch.FloatTensor(X), torch.FloatTensor(y)

# ==========================================
# 3. EXÉCUTION DU FINE-TUNING CAUSAL
# ==========================================
def main():
    # Chargement config Step 7
    try:
        with open("ray_results/best_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("⚠️ Config Step 7 manquante, utilisation des paramètres par défaut.")
        config = {'hidden_dims': [512, 256, 128], 'lr': 0.001, 'value_weight': 1.0, 'derivative_weight': 0.1, 'batch_size': 32}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabPFNSABRRegressor(hidden_dims=config['hidden_dims']).to(device)

    # Chargement des poids Step 8
    try:
        model.load_state_dict(torch.load("tabpfn_sabr_final.pth", map_location=device))
        print("✅ Poids de la Step 8 chargés.")
    except:
        print("ℹ️ tabpfn_sabr_final.pth non trouvé, initialisation à zéro.")

    # ON LIBÈRE LES GRADIENTS POUR LE VRAI FINE-TUNING
    for param in model.parameters():
        param.requires_grad = True

    generator = SABRCausalGenerator()
    from step6_loss_with_derivatives import create_loss_function
    criterion = create_loss_function(
        loss_type='derivative', 
        value_weight=config['value_weight'], 
        derivative_weight=config['derivative_weight']
    )

    # Optimiseur : Le Transformer apprend 10x moins vite que la tête MLP
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['lr'] * 0.1},
        {'params': model.custom_head.parameters(), 'lr': config['lr']}
    ])

    print("🧠 Fine-tuning Causal en cours...")
    model.train()
    for step in range(401):
        X_batch, y_batch = generator.generate(batch_size=config['batch_size'])
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        out = model(X_batch)
        
        # Sobolev Loss
        loss, _ = criterion(out[:, 0:1], y_batch[:, 0:1], 
                           {f'd{i}': out[:, i:i+1] for i in range(1, 7)},
                           {f'd{i}': y_batch[:, i:i+1] for i in range(1, 7)})
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}/400 | Loss: {loss.item():.6f}")

    # ==========================================
    # 4. GRAPHIQUE COMPARATIF FINAL
    # ==========================================
    model.eval()
    X_test, y_true = generator.generate(batch_size=200)
    idx = X_test[:, -1].argsort() # Tri par log-moneyness
    X_test, y_true = X_test[idx], y_true[idx]
    
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu()

    plt.figure(figsize=(10, 6))
    plt.plot(X_test[:, -1], y_true[:, 0], 'k--', label='Théorie (Causal Generator)', alpha=0.7)
    plt.plot(X_test[:, -1], y_pred[:, 0], 'r-', label='Ton Modèle Causal (Step 9)', linewidth=2)
    plt.title("Duel de Performance : Théorie vs Ton IA Causal")
    plt.xlabel("Log-Moneyness (K/F)")
    plt.ylabel("Volatilité")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("performance_duel_step9.png")
    plt.show()

    torch.save(model.state_dict(), "tabpfn_sabr_causal_final.pth")
    print("✅ Modèle final sauvegardé : tabpfn_sabr_causal_final.pth")

if __name__ == "__main__":
    main()