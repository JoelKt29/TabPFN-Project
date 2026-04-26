import numpy as np
import matplotlib.pyplot as plt

def sabr_hagan_vol(K, F, T, alpha, beta, rho, volvol):
    """
    Formule d'approximation de Hagan (2002) pour la volatilité implicite SABR.
    Implémentation standard compatible NumPy.
    """
    eps = 1e-07
    logfk = np.log(F / K)
    fkbeta = (F * K)**(1 - beta)
    
    a = (1 - beta)**2 * alpha**2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / (fkbeta**0.5)
    c = (2 - 3 * rho**2) * volvol**2 / 24
    d = fkbeta**0.5
    v = (1 - beta)**2 * logfk**2 / 24
    w = (1 - beta)**4 * logfk**4 / 1920
    z = volvol * (fkbeta**0.5) * logfk / alpha
    
    # Gestion de la fonction x(z)
    xz = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    
    numerator = alpha * (1 + (a + b + c) * T)
    denominator = d * (1 + v + w)
    
    # Prise en compte de la singularité ATM (quand K est très proche de F)
    vol_atm = numerator / d
    vol_standard = (numerator / denominator) * (z / xz)
    
    return np.where(np.abs(z) > eps, vol_standard, vol_atm)

def plot_sabr_parameters():
    # Configuration du marché de base
    F = 100.0  # Forward
    T = 1.0    # Maturité
    K = np.linspace(70, 130, 200) # Grille de Strikes
    
    # Paramètres SABR de base
    base_alpha = 0.20
    base_beta = 0.50
    base_rho = 0.00
    base_volvol = 0.40

    # Configuration visuelle (style Quant)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SABR Model: Parameter Impact on the Volatility Smile", 
                 fontsize=18, fontweight='bold', color='white', y=1.02)

    colors = ['#00ffcc', '#ff007f', '#ffcc00']

    # 1. Impact de Alpha (Niveau)
    ax = axes[0, 0]
    for i, alpha in enumerate([0.10, 0.20, 0.30]):
        vols = sabr_hagan_vol(K, F, T, alpha, base_beta, base_rho, base_volvol)
        ax.plot(K, vols, linewidth=2.5, color=colors[i], label=rf'$\alpha={alpha}$')
    ax.set_title(r"Impact of $\alpha$ (Volatility Level)", fontsize=14)

    # 2. Impact de Beta (Pente)
    ax = axes[0, 1]
    for i, beta in enumerate([0.2, 0.5, 0.8]):
        vols = sabr_hagan_vol(K, F, T, base_alpha, beta, base_rho, base_volvol)
        ax.plot(K, vols, linewidth=2.5, color=colors[i], label=rf'$\beta={beta}$')
    ax.set_title(r"Impact of $\beta$ (Backbone / Skewness)", fontsize=14)

    # 3. Impact de Rho (Asymétrie)
    ax = axes[1, 0]
    for i, rho in enumerate([-0.5, 0.0, 0.5]):
        vols = sabr_hagan_vol(K, F, T, base_alpha, base_beta, rho, base_volvol)
        ax.plot(K, vols, linewidth=2.5, color=colors[i], label=rf'$\rho={rho}$')
    ax.set_title(r"Impact of $\rho$ (Skew / Symmetry)", fontsize=14)

    # 4. Impact de VolVol (Courbure)
    ax = axes[1, 1]
    for i, volvol in enumerate([0.1, 0.4, 0.8]):
        vols = sabr_hagan_vol(K, F, T, base_alpha, base_beta, base_rho, volvol)
        ax.plot(K, vols, linewidth=2.5, color=colors[i], label=rf'$\nu={volvol}$')
    ax.set_title(r"Impact of $\nu$ (Vol of Vol / Smile Curvature)", fontsize=14)

    # Mise en forme commune à tous les graphiques
    for ax in axes.flat:
        ax.axvline(F, color='white', linestyle='--', alpha=0.5, label='ATM (K=F)')
        ax.set_xlabel("Strike (K)", fontsize=12)
        ax.set_ylabel("Implied Volatility", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig("sabr_parameters_impact.png", dpi=300, bbox_inches='tight')
    print("✅ Graphique généré et sauvegardé sous 'sabr_parameters_impact.png'")
    plt.show()

if __name__ == "__main__":
    plot_sabr_parameters()