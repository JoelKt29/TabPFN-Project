import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gradient_density_alignment():
    # 1. Génération de données synthétiques imitant le comportement SABR
    moneyness = np.linspace(0.8, 1.2, 1000)
    
    # Imitation de la magnitude du Skew SABR (Pic très fort en ATM, K/F = 1.0)
    # Plus on s'éloigne de 1.0, plus la dérivée s'aplatit
    skew_magnitude = 1.0 / (np.abs(moneyness - 1.0) + 0.05) 
    
    # Simulation des datasets
    n_points = 20000
    
    # Dataset Sobol (Uniforme entre 0.8 et 1.2)
    sobol_samples = np.random.uniform(0.8, 1.2, n_points)
    
    # Dataset AMR (La probabilité de tirage est proportionnelle à la dérivée)
    probabilities = skew_magnitude / np.sum(skew_magnitude)
    amr_samples = np.random.choice(moneyness, size=n_points, p=probabilities)

    # 2. Configuration du style
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Data Allocation vs Mathematical Complexity\n(Sobol vs Adaptive Mesh Refinement)", 
                 fontsize=16, fontweight='bold', y=1.02)

    # --- SOUS-GRAPHIQUE 1 : SOBOL ---
    color_grad = '#00ffcc'
    color_hist_sobol = '#aaaaaa'
    
    # Axe de gauche : Le Gradient
    ax1.plot(moneyness, skew_magnitude, color=color_grad, linewidth=2.5, label='Complexity: Skew Magnitude |dV/dK|')
    ax1.set_ylabel("Gradient Magnitude", color=color_grad, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_grad)
    
    # Axe de droite : La Densité Sobol
    ax1_twin = ax1.twinx()
    sns.histplot(sobol_samples, bins=60, color=color_hist_sobol, alpha=0.4, ax=ax1_twin, 
                 stat='density', label='Data Density (Sobol)')
    ax1_twin.set_ylabel("Data Density", color=color_hist_sobol)
    ax1_twin.tick_params(axis='y', labelcolor=color_hist_sobol)
    
    ax1.set_title("Standard Uniform Sampling (Mismatch between complexity and resources)", fontsize=12, pad=10)
    
    # Légendes combinées
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # --- SOUS-GRAPHIQUE 2 : AMR ---
    color_hist_amr = '#ff007f'
    
    # Axe de gauche : Le Gradient
    ax2.plot(moneyness, skew_magnitude, color=color_grad, linewidth=2.5, label='Complexity: Skew Magnitude |dV/dK|')
    ax2.set_ylabel("Gradient Magnitude", color=color_grad, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_grad)
    
    # Axe de droite : La Densité AMR
    ax2_twin = ax2.twinx()
    sns.histplot(amr_samples, bins=60, color=color_hist_amr, alpha=0.6, ax=ax2_twin, 
                 stat='density', label='Data Density (AMR)')
    ax2_twin.set_ylabel("Data Density", color=color_hist_amr)
    ax2_twin.tick_params(axis='y', labelcolor=color_hist_amr)
    
    ax2.set_title("Adaptive Mesh Refinement (Data allocation perfectly matches complexity)", fontsize=12, pad=10)
    ax2.set_xlabel("Moneyness (K/F)", fontsize=12, fontweight='bold')
    
    # Légendes combinées
    lines_3, labels_3 = ax2.get_legend_handles_labels()
    lines_4, labels_4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')

    # Ajustement et sauvegarde
    plt.tight_layout()
    plt.savefig("gradient_density_alignment.png", dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé sous 'gradient_density_alignment.png'")
    plt.show()

if __name__ == "__main__":
    plot_gradient_density_alignment()