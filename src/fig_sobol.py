import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
csv_path = data_dir / "sabr_with_derivatives_scaled.csv" 

def plot_sabr_inputs_with_beta():
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    # Select the key variables
    cols_to_plot = ['beta', 'alpha', 'rho', 'volvol', 'F', 'K']
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]

    # Subsample for responsiveness (2000 points max)
    n_samples = min(2000, len(df))
    df_sample = df.sample(n=n_samples, random_state=42)

    print("Generating the pairplot matrix (zoomed out)...")
    
    # Style visuel Dark Mode
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#1c1c1c", 
        "figure.facecolor": "#1c1c1c", 
        "text.color": "white", 
        "axes.labelcolor": "white", 
        "xtick.color": "white", 
        "ytick.color": "white",
        "font.size": 10,           # Police globale plus petite
        "axes.labelsize": 12,      # Taille des noms d'axes
        "xtick.labelsize": 9,      # Taille des chiffres X
        "ytick.labelsize": 9       # Taille des chiffres Y
    })

    # 'height=2.5' enlarges the figure (zoom out).
    # 'aspect=1.1' makes the cells slightly rectangular for breathing room.
    g = sns.pairplot(
        df_sample[cols_to_plot], 
        diag_kind='hist', 
        corner=True, 
        height=2.5,  
        aspect=1.1,  
        plot_kws={'alpha': 0.4, 's': 10, 'color': '#00ffcc', 'edgecolor': 'none'},
        diag_kws={'color': '#ff007f', 'bins': 30}
    )

    # Overall title (adjusted to avoid overlap)
    g.figure.suptitle(
        "SABR Full Parameter Space Analysis\n(Sobol Uniformity & Non-Linear Transformations)", 
        y=1.03, fontsize=20, color='white', fontweight='bold'
    )

    # Rotation des labels X pour qu'ils ne se touchent pas
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='x', rotation=45)

    # Forcer l'ajustement global de la figure pour laisser de la marge
    plt.subplots_adjust(bottom=0.1, left=0.08, right=0.95, top=0.92)

    # Save
    graph_dir = current_dir.parent / "figures"
    graph_dir.mkdir(parents=True, exist_ok=True)
    save_path = graph_dir / "sabr_complete_pairplot_dezoomed.png"
    
    # bbox_inches='tight' s'assure de prendre toute l'image lors de l'export
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure generated and zoomed out. Saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_sabr_inputs_with_beta()