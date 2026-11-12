"""Publication-ready complex figures for the deeptech simulation pipeline.

Figure 4: 3D Surface Plot of Burnout Probability (Interaction Surface)
Figure 5: Diverging Sensitivity Heatmap (Pareto-Optimal Allocation Shifts)

Saved to: optimization_and_ml/outputs/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication styling defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def generate_plot1_3d_surface():
    """Plot 1: 3D Surface Plot of Burnout Probability as a function of

    Lab-to-MVP Efficacy (E) and Failure Exposure (1/R).
    """
    # Grid domain
    x = np.linspace(1.0, 7.0, 100)       # Lab-to-MVP Efficacy (E)
    y = np.linspace(0.1, 2.0, 100)       # Failure Exposure (1/R)
    X, Y = np.meshgrid(x, y)

    # Sigmoid response with nonlinear interaction term E * (1/R)
    logits = -2.0 + 1.5 * Y - 0.8 * X + 0.5 * (X * Y)
    Z = 1.0 / (1.0 + np.exp(-logits))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Surface plot
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.92,
        antialiased=True,
        rstride=2,
        cstride=2,
    )

    # Viewing angle
    ax.view_init(elev=25, azim=-45)

    # Axis labels with mathematical notation
    ax.set_xlabel(r"Lab-to-MVP Efficacy ($E$)", labelpad=12, fontweight="bold")
    ax.set_ylabel(r"Failure Exposure ($1/R$)", labelpad=12, fontweight="bold")
    ax.set_zlabel(r"Burnout Probability $P(\mathrm{Burnout})$", labelpad=10, fontweight="bold")

    ax.set_title(
        r"Burnout Probability Response Surface: $E \times (1/R)$ Interaction",
        fontsize=13,
        pad=20,
        fontweight="bold",
    )

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label(r"$P(\mathrm{Burnout})$", rotation=270, labelpad=18, fontweight="bold")

    # Grid & panes styling
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.08)
    ax.grid(True, linestyle=":", alpha=0.5)

    output_path = os.path.join(OUTPUT_DIR, "fig4_3d_burnout_surface.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Plot 1: {output_path}")


def generate_plot2_sensitivity_heatmap():
    """Plot 2: Diverging Sensitivity Heatmap for Pareto allocation shifts."""
    rows = [
        "High Academic Penalty\n(-50 pts/wk)",
        "High Tech Shock Prob\n(20% chance)",
        "Low Initial Resilience\n(Baseline shift)",
    ]
    cols = [
        r"$\Delta$ Tech R&D ($T_{tech}$)",
        r"$\Delta$ Market Val ($T_{mkt}$)",
        r"$\Delta$ Academic ($T_{acad}$)",
    ]
    data = np.array([
        [-0.185,  0.065,  0.120],
        [-0.092, -0.021,  0.113],
        [-0.041, -0.012,  0.053],
    ])

    df_heatmap = pd.DataFrame(data, index=rows, columns=cols)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df_heatmap,
        cmap="coolwarm",
        center=0.0,
        annot=True,
        fmt=".3f",
        annot_kws={"weight": "bold", "size": 12},
        linewidths=1.5,
        linecolor="white",
        cbar_kws={
            "label": r"Pareto Allocation Shift ($\Delta$)",
            "pad": 0.04,
            "shrink": 0.85,
        },
        ax=ax,
    )

    ax.set_title(
        r"Sensitivity Analysis: Shift in Optimal Time Allocations ($\Delta$)",
        fontsize=13,
        pad=18,
        fontweight="bold",
    )
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    output_path = os.path.join(OUTPUT_DIR, "fig5_sensitivity_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Plot 2: {output_path}")


def main():
    print("=" * 65)
    print("Generating Publication-Ready Figures for Deeptech Simulation")
    print("=" * 65)
    generate_plot1_3d_surface()
    generate_plot2_sensitivity_heatmap()
    print(f"\n✅ Successfully generated both figures in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

