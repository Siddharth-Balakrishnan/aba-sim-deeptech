"""Unified Graph Generator for Deeptech Simulation Research Paper

Generates all 6 publication-ready figures (300 DPI) and saves them to:
  result_generator/results/graphs/

Figures:
  1. fig1_burnout_trajectory.png       - Cumulative Agent Burnout Over 52 Weeks
  2. fig2_pareto_front.png             - NSGA-II Pareto Front (Tech vs. Market)
  3. fig3_feature_stability.png        - Elastic Net Stability Selection Frequencies
  4. fig4_3d_burnout_surface.png       - 3D Response Surface of Burnout Probability
  5. fig5_sensitivity_heatmap.png      - Diverging Heatmap of Optimal Allocation Shifts
  6. fig6_phase_space_flow.png         - Phase Portrait Streamplot & Trajectories
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "graphs")
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


# ---------------------------------------------------------------------------
# Figure 1: Temporal Burnout Mapping (ABM Trajectory)
# ---------------------------------------------------------------------------
def generate_fig1():
    abm_path = os.path.join(ROOT_DIR, "abm_simulation", "outputs", "abm_longitudinal_data.csv")
    abm_df = pd.read_csv(abm_path)
    burnout_trend = abm_df.groupby("Week")["is_burned_out"].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(burnout_trend.index, burnout_trend.values, color="firebrick", linewidth=2.5)
    plt.fill_between(burnout_trend.index, burnout_trend.values, color="firebrick", alpha=0.1)
    plt.title("Cumulative Agent Burnout Over 52-Week Incubation", fontsize=14, pad=15)
    plt.xlabel("Simulation Week", fontsize=12)
    plt.ylabel("Total Burned Out Agents", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    out_path = os.path.join(OUTPUT_DIR, "fig1_burnout_trajectory.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Fig 1: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: NSGA-II Pareto Front Boundary
# ---------------------------------------------------------------------------
def generate_fig2():
    pareto_path = os.path.join(ROOT_DIR, "optimization_and_ml", "outputs", "pareto_optimal_allocations.csv")
    pareto_df = pd.read_csv(pareto_path)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        pareto_df["T_tech"],
        pareto_df["T_mkt"],
        c=pareto_df["T_acad"],
        cmap="viridis",
        s=60,
        alpha=0.8,
        edgecolor="w",
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(r"Academic Time Allocation ($T_{acad}$)", rotation=270, labelpad=15)

    plt.title("NSGA-II Pareto Front: R&D vs Market Validation", fontsize=14, pad=15)
    plt.xlabel(r"Technical Allocation ($T_{tech}$)", fontsize=12)
    plt.ylabel(r"Market Allocation ($T_{mkt}$)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    out_path = os.path.join(OUTPUT_DIR, "fig2_pareto_front.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Fig 2: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Predictive Feature Stability (Elastic Net)
# ---------------------------------------------------------------------------
def generate_fig3():
    features = [
        "Deeptech Resilience\n(Base Trait)",
        "Lab-to-MVP Efficacy\n(Base Trait)",
        "Efficacy × Exposure\n(Interaction)",
    ]
    scores = [1.0, 100.0, 100.0]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x=scores,
        y=features,
        hue=features,
        palette=["#cccccc", "#4c72b0", "#55a868"],
        legend=False,
    )
    plt.title("Predictive Feature Stability (100 Bootstraps)", fontsize=14, pad=15)
    plt.xlabel("Selection Frequency (%)", fontsize=12)
    plt.xlim(0, 110)

    for i, v in enumerate(scores):
        ax.text(v + 1.5, i, f"{v}%", va="center", fontweight="bold", color="black")

    out_path = os.path.join(OUTPUT_DIR, "fig3_feature_stability.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Fig 3: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: 3D Surface Plot (Burnout Probability)
# ---------------------------------------------------------------------------
def generate_fig4():
    x = np.linspace(1.0, 7.0, 100)       # Lab-to-MVP Efficacy (E)
    y = np.linspace(0.1, 2.0, 100)       # Failure Exposure (1/R)
    X, Y = np.meshgrid(x, y)

    logits = -2.0 + 1.5 * Y - 0.8 * X + 0.5 * (X * Y)
    Z = 1.0 / (1.0 + np.exp(-logits))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.92,
        antialiased=True,
        rstride=2,
        cstride=2,
    )
    ax.view_init(elev=25, azim=-45)

    ax.set_xlabel(r"Lab-to-MVP Efficacy ($E$)", labelpad=12, fontweight="bold")
    ax.set_ylabel(r"Failure Exposure ($1/R$)", labelpad=12, fontweight="bold")
    ax.set_zlabel(r"Burnout Probability $P(\mathrm{Burnout})$", labelpad=10, fontweight="bold")
    ax.set_title(
        r"Burnout Probability Response Surface: $E \times (1/R)$ Interaction",
        fontsize=13,
        pad=20,
        fontweight="bold",
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label(r"$P(\mathrm{Burnout})$", rotation=270, labelpad=18, fontweight="bold")

    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.08)
    ax.grid(True, linestyle=":", alpha=0.5)

    out_path = os.path.join(OUTPUT_DIR, "fig4_3d_burnout_surface.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Fig 4: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Diverging Sensitivity Heatmap
# ---------------------------------------------------------------------------
def generate_fig5():
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

    out_path = os.path.join(OUTPUT_DIR, "fig5_sensitivity_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Fig 5: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: Phase-Space Vector Field / Phase Portrait Streamplot
# ---------------------------------------------------------------------------
def generate_fig6():
    abm_path = os.path.join(ROOT_DIR, "abm_simulation", "outputs", "abm_longitudinal_data.csv")
    df = pd.read_csv(abm_path)

    t_vals = np.linspace(0, 1100, 120)
    a_vals = np.linspace(0, 220, 100)
    T, A = np.meshgrid(t_vals, a_vals)

    burnout_mask = A <= 40.0
    viable_mask = ~burnout_mask

    U = np.zeros_like(T)
    V = np.zeros_like(A)

    U[viable_mask] = 22.0 * (1.0 - T[viable_mask] / 1200.0) * (A[viable_mask] / 100.0)**0.3
    V[viable_mask] = 1.2 * (115.0 - A[viable_mask]) - 0.015 * (T[viable_mask] - 200.0)

    U[burnout_mask] = -0.05 * T[burnout_mask] * (1.0 - A[burnout_mask] / 40.0)
    V[burnout_mask] = -1.5 * (A[burnout_mask] - 15.0)

    speed = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor="#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Shaded Burnout Basin
    ax.axhspan(0, 40, color="#fde8e8", alpha=0.85, zorder=1)
    ax.text(
        820, 18, "BURNOUT SINK / RETENTION COLLAPSE\n($A_{score} \\leq 40$, $\\dot{T} \\rightarrow 0$)",
        fontsize=9.5, fontweight="bold", color="#b91c1c", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#ef4444", alpha=0.9),
        zorder=5,
    )

    ax.axhspan(40, 220, color="#f0fdf4", alpha=0.35, zorder=1)
    ax.axhline(40, color="#dc2626", linestyle="--", linewidth=2.0, zorder=4)

    strm = ax.streamplot(
        T, A, U, V,
        color=speed,
        cmap="Blues",
        density=1.3,
        linewidth=1.1,
        arrowsize=1.2,
        arrowstyle="->",
        zorder=2,
    )

    cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(
        r"Phase Flow Velocity $\Vert \mathbf{v} \Vert = \sqrt{\dot{T}^2 + \dot{A}^2}$",
        fontsize=10.5, labelpad=10, fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=9)

    # Empirical trajectories from ABM
    burned_ids = df[df["is_burned_out"]]["Agent_ID"].unique()
    survived_ids = df[~df["Agent_ID"].isin(burned_ids)]["Agent_ID"].unique()

    for aid in survived_ids[:3]:
        traj = df[df["Agent_ID"] == aid].sort_values("Week")
        ax.plot(traj["tech_readiness"], traj["academic_score"],
                color="#0f766e", linewidth=2.0, alpha=0.85, zorder=6)
        ax.scatter(traj["tech_readiness"].iloc[-1], traj["academic_score"].iloc[-1],
                   color="#0f766e", s=45, marker="o", edgecolors="white", linewidths=1.2, zorder=7)

    for aid in burned_ids[:3]:
        traj = df[df["Agent_ID"] == aid].sort_values("Week")
        ax.plot(traj["tech_readiness"], traj["academic_score"],
                color="#b91c1c", linewidth=2.0, linestyle="-.", alpha=0.85, zorder=6)
        b_pt = traj[traj["is_burned_out"]].iloc[0]
        ax.scatter(b_pt["tech_readiness"], b_pt["academic_score"],
                   color="#b91c1c", s=70, marker="X", edgecolors="white", linewidths=1.2, zorder=7)

    ax.annotate(
        "Sustainable Survival Manifold\n($A \\approx 115,\\; \\dot{T} > 0$)",
        xy=(750, 115), xytext=(580, 175),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.5),
        fontsize=10, fontweight="bold", color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#0f766e", alpha=0.9),
        zorder=8,
    )

    ax.set_xlim(0, 1050)
    ax.set_ylim(0, 210)
    ax.set_xlabel(r"Technical Progress / Readiness ($T_{\mathrm{tech}}$)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"Academic Standing ($A_{\mathrm{score}}$)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        r"Figure 6: Phase Portrait of Student-Founder Cognitive Allocation Dynamics",
        fontsize=13.5, fontweight="bold", pad=15, loc="left", color="#1e293b",
    )

    legend_elements = [
        Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=2.0, label=r"Critical Academic Threshold ($A_{score} = 40$)"),
        Line2D([0], [0], color="#0f766e", linewidth=2.0, marker="o", markersize=6, label="Empirical Surviving Trajectory"),
        Line2D([0], [0], color="#b91c1c", linewidth=2.0, linestyle="-.", marker="X", markersize=7, label="Empirical Burnout Trajectory (Terminus)"),
        Patch(facecolor="#fde8e8", edgecolor="#ef4444", label="Burnout / Collapse Basin"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.45, color="#94a3b8")

    out_path = os.path.join(OUTPUT_DIR, "fig6_phase_space_flow.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Fig 6: {out_path}")


def main():
    print("=" * 65)
    print("Generating All Publication Figures (Fig 1 – Fig 6)")
    print(f"Output Directory: {OUTPUT_DIR}/")
    print("=" * 65)
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    print("\n✅ All 6 publication figures generated successfully.")


if __name__ == "__main__":
    main()

