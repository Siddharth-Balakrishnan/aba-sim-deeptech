"""Figure 6: Phase-Space Vector Field / Phase Portrait Streamplot

Visualizes the dynamic trajectory of student founders in phase space:
Technical Progress (T_tech) vs. Academic Standing (A_score).

Shows vector flows converging toward a sustainable survival manifold
versus diverging into a burnout sink (A_score < 40 and T_tech collapse).

Saved to: optimization_and_ml/outputs/figures/fig6_phase_space_flow.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_fig6():
    # -------------------------------------------------------------
    # 1. Load empirical ABM trajectories
    # -------------------------------------------------------------
    abm_path = os.path.join(ROOT_DIR, "abm_simulation", "outputs", "abm_longitudinal_data.csv")
    df = pd.read_csv(abm_path)

    # -------------------------------------------------------------
    # 2. Define Phase Space Grid
    # -------------------------------------------------------------
    # T: Technical Readiness [0, 1100]
    # A: Academic Score [0, 220]
    t_vals = np.linspace(0, 1100, 120)
    a_vals = np.linspace(0, 220, 100)
    T, A = np.meshgrid(t_vals, a_vals)

    # -------------------------------------------------------------
    # 3. Formulate Dynamical Vector Field (dT/dt, dA/dt)
    # -------------------------------------------------------------
    # Based on the ABM cognitive allocation dynamics:
    # When A > 40:
    #   Agents allocate bandwidth between Tech and Academic.
    #   dT/dt > 0 drives tech progress toward frontier ~ 1000.
    #   dA/dt depends on whether academic allocation offsets the -30 pts/wk drain.
    #   At high tech readiness, friction/stabilization occurs.
    # When A <= 40:
    #   Burnout boundary breached: cognitive collapse.
    #   Effort ceases, tech progress stalls (dT/dt -> 0),
    #   and academic standing decays into a terminal institutional failure sink.
    
    burnout_mask = A <= 40.0
    viable_mask = ~burnout_mask

    U = np.zeros_like(T)  # dT/dt
    V = np.zeros_like(A)  # dA/dt

    # Viable regime (A > 40):
    # Higher A provides cognitive safety, enabling sustained tech investment
    # Natural equilibrium band around A ~ 100 - 150
    # Technical readiness approaches saturation / frontier
    U[viable_mask] = 22.0 * (1.0 - T[viable_mask] / 1200.0) * (A[viable_mask] / 100.0)**0.3
    # Academic score experiences mean reversion toward equilibrium allocation
    # with downward drag if tech demands are excessively high
    V[viable_mask] = 1.2 * (115.0 - A[viable_mask]) - 0.015 * (T[viable_mask] - 200.0)

    # Burnout sink regime (A <= 40):
    # Technical progress collapses to zero / slight regression
    U[burnout_mask] = -0.05 * T[burnout_mask] * (1.0 - A[burnout_mask] / 40.0)
    # Academic standing spirals down into institutional failure
    V[burnout_mask] = -1.5 * (A[burnout_mask] - 15.0)

    # Velocity magnitude for streamline coloring
    speed = np.sqrt(U**2 + V**2)

    # -------------------------------------------------------------
    # 4. Plot Setup (Minimalist Academic Editorial Theme)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor="#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Shaded Burnout Basin (A <= 40)
    ax.axhspan(0, 40, color="#fde8e8", alpha=0.85, zorder=1)
    ax.text(
        820, 18, "BURNOUT SINK / RETENTION COLLAPSE\n($A_{score} \\leq 40$, $\\dot{T} \\rightarrow 0$)",
        fontsize=9.5, fontweight="bold", color="#b91c1c", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#ef4444", alpha=0.9),
        zorder=5
    )

    # Sustainable Incubation Zone (A > 40)
    ax.axhspan(40, 220, color="#f0fdf4", alpha=0.35, zorder=1)

    # Critical Burnout Boundary (A = 40)
    ax.axhline(40, color="#dc2626", linestyle="--", linewidth=2.0, zorder=4,
               label=r"Critical Academic Threshold ($A_{score} = 40$)")

    # -------------------------------------------------------------
    # 5. Streamplot Vector Field
    # -------------------------------------------------------------
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

    # Colorbar for flow velocity
    cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"Phase Flow Velocity $\Vert \mathbf{v} \Vert = \sqrt{\dot{T}^2 + \dot{A}^2}$",
                   fontsize=10.5, labelpad=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # -------------------------------------------------------------
    # 6. Overlay Empirical Student-Founder Trajectories from ABM
    # -------------------------------------------------------------
    # Select representative agents: surviving vs burned-out
    burned_ids = df[df["is_burned_out"]]["Agent_ID"].unique()
    survived_ids = df[~df["Agent_ID"].isin(burned_ids)]["Agent_ID"].unique()

    # Plot 3 representative surviving trajectories
    for aid in survived_ids[:3]:
        agent_traj = df[df["Agent_ID"] == aid].sort_values("Week")
        ax.plot(agent_traj["tech_readiness"], agent_traj["academic_score"],
                color="#0f766e", linewidth=2.0, alpha=0.85, zorder=6)
        # End point
        ax.scatter(agent_traj["tech_readiness"].iloc[-1], agent_traj["academic_score"].iloc[-1],
                   color="#0f766e", s=45, marker="o", edgecolors="white", linewidths=1.2, zorder=7)

    # Plot 3 representative burned-out trajectories
    for aid in burned_ids[:3]:
        agent_traj = df[df["Agent_ID"] == aid].sort_values("Week")
        ax.plot(agent_traj["tech_readiness"], agent_traj["academic_score"],
                color="#b91c1c", linewidth=2.0, linestyle="-.", alpha=0.85, zorder=6)
        # Burnout point (first point where is_burned_out is True)
        b_pt = agent_traj[agent_traj["is_burned_out"]].iloc[0]
        ax.scatter(b_pt["tech_readiness"], b_pt["academic_score"],
                   color="#b91c1c", s=70, marker="X", edgecolors="white", linewidths=1.2, zorder=7)

    # Annotation for Stable Attractor / Viable Basin
    ax.annotate(
        "Sustainable Survival Manifold\n($A \\approx 115,\\; \\dot{T} > 0$)",
        xy=(750, 115), xytext=(580, 175),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.5),
        fontsize=10, fontweight="bold", color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#0f766e", alpha=0.9),
        zorder=8
    )

    # -------------------------------------------------------------
    # 7. Formatting & Legend
    # -------------------------------------------------------------
    ax.set_xlim(0, 1050)
    ax.set_ylim(0, 210)
    ax.set_xlabel(r"Technical Progress / Readiness ($T_{\mathrm{tech}}$)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"Academic Standing ($A_{\mathrm{score}}$)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        r"Figure 6: Phase Portrait of Student-Founder Cognitive Allocation Dynamics",
        fontsize=13.5, fontweight="bold", pad=15, loc="left", color="#1e293b"
    )

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=2.0, label="Critical Academic Threshold ($A_{score} = 40$)"),
        Line2D([0], [0], color="#0f766e", linewidth=2.0, marker="o", markersize=6, label="Empirical Surviving Trajectory"),
        Line2D([0], [0], color="#b91c1c", linewidth=2.0, linestyle="-.", marker="X", markersize=7, label="Empirical Burnout Trajectory (Terminus)"),
        Patch(facecolor="#fde8e8", edgecolor="#ef4444", label="Burnout / Collapse Basin"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

    ax.grid(True, linestyle=":", alpha=0.45, color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    output_path = os.path.join(OUTPUT_DIR, "fig6_phase_space_flow.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Generated Graph 6: {output_path}")


if __name__ == "__main__":
    generate_fig6()
