import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a figures directory
out_dir = 'optimization_and_ml/outputs/figures'
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Figure 1: Temporal Burnout Mapping (ABM Trajectory)
# ---------------------------------------------------------
abm_df = pd.read_csv('abm_simulation/outputs/abm_longitudinal_data.csv')
burnout_trend = abm_df.groupby('Week')['is_burned_out'].sum()

plt.figure(figsize=(10, 6))
plt.plot(burnout_trend.index, burnout_trend.values, color='firebrick', linewidth=2.5)
plt.fill_between(burnout_trend.index, burnout_trend.values, color='firebrick', alpha=0.1)
plt.title('Cumulative Agent Burnout Over 52-Week Incubation', fontsize=14, pad=15)
plt.xlabel('Simulation Week', fontsize=12)
plt.ylabel('Total Burned Out Agents', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{out_dir}/fig1_burnout_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Figure 2: NSGA-II Pareto Front Boundary
# ---------------------------------------------------------
pareto_df = pd.read_csv('optimization_and_ml/outputs/pareto_optimal_allocations.csv')

plt.figure(figsize=(10, 6))
# Scatter plot representing the tradeoff between Tech and Market
sc = plt.scatter(pareto_df['T_tech'], pareto_df['T_mkt'], 
                 c=pareto_df['T_acad'], cmap='viridis', 
                 s=60, alpha=0.8, edgecolor='w')
cbar = plt.colorbar(sc)
cbar.set_label('Academic Time Allocation ($T_{acad}$)', rotation=270, labelpad=15)

plt.title('NSGA-II Pareto Front: R&D vs Market Validation', fontsize=14, pad=15)
plt.xlabel('Technical Allocation ($T_{tech}$)', fontsize=12)
plt.ylabel('Market Allocation ($T_{mkt}$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{out_dir}/fig2_pareto_front.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Figure 3: Predictive Feature Stability (Elastic Net)
# ---------------------------------------------------------
features = ['Deeptech Resilience\n(Base Trait)', 
            'Lab-to-MVP Efficacy\n(Base Trait)', 
            'Efficacy × Exposure\n(Interaction)']
scores = [1.0, 100.0, 100.0]

plt.figure(figsize=(10, 5))
ax = sns.barplot(x=scores, y=features, hue=features, palette=['#cccccc', '#4c72b0', '#55a868'], legend=False)
plt.title('Predictive Feature Stability (100 Bootstraps)', fontsize=14, pad=15)
plt.xlabel('Selection Frequency (%)', fontsize=12)
plt.xlim(0, 110)

for i, v in enumerate(scores):
    ax.text(v + 1.5, i, f"{v}%", va='center', fontweight='bold', color='black')
    
plt.savefig(f'{out_dir}/fig3_feature_stability.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Generated 3 figures in {out_dir}/")