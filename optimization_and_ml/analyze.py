"""Optimization & ML analysis for the deeptech simulation pipeline.

Part A – Data Preparation
Part B – Elastic Net Stability Selection
Part C – NSGA-II Multi-Objective Optimization

Outputs are written to  optimization_and_ml/outputs/
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# ===================================================================
# PART A: Data Preparation
# ===================================================================
def prepare_data():
    """Load ABM longitudinal data and baseline traits, return feature matrix."""
    # Load longitudinal ABM data
    abm_path = os.path.join(ROOT_DIR, "abm_simulation", "outputs",
                            "abm_longitudinal_data.csv")
    df = pd.read_csv(abm_path)

    # Extract each agent's final-step state: week of burnout, or week 52 / last week
    def get_agent_final_state(group):
        burned = group[group["is_burned_out"]]
        if not burned.empty:
            return burned.iloc[0]
        return group.iloc[-1]

    df_final = (
        df.groupby("Agent_ID", as_index=False, group_keys=False)
        .apply(get_agent_final_state)
        .reset_index(drop=True)
    )

    # Load baseline traits to recover lab_to_mvp_efficacy
    baseline_path = os.path.join(ROOT_DIR, "guess_baseline_calibration",
                                 "outputs", "calibrated_agents_baseline.csv")
    df_baseline = pd.read_csv(baseline_path)[["Agent_ID", "lab_to_mvp_efficacy"]]
    df_final = df_final.merge(df_baseline, on="Agent_ID", how="left")

    # Interaction term: high efficacy × high failure exposure
    # Inverse of resilience represents failure-load exposure
    df_final["efficacy_x_exposure"] = (
        df_final["lab_to_mvp_efficacy"]
        * (1.0 / df_final["deeptech_resilience"].clip(lower=0.01))
    )

    print("=" * 60)
    print("PART A: Data Preparation")
    print("=" * 60)
    print(f"Final-state records: {len(df_final)}")
    print(f"Burnout count: {df_final['is_burned_out'].sum()} / {len(df_final)} ({df_final['is_burned_out'].mean():.1%})")
    print(f"Features: ['deeptech_resilience', 'lab_to_mvp_efficacy', 'efficacy_x_exposure']")
    print(f"Target: 'is_burned_out' (binary)\n")

    return df_final


# ===================================================================
# PART B: Elastic Net Stability Selection
# ===================================================================
def elastic_net_stability(df_final, n_bootstrap=100, subsample_frac=0.80):
    """Run Elastic Net on bootstrapped subsets and report feature stability."""
    feature_cols = ["deeptech_resilience", "lab_to_mvp_efficacy",
                    "efficacy_x_exposure"]
    X = df_final[feature_cols].values
    y = df_final["is_burned_out"].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selection_counts = np.zeros(len(feature_cols))
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Draw an 80 % subsample (without replacement)
        n_sub = int(len(X_scaled) * subsample_frac)
        idx = rng.choice(len(X_scaled), size=n_sub, replace=False)
        X_sub, y_sub = X_scaled[idx], y[idx]

        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                           random_state=42)
        model.fit(X_sub, y_sub)

        # A feature is selected if its coefficient > 0.01
        selected = model.coef_ > 0.01
        selection_counts += selected

    # Express as percentages
    freqs = selection_counts / n_bootstrap * 100

    print("=" * 60)
    print("PART B: Elastic Net Stability Selection  (100 bootstraps)")
    print("=" * 60)
    for name, pct in zip(feature_cols, freqs):
        bar = "#" * int(pct // 2)
        print(f"  {name:<25s}  selected {pct:5.1f}%  {bar}")
    print()

    return dict(zip(feature_cols, freqs))


# ===================================================================
# PART C: NSGA-II Multi-Objective Optimization
# ===================================================================
class FounderResourceProblem(Problem):
    """Bi-objective resource-allocation problem for student founders.

    Variables
    ---------
    x0 = T_tech   (fraction of weekly time on technical R&D)
    x1 = T_mkt    (fraction on market validation)
    x2 = T_acad   (fraction on academic work)

    Objectives (both minimized)
    ----------
    f1 = -T_tech   → maximise technical R&D
    f2 = -T_mkt    → maximise market validation

    Constraints (≤ 0 feasible)
    -----------
    g1 = T_tech + T_mkt + T_acad - 1.0   (cannot exceed 100 %)
    g2 = 0.4 - T_acad                     (academic ≥ 40 %)
    """

    def __init__(self):
        super().__init__(
            n_var=3,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.array([0.0, 0.0, 0.0]),
            xu=np.array([1.0, 1.0, 1.0]),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        T_tech = X[:, 0]
        T_mkt = X[:, 1]
        T_acad = X[:, 2]

        # Objectives (minimize negatives ⟹ maximise originals)
        f1 = -T_tech
        f2 = -T_mkt
        out["F"] = np.column_stack([f1, f2])

        # Constraints (≤ 0 is feasible)
        g1 = T_tech + T_mkt + T_acad - 1.0
        g2 = 0.4 - T_acad
        out["G"] = np.column_stack([g1, g2])


def run_nsga2():
    """Run NSGA-II and return the Pareto-optimal allocations."""
    problem = FounderResourceProblem()
    algorithm = NSGA2(pop_size=100)
    termination = get_termination("n_gen", 50)

    res = minimize(problem, algorithm, termination, seed=42, verbose=False)

    # Build a tidy DataFrame of Pareto-optimal solutions
    df_pareto = pd.DataFrame(res.X, columns=["T_tech", "T_mkt", "T_acad"])
    df_pareto["f1_neg_tech"] = res.F[:, 0]
    df_pareto["f2_neg_mkt"] = res.F[:, 1]

    output_path = os.path.join(SCRIPT_DIR, "outputs",
                               "pareto_optimal_allocations.csv")
    df_pareto.to_csv(output_path, index=False)

    print("=" * 60)
    print("PART C: NSGA-II Pareto Front  (pop=100, gen=50)")
    print("=" * 60)
    print(f"Pareto-optimal solutions found: {len(df_pareto)}")
    print(f"Saved to: {output_path}")
    print("\nSample allocations (first 10):")
    print(df_pareto.head(10).to_string(index=False))
    print()

    return df_pareto


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    df_final = prepare_data()
    stability = elastic_net_stability(df_final)
    pareto = run_nsga2()
    print("✅ All analyses complete.")

