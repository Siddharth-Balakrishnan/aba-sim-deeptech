"""Table Generator for Deeptech Simulation Research Paper

Generates the 4 core CSV data tables and saves them to:
  result_generator/results/tables/

Tables:
  1. table_1_descriptive_stats.csv       - Baseline GUESSS metrics (Mean, SD, Median, Min, Max)
  2. table_2_abm_survival_milestones.csv - Quarterly ABM milestone tracking
  3. table_3_nsga2_pareto_summary.csv    - NSGA-II Pareto-optimal allocations summary
  4. table_4_elastic_net_stability.csv   - Elastic Net stability selection frequencies & ranks
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Table 1: Descriptive Statistics of Baseline Metrics (N=300)
# ---------------------------------------------------------------------------
def generate_table_1():
    baseline_path = os.path.join(ROOT_DIR, "guess_baseline_calibration", "outputs", "calibrated_agents_baseline.csv")
    df_base = pd.read_csv(baseline_path)

    metrics_map = {
        "General_Entrepreneurial_Intention": "General Entrepreneurial Intention",
        "Locus_of_Control": "Locus of Control",
        "Institutional_Support": "Institutional Support / Environment",
        "deeptech_resilience": "Deeptech Resilience",
        "lab_to_mvp_efficacy": "Lab-to-MVP Efficacy",
    }

    records = []
    for col, display_name in metrics_map.items():
        series = df_base[col]
        records.append({
            "Metric": display_name,
            "N": len(series),
            "Mean": round(series.mean(), 3),
            "Std_Dev": round(series.std(), 3),
            "Median": round(series.median(), 3),
            "Min": round(series.min(), 3),
            "Max": round(series.max(), 3),
        })

    table_1 = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "table_1_descriptive_stats.csv")
    table_1.to_csv(out_path, index=False)
    print(f"  ✓ Saved Table 1: {out_path}")
    print(table_1.to_string(index=False))
    print()
    return table_1


# ---------------------------------------------------------------------------
# Table 2: Longitudinal ABM Survival Milestones (Quarterly Tracking)
# ---------------------------------------------------------------------------
def generate_table_2():
    abm_path = os.path.join(ROOT_DIR, "abm_simulation", "outputs", "abm_longitudinal_data.csv")
    df_abm = pd.read_csv(abm_path)

    quarters = [
        {"Quarter": "Q1", "Week": 13.0, "Description": "Week 13 (Foundation)"},
        {"Quarter": "Q2", "Week": 25.0, "Description": "Week 25 (Mid-Year Review)"},
        {"Quarter": "Q3", "Week": 39.0, "Description": "Week 39 (MVP Maturation)"},
        {"Quarter": "Q4", "Week": 51.0, "Description": "Week 51 (Final Incubation)"},
    ]

    total_agents = df_abm["Agent_ID"].nunique()
    records = []

    for q in quarters:
        w = q["Week"]
        sub = df_abm[df_abm["Week"] == w]
        burned_count = int(sub["is_burned_out"].sum())
        active_count = total_agents - burned_count
        survival_rate = round((active_count / total_agents) * 100.0, 1)

        records.append({
            "Quarter": q["Quarter"],
            "Simulation_Week": int(w),
            "Milestone_Stage": q["Description"],
            "Active_Agents": active_count,
            "Cumulative_Burnout": burned_count,
            "Survival_Rate_Pct": f"{survival_rate}%",
            "Mean_Tech_Readiness": round(sub["tech_readiness"].mean(), 2),
            "Mean_Market_Validation": round(sub["market_readiness"].mean(), 2),
            "Mean_Academic_Score": round(sub["academic_score"].mean(), 2),
        })

    table_2 = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "table_2_abm_survival_milestones.csv")
    table_2.to_csv(out_path, index=False)
    print(f"  ✓ Saved Table 2: {out_path}")
    print(table_2.to_string(index=False))
    print()
    return table_2


# ---------------------------------------------------------------------------
# Table 3: NSGA-II Pareto Front Summary
# ---------------------------------------------------------------------------
def generate_table_3():
    pareto_path = os.path.join(ROOT_DIR, "optimization_and_ml", "outputs", "pareto_optimal_allocations.csv")
    df_pareto = pd.read_csv(pareto_path)

    dimensions = [
        {"Dimension": "Technical R&D Allocation", "Symbol": "T_tech", "Col": "T_tech"},
        {"Dimension": "Market Validation Allocation", "Symbol": "T_mkt", "Col": "T_mkt"},
        {"Dimension": "Academic Standing Allocation", "Symbol": "T_acad", "Col": "T_acad"},
    ]

    records = []
    for d in dimensions:
        col = df_pareto[d["Col"]]
        records.append({
            "Allocation_Dimension": d["Dimension"],
            "Symbol": d["Symbol"],
            "Min": round(col.min(), 4),
            "Mean": round(col.mean(), 4),
            "Median": round(col.median(), 4),
            "Max": round(col.max(), 4),
            "Std_Dev": round(col.std(), 4),
        })

    table_3 = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "table_3_nsga2_pareto_summary.csv")
    table_3.to_csv(out_path, index=False)
    print(f"  ✓ Saved Table 3: {out_path}")
    print(table_3.to_string(index=False))
    print()
    return table_3


# ---------------------------------------------------------------------------
# Table 4: Elastic Net Stability Selection Metrics
# ---------------------------------------------------------------------------
def generate_table_4():
    abm_path = os.path.join(ROOT_DIR, "abm_simulation", "outputs", "abm_longitudinal_data.csv")
    baseline_path = os.path.join(ROOT_DIR, "guess_baseline_calibration", "outputs", "calibrated_agents_baseline.csv")

    df_abm = pd.read_csv(abm_path)
    df_baseline = pd.read_csv(baseline_path)

    def get_final(group):
        b = group[group["is_burned_out"]]
        if not b.empty:
            return b.iloc[0]
        return group.iloc[-1]

    df_final = (
        df_abm.groupby("Agent_ID", as_index=False, group_keys=False)
        .apply(get_final)
        .reset_index(drop=True)
    )
    df_final = df_final.merge(df_baseline[["Agent_ID", "lab_to_mvp_efficacy"]], on="Agent_ID")

    df_final["efficacy_x_exposure"] = (
        df_final["lab_to_mvp_efficacy"]
        * (1.0 / df_final["deeptech_resilience"].clip(lower=0.01))
    )

    features = [
        ("lab_to_mvp_efficacy", "Lab-to-MVP Efficacy", "Base Trait"),
        ("efficacy_x_exposure", "Efficacy × Exposure (1/Resilience)", "Interaction Term"),
        ("deeptech_resilience", "Deeptech Resilience", "Base Trait"),
    ]

    feature_cols = [f[0] for f in features]
    X = df_final[feature_cols].values
    y = df_final["is_burned_out"].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rng = np.random.RandomState(42)
    n_bootstrap = 100
    coef_matrix = np.zeros((n_bootstrap, len(features)))
    selected_matrix = np.zeros((n_bootstrap, len(features)), dtype=bool)

    for i in range(n_bootstrap):
        idx = rng.choice(len(X_scaled), size=int(0.8 * len(X_scaled)), replace=False)
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
        model.fit(X_scaled[idx], y[idx])
        coef_matrix[i, :] = model.coef_
        selected_matrix[i, :] = model.coef_ > 0.01

    selection_pct = selected_matrix.mean(axis=0) * 100.0
    mean_coef = coef_matrix.mean(axis=0)

    records = []
    for idx, (col, display_name, ftype) in enumerate(features):
        records.append({
            "Feature": display_name,
            "Type": ftype,
            "Selection_Frequency_Pct": f"{round(selection_pct[idx], 1)}%",
            "Mean_Coefficient": round(mean_coef[idx], 4),
            "_sort_key": (selection_pct[idx], mean_coef[idx]),
        })

    table_4 = pd.DataFrame(records)
    table_4 = table_4.sort_values(by="_sort_key", ascending=False).reset_index(drop=True)
    table_4.insert(0, "Stability_Rank", range(1, len(table_4) + 1))
    table_4 = table_4.drop(columns=["_sort_key"])

    out_path = os.path.join(OUTPUT_DIR, "table_4_elastic_net_stability.csv")
    table_4.to_csv(out_path, index=False)
    print(f"  ✓ Saved Table 4: {out_path}")
    print(table_4.to_string(index=False))
    print()
    return table_4


def main():
    print("=" * 65)
    print("Generating All Publication Tables (Table 1 – Table 4)")
    print(f"Output Directory: {OUTPUT_DIR}/")
    print("=" * 65)
    generate_table_1()
    generate_table_2()
    generate_table_3()
    generate_table_4()
    print("\n✅ All 4 publication tables generated successfully.")


if __name__ == "__main__":
    main()

