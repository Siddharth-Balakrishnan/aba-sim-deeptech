import numpy as np
import pandas as pd

import os

def main():
    # Resolve paths relative to this script's directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Number of synthetic agents
    N = 300

    # Base GUESSS metrics: means and standard deviations (Likert 1-7)
    means = np.array([3.2, 4.8, 3.5])  # [General Entrepreneurial Intention, Locus of Control, Institutional Support]
    sds = np.array([1.4, 1.1, 1.2])

    # Correlation matrix (moderate positive correlation ~0.35)
    corr = 0.35
    corr_matrix = np.full((3, 3), corr)
    np.fill_diagonal(corr_matrix, 1.0)

    # Covariance matrix = corr * sd_i * sd_j
    cov_matrix = np.outer(sds, sds) * corr_matrix

    # Generate correlated baseline data
    baseline_data = np.random.multivariate_normal(mean=means, cov=cov_matrix, size=N)
    df = pd.DataFrame(baseline_data, columns=[
        "General_Entrepreneurial_Intention",
        "Locus_of_Control",
        "Institutional_Support"
    ])

    # Deeptech-specific metrics
    # deeptech_resilience: weighted sum of Locus and Institutional Support + Gaussian noise
    df["deeptech_resilience"] = (
        0.5 * df["Locus_of_Control"] +
        0.5 * df["Institutional_Support"] +
        np.random.normal(loc=0.0, scale=0.3, size=N)
    )

    # lab_to_mvp_efficacy: subtract complexity penalty, floor at 1.0
    penalty = 0.8
    df["lab_to_mvp_efficacy"] = df["General_Entrepreneurial_Intention"] - penalty
    df["lab_to_mvp_efficacy"] = df["lab_to_mvp_efficacy"].clip(lower=1.0)

    # cognitive_bandwidth: static baseline
    df["cognitive_bandwidth"] = 100

    # Clamp Likert-like values to the 1-7 range (for the three base traits and derived ones)
    likert_cols = [
        "General_Entrepreneurial_Intention",
        "Locus_of_Control",
        "Institutional_Support",
        "deeptech_resilience",
        "lab_to_mvp_efficacy",
    ]
    df[likert_cols] = df[likert_cols].clip(lower=1.0, upper=7.0)

    # Add Agent_ID
    df.insert(0, "Agent_ID", range(1, N + 1))

    # Output path
    output_path = os.path.join(SCRIPT_DIR, "outputs", "calibrated_agents_baseline.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Generated {N} calibrated agent profiles.")
    print(f"Saved to {output_path}")
    print("First few rows:")
    print(df.head())
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()

