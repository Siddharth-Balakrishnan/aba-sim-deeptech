# Deeptech Student-Founder Resource Allocation Simulation Pipeline

A three-part simulation and empirical analysis pipeline supporting research on student-founder incubation within deeptech and scientific venture ecosystems. The framework models the multi-dimensional cognitive, technical, and academic trade-offs encountered by university founders through an Agent-Based Model (ABM), resolves trade-off frontiers via Non-dominated Sorting Genetic Algorithm II (NSGA-II) multi-objective optimization, and quantifies burnout drivers using Elastic Net stability selection.

---

## Table of Contents
1. [Overview & Theoretical Architecture](#overview--theoretical-architecture)
2. [Simulation Pipeline Architecture](#simulation-pipeline-architecture)
3. [Modules, Parameters & Dynamics](#modules-parameters--dynamics)
   - [Module 1: GUESSS Baseline Calibration](#module-1-guesss-baseline-calibration)
   - [Module 2: Agent-Based Incubation Model (Mesa)](#module-2-agent-based-incubation-model-mesa)
   - [Module 3: NSGA-II Multi-Objective Optimization & Elastic Net ML](#module-3-nsga-ii-multi-objective-optimization--elastic-net-ml)
   - [Module 4: Results & Visualizations Generator](#module-4-results--visualizations-generator)
4. [Generated Figures & Tables Catalog](#generated-figures--tables-catalog)
5. [Repository Structure](#repository-structure)
6. [Prerequisites & System Specifications](#prerequisites--system-specifications)
7. [Step-by-Step Execution Guide](#step-by-step-execution-guide)

---

## Overview & Theoretical Architecture

University deeptech spinouts demand simultaneous commitments across three competing domains:
1. **Technical R&D ($T_{\mathrm{tech}}$):** High uncertainty, lengthy laboratory development cycles, and hardware/computational prototyping.
2. **Market Validation ($T_{\mathrm{mkt}}$):** Customer discovery, regulatory clearance, and commercial positioning.
3. **Academic Standing ($A_{\mathrm{score}}$):** Degree progress, coursework, publications, and defense requirements.

Because founders possess finite cognitive capacity (modeled as a weekly cognitive budget of 100 bandwidth points), aggressive technical and commercial devotion inevitably draws down academic bandwidth. If academic standing decays below institutional survival thresholds ($A_{\mathrm{score}} \le 40$) or resilience is fully depleted, the student founder experiences **burnout**, terminating project execution.

This codebase provides an end-to-end computational pipeline that:
- Calibrates 300 synthetic student-founder profiles against global **GUESSS** (Global University Entrepreneurial Spirit Students' Survey) empirical distributions for STEM cohorts.
- Runs a 52-week longitudinal agent simulation capturing stochastic technological shocks and burnout cascades.
- Applies **NSGA-II** Pareto optimization to compute non-dominated weekly allocation policies ($T_{\mathrm{tech}}^*, T_{\mathrm{mkt}}^*, T_{\mathrm{acad}}^*$).
- Employs **Elastic Net stability selection** across 100 bootstrapped iterations to identify critical non-linear predictors of survival and burnout.
- Exports publication-quality vector plots, heatmaps, phase portraits (300 DPI), and structured statistical tables.

---

## Simulation Pipeline Architecture

```mermaid
flowchart TD
    A["GUESSS Survey STEM Baselines\n(Means, Covariances)"] -->|Multivariate Normal Sampling| B["Module 1: guess_baseline_calibration\ncalibrated_agents_baseline.csv (N=300)"]
    B -->|Agent Initialization| C["Module 2: abm_simulation (Mesa)\n52-Week Cycle with Cognitive Allocation & Shocks"]
    C -->|Longitudinal Tracking| D["abm_longitudinal_data.csv\n(Trajectories, Burnout States)"]
    D --> E["Module 3: optimization_and_ml\nanalyze.py"]
    E -->|Elastic Net Stability (100 Bootstraps)| F["Feature Stability Ranks\n(Efficacy x Exposure Interaction)"]
    E -->|NSGA-II Pareto Optimization| G["pareto_optimal_allocations.csv\n(Optimal Trade-off Policies)"]
    D & F & G --> H["Module 4: result_generator\nrun_all.py"]
    H --> I["results/graphs/\n6 Publication Figures (300 DPI)"]
    H --> J["results/tables/\n4 Structured CSV Data Tables"]
```

---

## Modules, Parameters & Dynamics

### Module 1: GUESSS Baseline Calibration
- **Script:** [`guess_baseline_calibration/generate_baselines.py`](file:///home/s_b_yrrah/projects/Final/aba-sim-deeptech/guess_baseline_calibration/generate_baselines.py)
- **Output:** `guess_baseline_calibration/outputs/calibrated_agents_baseline.csv`
- **Cohort Size ($N$):** 300 student founders.

#### Empirical Base Traits (Likert Scale: 1.0 to 7.0)
Calibrated from empirical STEM student distributions (Computer Science & Engineering cohorts):
- **General Entrepreneurial Intention ($GEI$):** $\mu = 3.2$, $\sigma = 1.4$
- **Locus of Control ($LOC$):** $\mu = 4.8$, $\sigma = 1.1$
- **Institutional Support ($IS$):** $\mu = 3.5$, $\sigma = 1.2$
- **Correlation Structure:** Correlated sampling via a covariance matrix $\Sigma_{ij} = \rho \cdot \sigma_i \sigma_j$ with $\rho = 0.35$ using `numpy.random.multivariate_normal`.

#### Derived Deeptech Metrics
1. **Deeptech Resilience ($R$):**
   $$R = \text{clip}\left(0.5 \cdot LOC + 0.5 \cdot IS + \mathcal{N}(0, 0.3^2),\, 1.0,\, 7.0\right)$$
2. **Lab-to-MVP Efficacy ($E$):**
   Reflects the translation discount of complex lab technology to commercially viable prototypes:
   $$E = \text{clip}\left(GEI - 0.8,\, 1.0,\, 7.0\right)$$
3. **Weekly Cognitive Bandwidth ($CB$):** Static baseline of 100 points/week per agent.

---

### Module 2: Agent-Based Incubation Model (Mesa)
- **Script:** [`abm_simulation/model.py`](file:///home/s_b_yrrah/projects/Final/aba-sim-deeptech/abm_simulation/model.py)
- **Output:** `abm_simulation/outputs/abm_longitudinal_data.csv`
- **Timeline:** 52 discrete simulation steps (1 step = 1 calendar week).

#### Agent State Variables
- Initial Academic Score: $A_0 = 100.0$
- Initial Technical Readiness: $T_{\mathrm{tech}, 0} = 0.0$
- Initial Market Readiness: $T_{\mathrm{mkt}, 0} = 0.0$
- Burnout Flag: $\text{is\_burned\_out} = \text{False}$

#### Weekly Step Rules (if not burned out)
1. **Cognitive Bandwidth Allocation:**
   - Efficacy Fraction: $\phi = \frac{E}{7.0} \in [0.14, 1.0]$
   - Technical Allocation: $T_{\mathrm{points}} = 100 \times \phi$
   - Market Allocation: $M_{\mathrm{points}} = 100 \times (1 - \phi) \times 0.5$
   - Academic Allocation: $A_{\mathrm{points}} = 100 - T_{\mathrm{points}} - M_{\mathrm{points}}$
2. **Accumulation & Academic Drain:**
   - $T_{\mathrm{tech}} \leftarrow T_{\mathrm{tech}} + T_{\mathrm{points}}$
   - $T_{\mathrm{mkt}} \leftarrow T_{\mathrm{mkt}} + M_{\mathrm{points}}$
   - Academic score is credited with $A_{\mathrm{points}}$ and subjected to flat weekly academic friction:
     $$A_{\mathrm{score}} \leftarrow A_{\mathrm{score}} + A_{\mathrm{points}} - 30.0$$
3. **Stochastic Vulnerability Shocks:**
   - 10% weekly probability of Technical R&D shock: $R \leftarrow R - 0.5$
   - 10% weekly probability of Market shock: $R \leftarrow R - 0.5$
4. **Burnout Terminal Condition:**
   $$\text{If } (R \le 0) \lor (A_{\mathrm{score}} \le 40.0) \implies \text{is\_burned\_out} \leftarrow \text{True}$$
   Once burned out, cognitive execution ceases and readiness accumulation halts.

---

### Module 3: NSGA-II Multi-Objective Optimization & Elastic Net ML
- **Script:** [`optimization_and_ml/analyze.py`](file:///home/s_b_yrrah/projects/Final/aba-sim-deeptech/optimization_and_ml/analyze.py)
- **Output:** `optimization_and_ml/outputs/pareto_optimal_allocations.csv`

#### Part A: Data Consolidation & Interaction Modelling
Extracts the final state per agent (at week of burnout or week 52 completion) and constructs the non-linear interaction term representing high efficacy under extreme failure loads:
$$\text{Efficacy} \times \text{Exposure} = E \times \left(\frac{1}{\max(R, 0.01)}\right)$$

#### Part B: Elastic Net Stability Selection
- Fits `sklearn.linear_model.ElasticNet` ($\alpha = 0.1, \rho_{\mathrm{L1}} = 0.5$) across 100 bootstrap resamples (80% sub-samples without replacement).
- Selection condition: absolute standardized coefficient $|\beta| > 0.01$.
- Demonstrates that `lab_to_mvp_efficacy` and the interaction term `efficacy_x_exposure` are stably selected in 100% of iterations, while raw baseline resilience is absorbed by the interaction term.

#### Part C: NSGA-II Bi-Objective Formulation
Solves the strategic trade-off between R&D velocity and commercial validation under academic preservation constraints:
- **Decision Variables:**
  $$X = [T_{\mathrm{tech}},\, T_{\mathrm{mkt}},\, T_{\mathrm{acad}}] \in [0.0, 1.0]^3$$
- **Objective Functions (Minimized):**
  - $f_1 = -T_{\mathrm{tech}}$ (Maximize Technical R&D)
  - $f_2 = -T_{\mathrm{mkt}}$ (Maximize Market Validation)
- **Inequality Constraints ($g_i \le 0$):**
  - $g_1: T_{\mathrm{tech}} + T_{\mathrm{mkt}} + T_{\mathrm{acad}} - 1.0 \le 0$ (Total bandwidth budget cap $\le 100\%$)
  - $g_2: 0.40 - T_{\mathrm{acad}} \le 0$ (Minimum 40% academic threshold required to avoid academic drain burnout)
- **Solver Configuration:** Population size = 100, Generations = 50, Selection = Tournament, Crossover = SBX, Mutation = Polynomial.

---

### Module 4: Results & Visualizations Generator
- **Location:** [`result_generator/`](file:///home/s_b_yrrah/projects/Final/aba-sim-deeptech/result_generator)
- Consolidates all visualization and reporting into a unified execution framework.
- Generates all 6 publication figures and 4 CSV tables into clean subfolders.

---

## Generated Figures & Tables Catalog

### Figures (`result_generator/results/graphs/`)
All figures are produced at 300 DPI print resolution:

| Figure | Filename | Description |
| :--- | :--- | :--- |
| **Fig 1** | `fig1_burnout_trajectory.png` | Cumulative agent burnout curve over the 52-week incubation cycle. |
| **Fig 2** | `fig2_pareto_front.png` | Non-dominated Pareto frontier showing the trade-off between $T_{\mathrm{tech}}$ and $T_{\mathrm{mkt}}$, colored by $T_{\mathrm{acad}}$. |
| **Fig 3** | `fig3_feature_stability.png` | Bootstrap selection frequencies proving stability of the interaction term. |
| **Fig 4** | `fig4_3d_burnout_surface.png` | 3D response surface of burnout probability as a function of Efficacy ($E$) and Exposure ($1/R$). |
| **Fig 5** | `fig5_sensitivity_heatmap.png` | Diverging heatmap illustrating shifts in optimal allocation policies under external shock scenarios. |
| **Fig 6** | `fig6_phase_space_flow.png` | Phase-portrait streamplot displaying vector flows $(\dot{T}, \dot{A})$, the $A=40$ burnout sink, and overlaid empirical agent paths. |

### Tables (`result_generator/results/tables/`)
Structured CSV tables suitable for research manuscript publication:

| Table | Filename | Contents |
| :--- | :--- | :--- |
| **Table I** | `table_1_descriptive_stats.csv` | Summary statistics (Mean, SD, Median, Min, Max) for all baseline GUESSS metrics ($N=300$). |
| **Table II** | `table_2_abm_survival_milestones.csv` | Longitudinal milestone metrics across quarters Q1 (W13), Q2 (W25), Q3 (W39), and Q4 (W51). |
| **Table III** | `table_3_nsga2_pareto_summary.csv` | Distribution statistics of Pareto-optimal time allocations ($T_{\mathrm{tech}}, T_{\mathrm{mkt}}, T_{\mathrm{acad}}$). |
| **Table IV** | `table_4_elastic_net_stability.csv` | Stability selection ranks, non-zero frequencies (%), and mean coefficients across 100 bootstraps. |

---

## Repository Structure

```
aba-sim-deeptech/
├── .gitignore                          # Ignores venvs, cache, and all output directories
├── requirements.txt                    # Unified dependencies for the entire repository
├── README.md                           # Master project documentation
│
├── guess_baseline_calibration/         # MODULE 1: GUESSS Baseline Sampling
│   ├── generate_baselines.py           # Calibration script
│   ├── requirements.txt                # Module-level requirements
│   ├── README.md                       # Module documentation
│   └── outputs/                        # Generated baseline CSVs (git-ignored)
│
├── abm_simulation/                     # MODULE 2: Mesa Agent-Based Simulation
│   ├── model.py                        # ABM execution script
│   ├── requirements.txt                # Module-level requirements
│   └── outputs/                        # Longitudinal ABM data (git-ignored)
│
├── optimization_and_ml/                # MODULE 3: NSGA-II Optimization & Elastic Net
│   ├── analyze.py                      # Multi-objective optimization & ML script
│   ├── requirements.txt                # Module-level requirements
│   └── outputs/                        # Pareto front solutions (git-ignored)
│
└── result_generator/                  # MODULE 4: Unified Results & Export Engine
    ├── generate_graphs.py              # Single script generating all 6 figures (300 DPI)
    ├── generate_paper_tables.py        # Script generating all 4 CSV tables
    ├── run_all.py                      # Master execution script (Graphs + Tables)
    ├── README.md                       # Results generator documentation
    └── results/                        # Export directory (git-ignored)
        ├── graphs/                     # High-res publication plots (fig1 - fig6)
        └── tables/                     # Publication tables (table_1 - table_4)
```

---

## Prerequisites & System Specifications

- **Operating System:** Linux (Ubuntu 22.04 / 24.04 / 25.10 or compatible POSIX environment)
- **Python Version:** Python 3.10 to 3.13
- **Hardware Footprint:** Minimal (runs smoothly on dual/quad-core CPUs, < 2GB RAM required)

### Unified Dependencies (`requirements.txt`)
- `numpy>=2.0.0`
- `pandas>=2.2.0`
- `scipy>=1.14.0`
- `mesa>=3.0.0`
- `pymoo>=0.6.0`
- `scikit-learn>=1.5.0`
- `matplotlib>=3.9.0`
- `seaborn>=0.13.0`

---

## Step-by-Step Execution Guide

### Step 1: Clone Repository & Create Virtual Environment
```bash
cd ~/projects/Final/aba-sim-deeptech

# Create isolated Python virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate
```

### Step 2: Install Unified Dependencies
Install all required libraries using the root `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run the Full Simulation Pipeline

#### 3.1 Generate Calibrated Baseline Profiles (Module 1)
Generates 300 synthetic student-founder profiles calibrated against GUESSS empirical baselines:
```bash
python guess_baseline_calibration/generate_baselines.py
```
*Output created:* `guess_baseline_calibration/outputs/calibrated_agents_baseline.csv`

#### 3.2 Execute the 52-Week Agent-Based Model (Module 2)
Simulates weekly bandwidth allocations, stochastic shocks, and burnout states across 52 weeks:
```bash
python abm_simulation/model.py
```
*Output created:* `abm_simulation/outputs/abm_longitudinal_data.csv`

#### 3.3 Run Elastic Net ML & NSGA-II Optimization (Module 3)
Calculates stability selection on burnout predictors and discovers the Pareto-optimal allocation frontier:
```bash
python optimization_and_ml/analyze.py
```
*Output created:* `optimization_and_ml/outputs/pareto_optimal_allocations.csv`

#### 3.4 Generate All Publication Figures & Tables (Module 4)
Executes both table computation and graph generation in a single automated step:
```bash
python result_generator/run_all.py
```
*Outputs created:*
- Figures: `result_generator/results/graphs/fig1_burnout_trajectory.png` through `fig6_phase_space_flow.png`
- Tables: `result_generator/results/tables/table_1_descriptive_stats.csv` through `table_4_elastic_net_stability.csv`

*(Optional)* Run graph and table generators independently:
```bash
# Generate figures only
python result_generator/generate_graphs.py

# Generate tables only
python result_generator/generate_paper_tables.py
```
