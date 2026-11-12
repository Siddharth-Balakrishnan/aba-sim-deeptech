# Result Generator

Consolidated generation module for all research paper publication figures and data tables.

## Directory Structure

```
result_generator/
├── README.md
├── generate_graphs.py         # Generates all 6 publication figures (300 DPI)
├── generate_paper_tables.py   # Generates all 4 publication data tables (CSV)
├── run_all.py                 # Master runner executing both generators
└── results/
    ├── graphs/
    │   ├── fig1_burnout_trajectory.png
    │   ├── fig2_pareto_front.png
    │   ├── fig3_feature_stability.png
    │   ├── fig4_3d_burnout_surface.png
    │   ├── fig5_sensitivity_heatmap.png
    │   └── fig6_phase_space_flow.png
    └── tables/
        ├── table_1_descriptive_stats.csv
        ├── table_2_abm_survival_milestones.csv
        ├── table_3_nsga2_pareto_summary.csv
        └── table_4_elastic_net_stability.csv
```

## Usage

From the project root:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate all figures and tables in one run:
python result_generator/run_all.py

# Or run separately:
python result_generator/generate_graphs.py
python result_generator/generate_paper_tables.py
```

