# guess_baseline_calibration

## Module Overview

This module calibrates a synthetic population of student‑founder agents against the **GUESSS** (Graduate Undergraduate Entrepreneurial Survey of Student‑founders) baseline statistics for Computer Science and Engineering cohorts.

### Setup

```bash
# From the repository root
python3 -m venv venv   # Create a virtual environment
source venv/bin/activate
pip install -r guess_baseline_calibration/requirements.txt
```

### Running the Calibration

```bash
python guess_baseline_calibration/generate_baselines.py
```

Running the script will generate **300** synthetic agent profiles and write them to `guess_baseline_calibration/outputs/calibrated_agents_baseline.csv`.

