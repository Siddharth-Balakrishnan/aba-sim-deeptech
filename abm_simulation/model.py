"""Mesa ABM simulation for deeptech student‑founder agents.

- Loads baseline traits from `guess_baseline_calibration/outputs/calibrated_agents_baseline.csv`.
- Creates a `StudentFounderAgent` for each row (300 agents).
- Runs a 52‑step weekly simulation.
- Saves longitudinal data to `abm_simulation/outputs/abm_longitudinal_data.csv`.
"""

import random
import numpy as np
import pandas as pd
from mesa import Model, Agent
from mesa.datacollection import DataCollector

# Simple scheduler to replace missing Mesa activation classes
class SimpleScheduler:
    """A minimal scheduler that stores agents and steps them sequentially."""

    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in self.agents:
            agent.step()


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
class StudentFounderAgent(Agent):
    """Agent representing a synthetic student‑founder.

    Parameters
    ----------
    unique_id: int
        Mesa identifier.
    model: Model
        Reference to the containing model.
    traits: dict
        Baseline traits loaded from the CSV.
    """

    def __init__(self, model, traits, agent_id):
        super().__init__(model)
        self.agent_id = agent_id
        # Baseline traits
        self.deeptech_resilience = traits["deeptech_resilience"]
        self.lab_to_mvp_efficacy = traits["lab_to_mvp_efficacy"]
        self.cognitive_bandwidth = traits["cognitive_bandwidth"]
        # State variables
        self.academic_score = 100.0
        self.tech_readiness = 0.0
        self.market_readiness = 0.0
        self.is_burned_out = False

    def step(self):
        if self.is_burned_out:
            return
        # ---------------------------------------------------------------
        # 1️⃣ Allocate cognitive bandwidth
        # ---------------------------------------------------------------
        efficacy_frac = max(0.0, min(1.0, self.lab_to_mvp_efficacy / 7.0))
        tech_points = self.cognitive_bandwidth * efficacy_frac
        market_points = self.cognitive_bandwidth * (1 - efficacy_frac) * 0.5
        acad_points = self.cognitive_bandwidth - tech_points - market_points
        self.tech_readiness += tech_points
        self.market_readiness += market_points
        self.academic_score += acad_points
        # ---------------------------------------------------------------
        # 2️⃣ Academic drain
        # ---------------------------------------------------------------
        self.academic_score -= 30.0
        # ---------------------------------------------------------------
        # 3️⃣ Stochastic shocks to deep‑tech resilience
        # ---------------------------------------------------------------
        if random.random() < 0.10:
            self.deeptech_resilience -= 0.5
        if random.random() < 0.10:
            self.deeptech_resilience -= 0.5
        # ---------------------------------------------------------------
        # 4️⃣ Burnout check
        # ---------------------------------------------------------------
        if self.deeptech_resilience <= 0 or self.academic_score <= 40:
            self.is_burned_out = True

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class DeeptechIncubationModel(Model):
    """Mesa model that runs the ABM for 52 weekly steps.

    The model loads the baseline CSV, creates agents, and records a set of
    metrics each step using :class:`mesa.DataCollector`.
    """

    def __init__(self, csv_path, steps=52):
        super().__init__()
        self.steps = steps
        self.schedule = SimpleScheduler(self)
        self.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={
                "tech_readiness": "tech_readiness",
                "market_readiness": "market_readiness",
                "academic_score": "academic_score",
                "deeptech_resilience": "deeptech_resilience",
                "is_burned_out": "is_burned_out",
            },
        )
        # Load baseline traits
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            traits = {
                "deeptech_resilience": row["deeptech_resilience"],
                "lab_to_mvp_efficacy": row["lab_to_mvp_efficacy"],
                "cognitive_bandwidth": row["cognitive_bandwidth"],
            }
            agent = StudentFounderAgent(model=self, traits=traits, agent_id=int(row["Agent_ID"]))
            self.schedule.add(agent)

    def step(self):
        """Advance the model one step (one week)."""
        self.datacollector.collect(self)
        self.schedule.step()
        self.time += 1

    def run_model(self):
        for _ in range(self.steps):
            self.step()

# ---------------------------------------------------------------------------
# Execution block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    baseline_path = "guess_baseline_calibration/outputs/calibrated_agents_baseline.csv"
    model = DeeptechIncubationModel(csv_path=baseline_path, steps=52)
    model.run_model()

    # Retrieve collected data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    df_long = agent_data.reset_index()
    df_long = df_long.rename(columns={"Step": "Week", "AgentID": "Agent_ID"})

    # Burnout statistics
    burnout_counts = (
        df_long[df_long["is_burned_out"]]
        .groupby("Agent_ID")
        .size()
        .gt(0)
        .sum()
    )

    output_path = "abm_simulation/outputs/abm_longitudinal_data.csv"
    df_long.to_csv(output_path, index=False)

    print(f"✅ Simulation complete. Saved longitudinal data to {output_path}")
    print(f"Data shape: {df_long.shape}")
    print(f"Number of agents that burned out at any point: {burnout_counts}")
