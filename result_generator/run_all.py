"""Master Runner to Generate All Paper Results (Graphs & Tables)"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from generate_graphs import main as run_graphs
from generate_paper_tables import main as run_tables

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("STEP 1: GENERATING PUBLICATION TABLES")
    print("#" * 70)
    run_tables()

    print("\n" + "#" * 70)
    print("STEP 2: GENERATING PUBLICATION GRAPHS")
    print("#" * 70)
    run_graphs()

    print("\n" + "=" * 70)
    print("🎉 ALL RESEARCH PAPER RESULTS SUCCESSFULLY GENERATED!")
    print(f"Graphs: {os.path.join(SCRIPT_DIR, 'results', 'graphs')}")
    print(f"Tables: {os.path.join(SCRIPT_DIR, 'results', 'tables')}")
    print("=" * 70 + "\n")

