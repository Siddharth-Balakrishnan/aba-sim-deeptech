"""Wrapper to run Figure 6 generation from repository root."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimization_and_ml.generate_graph_6 import generate_fig6

if __name__ == "__main__":
    generate_fig6()

