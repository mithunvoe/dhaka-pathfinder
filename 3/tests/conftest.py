"""Make src/ importable so tests can `from pso_wifi_placement import ...`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
