"""
main.py
========
Orchestrator for the Urban Traffic Prediction Project
Runs the full pipeline: data cleaning, merging, EDA, and modeling
"""

import subprocess
import sys
import os
from datetime import datetime

# -------------------------------
# Helper function to run scripts
# -------------------------------
def run_script(script_path):
    print("\n" + "="*80)
    print(f"Running: {script_path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        print(f"Completed: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

# -------------------------------
# Ensure output directories exist
# -------------------------------
output_dirs = [
    "data/processed",
    "models/saved_models/plots",
    "models/saved_models/reports",
    "models/saved_models/models"
]

for d in output_dirs:
    os.makedirs(d, exist_ok=True)

# -------------------------------
# Sequentially run the scripts
# -------------------------------
script_sequence = [
    "scripts/01_load_and_clean.py",
    "scripts/02_merge_datasets.py",
    "scripts/03_eda.py",
    "scripts/04_traffic_congestion_modeling.py"
]


for script in script_sequence:
    run_script(script)

print("\n" + "="*80)
print("ALL SCRIPTS EXECUTED SUCCESSFULLY")
print("Pipeline complete. All outputs are saved in their respective folders.")
print("="*80)
