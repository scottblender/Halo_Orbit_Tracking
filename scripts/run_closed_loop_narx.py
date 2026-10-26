"""Generate applied-control NARX training, validation and test arcs.

Run from the repository root after installing the package:
    python scripts/run_closed_loop_narx.py
"""
from pathlib import Path
from halo_tracking.study import load_config
from halo_tracking.closed_loop_narx import generate_closed_loop_dataset

if __name__ == "__main__":
    destination = Path("results/narx_closed_loop")
    summary = generate_closed_loop_dataset(load_config(), destination)
    for split, values in summary.items():
        print(split, values)
    print("Saved simulated closed-loop histories to", destination)
