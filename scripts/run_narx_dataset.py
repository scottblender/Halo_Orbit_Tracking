"""Generate the phase-aware observer and Gateway datasets.

Usage: python scripts/run_narx_dataset.py
"""
from pathlib import Path

from halo_tracking.study import load_config
from halo_tracking.narx_dataset import generate_phase_aware_dataset


if __name__ == "__main__":
    destination = Path("results") / "narx_phase_aware"
    summary = generate_phase_aware_dataset(load_config(), destination)
    print("Saved phase-aware datasets to", destination)
    for split, counts in summary.items():
        print(split, counts)
