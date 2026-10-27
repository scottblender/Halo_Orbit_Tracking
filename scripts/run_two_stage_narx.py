"""Generate Monte Carlo episodes and fit a two-stage guidance predictor."""
import json
from pathlib import Path
from halo_tracking.study import load_config
from halo_tracking.closed_loop_narx import generate_closed_loop_dataset
from halo_tracking.two_stage_narx import evaluate_two_stage

if __name__ == "__main__":
    cfg = load_config()
    dataset = Path("results/narx_closed_loop")
    print("Generating closed-loop Monte Carlo episodes...", flush=True)
    generate_closed_loop_dataset(cfg, dataset)
    report = evaluate_two_stage(dataset, Path("results/narx_two_stage"), cfg)
    print(json.dumps(report, indent=2))
