"""Reproducible observer-only year-long STM vs learned NARX comparison.

From the checkout: python scripts/run_one_year_study.py --days 365 --arcs 5

Training arcs are separate from the rollouts; Gateway stays on a closure-
validated prescribed periodic CR3BP trajectory. Output files in results/.
"""
import argparse
import json
from pathlib import Path

from halo_tracking.study import load_config
from halo_tracking.closed_loop_narx import generate_closed_loop_dataset
from halo_tracking.two_stage_narx import evaluate_two_stage, fit_observer_policy
from halo_tracking.year_study import run_paired_study
from halo_tracking.neural_narx import fit_neural_policy
from halo_tracking.narx_audit import audit_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=float, default=365.)
    p.add_argument("--arcs", type=int, default=5)
    p.add_argument("--seed", type=int, default=24017)
    p.add_argument("--measurement-cadence-s", type=float, default=21600.)
    p.add_argument("--decisions-per-period", type=int, default=8)
    p.add_argument("--training-arcs", type=int, default=20)
    p.add_argument("--reuse-training", action="store_true",
                   help="Reuse preexisting short-arc NPZ files rather than regenerate them")
    p.add_argument("--output", type=Path, default=Path("results/one_year"))
    args = p.parse_args()

    cfg = load_config()
    cfg["narx"]["training_arcs"] = args.training_arcs
    cfg["narx"]["decisions_per_period"] = args.decisions_per_period
    cfg["narx"]["closed_loop_measurement_cadence_s"] = args.measurement_cadence_s
    dataset = args.output / "teacher_training"
    if not args.reuse_training or not all((dataset / (split + ".npz")).exists()
                                        for split in ("training", "validation", "test")):
        print("Generating independent short-arc STM teaching data...", flush=True)
        generate_closed_loop_dataset(cfg, dataset)
    audit_dataset(dataset, cfg)
    neural, _ = fit_neural_policy(dataset, cfg, args.output / "neural")
    print("Training observer-only predictors...", flush=True)
    model = fit_observer_policy(dataset, cfg, args.output / "observer_narx.joblib")
    offline = evaluate_two_stage(dataset, args.output / "offline", cfg)
    print("Offline model test:", json.dumps(offline["observer"].get("test")), flush=True)
    results = run_paired_study(
        cfg, args.output, model, days=args.days, arcs=args.arcs,
        seed=args.seed, measurement_cadence_s=args.measurement_cadence_s,
        decisions_per_period=args.decisions_per_period, neural_bundle=neural)
    print("One-year comparison:", json.dumps(results["aggregate"], indent=2), flush=True)
    print("Saved:", args.output / "summary.json", args.output / "arc_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
