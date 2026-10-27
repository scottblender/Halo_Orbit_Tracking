"""Inspect teacher feature alignment and early 365-day policy termination.

Run after the one-year study:
    python scripts/inspect_narx_rollout.py
No trajectories are silently extrapolated beyond their last simulated epoch.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from halo_tracking.two_stage_narx import _features
from halo_tracking.study import load_config


def dataset_check(path, cfg):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    x, mask = _features(d, "observer")
    lag = int(cfg["narx"]["history_steps"])
    history = d["observer_history_error_km_m_s"]
    action = d["observer_history_applied_delta_v_rotating_m_s"]
    nav = d["observer_navigation_error_km_m_s"]
    applied = d["observer_applied_delta_v_rotating_m_s"]
    errors = []
    for k in range(lag, len(d["decision_times_tu"])):
        errors.append(np.max(np.abs(history[:, k] - nav[:, k-lag+1:k+1])))
        errors.append(np.max(np.abs(action[:, k] - applied[:, k-lag:k])))
    step = np.diff(d["decision_times_tu"]) * cfg["time_unit_s"]
    triggered = d["observer_maneuver_required"].astype(bool)
    valid = d["observer_maneuver_valid"].astype(bool)
    targets = d["observer_target_delta_v_rotating_m_s"]
    return {
        "feature_shape": list(x.shape),
        "usable_feature_rows": int(np.sum(mask & np.isfinite(x).all(axis=1))),
        "decision_step_s_min_max": [float(np.min(step)), float(np.max(step))],
        "lag_alignment_max_absolute_error": float(max(errors)) if errors else None,
        "triggered": int(triggered.sum()),
        "feasible": int((triggered & valid).sum()),
        "rejected": int((triggered & ~valid).sum()),
        "finite_feasible_target_fraction": float(np.mean(
            np.isfinite(targets[triggered & valid]).all(axis=-1)))
            if np.any(triggered & valid) else None,
        "target_action_same_object": False,
        "note": "Target is teacher-requested delta-v; lagged action is previously executed delta-v."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/one_year"))
    parser.add_argument("--days", type=float, default=40.)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = args.results if args.results.is_absolute() else root / args.results
    cfg = load_config()
    directory = results / "teacher_training"
    report = {
        "teacher_decisions_per_period": int(cfg["narx"]["decisions_per_period"]),
        "teacher_optical_cadence_s": float(cfg["narx"].get("closed_loop_measurement_cadence_s", 21600)),
        "teacher": {}
    }
    for split in ("training", "validation", "test"):
        path = directory / f"{split}.npz"
        if path.exists():
            report["teacher"][split] = dataset_check(path, cfg)
    csv_path = results / "arc_metrics.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            report["annual_summary"] = list(csv.DictReader(f))
    out = results / "narx_suitability"
    out.mkdir(exist_ok=True, parents=True)
    for controller in ("uncontrolled", "stm", "narx"):
        p = results / f"{controller}_arc_000.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as d:
            day = d["day"]
            dep = d["observer_truth_position_departure_km"]
            valid = (day <= args.days) & np.isfinite(dep)
            if not np.any(valid):
                continue
            if controller == "narx":
                report["narx_trace"] = {
                    "last_finite_day": float(day[np.flatnonzero(np.isfinite(dep))[-1]]),
                    "model_calls": int(d["model_invoked"].sum()) if "model_invoked" in d else None,
                    "requested": int(d["maneuver_requested"].sum()),
                    "rejected": int(d["maneuver_rejected"].sum()),
                    "executed": int(np.sum(np.linalg.norm(
                        d["applied_delta_v_rotating_m_s"], axis=1) > 1e-12)),
                    "note": "Exact halted_reason is in arc_metrics.csv, not the NPZ trace."
                }
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot(day[valid], dep[valid], label=controller)
            ax[0].axhline(cfg["narx"]["observer_guidance"]["position_tolerance_km"],
                          linestyle="--", label="100 km corridor")
            ax[0].axhline(20000, linestyle=":", label="default cutoff 20000 km")
            ax[0].set_ylabel("Truth position departure (km)")
            ax[0].legend()
            dv = np.linalg.norm(d["applied_delta_v_rotating_m_s"], axis=1)
            ax[1].step(day[valid], dv[valid], where="post", label="Executed Δv")
            if "maneuver_requested" in d:
                requested = np.flatnonzero(valid & d["maneuver_requested"])
                ax[1].scatter(day[requested], np.zeros(len(requested)), marker="|",
                              label="Maneuver requested")
            ax[1].set_xlabel("Simulation day")
            ax[1].set_ylabel("Δv (m/s)")
            ax[1].legend()
            fig.suptitle(f"{controller}: early-window departures and maneuvers")
            fig.tight_layout()
            fig.savefig(out / f"{controller}_first_{int(args.days)}_days.png", dpi=160)
            plt.close(fig)
    (out / "rollout_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
