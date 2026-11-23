"""Benchmark the single-burn controller on a common full-year sample grid."""
import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
from halo_tracking import lso_stationkeeping as lso

TRIALS = 20
MASTER_SEED = 20261122
# Independent orbit corrections can shift matching epochs by a fraction of a second.
GRID_TOLERANCE_DAYS = 1e-5
ROOT = Path(__file__).resolve().parents[1]


def run(index, fit_path):
    orbit = lso.OrbitModel(*lso.physics.load_settings(ROOT))
    gains = np.asarray(json.loads(fit_path.read_text())["gains"])
    settings = orbit.settings
    rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED).spawn(TRIALS)[index - 1])
    noise_seed = int(rng.integers(2**32))
    sigma = np.repeat([settings["initial_position_sigma_km"],
                       settings["initial_velocity_sigma_m_s"]], 3)
    initial = rng.normal(size=6) * sigma

    # Inject a randomized initial truth state without changing controller code.
    source = inspect.getsource(lso.stationkeeping_trial)
    signature = "navigation='extended'):"
    assignment = "state = orbit.state(np.array([.03,-.02,.01,.0003,-.0002,.0001]), 0.)"
    assert source.count(signature) == source.count(assignment) == 1
    source = source.replace(signature, "navigation='extended', initial_departure=None):")
    source = source.replace(assignment, "state = orbit.state(initial_departure, 0.)")
    namespace = vars(lso).copy()
    exec(compile(source, lso.__file__, "exec"), namespace)
    result = namespace["stationkeeping_trial"](
        orbit, gains, seed=noise_seed, duration_days=settings["annual_horizon_days"],
        planning_stride_steps=settings["annual_planning_stride_steps"],
        stop_on_departure_cutoff=False, initial_departure=initial,
    )
    summary = result["summary"]
    summary.update(trial=index, initial_departure=initial.tolist())
    day, output = result["day"], result["output"]
    assert summary["completed"] and np.isfinite(output).all() and np.all(np.diff(day) >= 0)
    values = np.column_stack((np.linalg.norm(output[:, :3], axis=1),
                              np.linalg.norm(output[:, 3:], axis=1)))
    np.savez_compressed(f"trial-{index:02d}.npz", day=day, values=values,
                        summary=json.dumps(summary))
    print(json.dumps(summary), flush=True)


def aggregate(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = lso.physics.load_settings(ROOT)[1]
    limits = np.array([settings["position_limit_km"], settings["velocity_limit_m_s"]])
    trials, rows, grid = [], [], None
    horizon = settings["annual_horizon_days"]
    largest_shift = 0.0
    for index in range(1, TRIALS + 1):
        path = directory / f"trial-{index:02d}.npz"
        with np.load(path) as data:
            day, values = data["day"], data["values"]
            row = json.loads(data["summary"].item())
        if not row["completed"] or row["trial"] != index:
            raise ValueError(f"{path}: expected completed trial {index}")
        if (day.ndim != 1 or day.size < 2 or not np.isfinite(day).all()
                or day[0] != 0 or not np.isclose(day[-1], horizon, rtol=0, atol=1e-9)):
            raise ValueError(f"{path}: invalid propagation epochs for the requested horizon")
        steps = np.diff(day)
        if np.any(steps < 0):
            raise ValueError(f"{path}: propagation epochs must be nondecreasing")
        if values.shape != (day.size, 2) or not np.isfinite(values).all():
            raise ValueError(f"{path}: expected finite position and velocity errors at every epoch")
        if grid is None:
            grid = day
            repeated_epochs = steps == 0
        if day.shape != grid.shape or not np.array_equal(steps == 0, repeated_epochs):
            raise ValueError(f"{path}: sample layout differs from trial 1")
        shift = float(np.max(np.abs(day - grid)))
        if shift > GRID_TOLERANCE_DAYS:
            raise ValueError(
                f"{path}: maximum epoch shift {shift:g} days exceeds "
                f"{GRID_TOLERANCE_DAYS:g} days"
            )
        largest_shift = max(largest_shift, shift)
        assert np.allclose(values.max(axis=0),
                           [row["maximum_position_km"], row["maximum_velocity_m_s"]])
        assert row["all_evaluated_samples_inside"] == bool((values <= limits).all())
        rows.append(row)
        trials.append(values)

    print(f"Largest timestamp difference across matched samples: {largest_shift:.12g} days")
    assert len({row["seed"] for row in rows}) == TRIALS
    values = np.stack(trials)
    mean = values.mean(axis=0)
    low, high = np.percentile(values, [5, 95], axis=0)
    exits = np.flatnonzero(np.any(mean > limits, axis=1))
    success = sum(row["all_evaluated_samples_inside"] for row in rows)
    delta_v = np.array([row["annual_delta_v_m_s"] for row in rows])
    burns = np.array([row["burns"] for row in rows])
    summary = dict(
        trials=TRIALS, successful=success, days=float(grid[-1]), samples=len(grid),
        peaks_of_mean=mean.max(axis=0).tolist(), peak_mean_days=grid[mean.argmax(axis=0)].tolist(),
        peaks_of_p95=high.max(axis=0).tolist(), worst_peaks=values.max(axis=(0, 1)).tolist(),
        first_mean_exit_day=float(grid[exits[0]]) if exits.size else None,
        first_exits={row["trial"]: row["first_sampled_exit_day"] for row in rows
                     if row["first_sampled_exit_day"] is not None},
        mean_delta_v_m_s=float(delta_v.mean()), std_delta_v_m_s=float(delta_v.std(ddof=1)),
        range_delta_v_m_s=[float(delta_v.min()), float(delta_v.max())],
        mean_burns=float(burns.mean()), range_burns=[int(burns.min()), int(burns.max())],
        master_seed=MASTER_SEED,
    )
    print(json.dumps(summary, indent=2), flush=True)
    output = Path("benchmark-results")
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = ["trial", "seed", "maximum_position_km", "maximum_velocity_m_s",
               "all_evaluated_samples_inside", "first_sampled_exit_day", "annual_delta_v_m_s", "burns"]
    with (output / "stationkeeping_mc_trials.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns + ["initial_x_km", "initial_y_km", "initial_z_km",
                                   "initial_vx_m_s", "initial_vy_m_s", "initial_vz_m_s"])
        for row in rows:
            writer.writerow([row[key] for key in columns] + row["initial_departure"])
    np.savetxt(output / "stationkeeping_mc_timeseries.csv",
               np.column_stack([grid, mean, low, high, np.all(values <= limits, axis=2).mean(axis=0)]),
               fmt="%.12g", delimiter=",", comments="",
               header=("day,mean_position_km,mean_velocity_m_s,p05_position_km,p05_velocity_m_s,"
                       "p95_position_km,p95_velocity_m_s,fraction_inside_corridor"))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    fig.subplots_adjust(left=.11, bottom=.14, top=.76, right=.96, hspace=.30)
    fig.text(.11, .93, "Single-burn NRHO stationkeeping", fontsize=22, weight="bold", color="#172a44")
    fig.text(.11, .875, f"{TRIALS} trials over {grid[-1]:g} days | {success} stayed within both limits",
             fontsize=11, color="#465d74")
    for col, label in enumerate(["Position error (km)", "Velocity error (m/s)"]):
        ax = axes[col]
        ax.plot(grid, mean[:, col], color="#174e86", lw=.9, label="Mean across trials")
        ax.fill_between(grid, low[:, col], high[:, col], color="#7faad5", alpha=.42,
                        label="5th–95th percentile across trials", linewidth=0)
        ax.axhline(limits[col], color="#af372e", ls="--", lw=1.2, label="Corridor limit")
        ax.set(ylabel=label, xlim=(0, grid[-1]))
        if max(mean[:, col].max(), high[:, col].max()) > 10 * limits[col]:
            ax.set_yscale("symlog", linthresh=limits[col])
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(color="#dce4ec", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Elapsed time (days)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.102, .832),
               frameon=False, ncols=3)
    fig.text(.11, .032, "Errors are magnitudes relative to the reference orbit. Compliance is evaluated at sampled epochs.",
             fontsize=9, color="#465d74")
    fig.savefig(output / "stationkeeping_mc.png", dpi=190)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--trial", type=int, choices=range(1, TRIALS + 1))
    modes.add_argument("--aggregate", type=Path)
    parser.add_argument("--fit", type=Path)
    args = parser.parse_args()
    if args.aggregate is not None:
        aggregate(args.aggregate)
    elif args.fit is not None:
        run(args.trial, args.fit)
    else:
        parser.error("--trial requires --fit")
