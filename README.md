# Halo Orbit Tracking from a Lunar Surface Observer

A Python/Jupyter study of position, velocity, acceleration, and conditional orbit uncertainty for a known nominal Earth–Moon L2 spacecraft, using optical measurements from a lunar surface observer (LSO) at **30° N, 270° east longitude**.

This is orbit tracking with a supplied nominal initial state and an uncertain prior. It does not perform IOD. The initial estimator is a six-state CR3BP EKF; acceleration and its uncertainty are derived from the dynamics.

## Start here

From the repository root, create an environment and install:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name halo-tracking --display-name "Python (halo tracking)"
python -m jupyterlab
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` without activation. On macOS/Linux activate with `source .venv/bin/activate`.

Choose the **Python (halo tracking)** kernel. Open `notebooks/01_nominal_trajectory.ipynb`, then continue in numerical order. Every notebook also runs independently from a fresh kernel and locates the project from the root or notebooks directory.

For a complete saved baseline without Jupyter:

```powershell
python scripts/run_study.py
python scripts/run_study.py --covariance-only --output results/covariance_only
python -m pytest
python scripts/check_notebooks.py
# In a restricted environment without kernel sockets:
python scripts/check_notebooks.py --in-process
```

## Supplied orbit

The source row is preserved in `data/initial_conditions/supplied_orbit.csv`:

| Field | Value |
| --- | --- |
| Catalog identifiers, semantics retained without interpretation | 241, 6 |
| Family | SNRHL2 |
| Nominal period | 1.8572 TU |
| Reported stability index | 1.6875 |
| Normalized position | [1.0444, -0.0277, -0.1879] |
| Normalized rotating velocity | [-0.0329, -0.1430, 0.0810] |

**Important:** the earlier goal described a northern halo observer, but this supplied catalog family is identified as a southern NRHO in the existing project. The state is used exactly as provided; it is not reflected or replaced. Its four-decimal precision does not establish periodicity. Notebook 01 reports one-period closure error and Jacobi drift. Until a full-precision catalog state is supplied, figures describe an initial-value trajectory rather than a validated periodic halo orbit.

## Organization

| Location | Purpose |
| --- | --- |
| `notebooks/01_nominal_trajectory.ipynb` | Orbit geometry, physical kinematics, closure and conservation checks |
| `notebooks/02_surface_measurements.ipynb` | LSO geometry, elevation mask, angle and angular-rate histories |
| `notebooks/03_orbit_estimation.ipynb` | Paired EKF estimates and ±3σ error plots |
| `notebooks/04_performance_comparison.ipynb` | Covariance forecast and a 100-trial paired Monte Carlo comparison |
| `notebooks/05_horizon_uncertainty.ipynb` | Exact 5/10/20-period uncertainty checkpoints and continued-trajectory check |
| `notebooks/06_gateway_tracking_uncertainty.ipynb` | Observer covariance contribution to known-Gateway pointing/rates |
| `src/halo_tracking/` | Reusable dynamics, frames, sensor, measurements, estimator, validation, plotting |
| `configs/baseline.json` | Editable physical constants and all study assumptions |
| `docs/model.md` | Equations, reference frames, units, uncertainty interpretation |
| `docs/provenance.md` | Related prior code and what was reused conceptually |
| `tests/` | Numerical dynamics, STM, frame, measurement, and covariance checks |
| `results/` | Generated arrays, metrics, configuration snapshots, figures; ignored by Git |

## Baseline assumptions

- Earth–Moon barycentric rotating CR3BP, μ = 0.01215058560962404; LU = 384400 km; TU = 375190.25852 s. Constants remain configurable and must match the original catalog normalization for definitive results.
- Spherical Moon, radius 1737.4 km, synchronous rotation, model north aligned with the CR3BP z axis. The near-side prime meridian points toward Earth (−x); east-positive longitude is a right-handed rotation about +z. No real lunar epoch or IAU orientation is implied.
- Angles reference model inertial axes aligned with the synodic axes at t=0. They are not automatically ICRF/J2000 right ascension and declination.
- Geometric observations above 10° elevation, every 1800 s, over one nominal supplied period. No terrain, illumination, solar exclusion, finite exposure, light-time, aberration, station-position errors, or maneuvers.
- Illustrative 1σ measurement errors: 1 arcsec per angle and 0.01 arcsec/s per angular rate. These are configurable study inputs, not validated sensor specifications.
- The angular-rate channels are independently measured. Differencing the same angle samples creates correlated noise and is deliberately rejected by this baseline implementation. A full same-epoch covariance override is supported; inter-epoch correlation is not.
- Initial 1σ rotating-state uncertainty: 1 km per position component and 0.00001 km/s (0.01 m/s) per velocity component. Both estimators receive the same random initial error and shared angle samples.
- Perfect-model baseline: zero process noise. Optional continuous normalized white-acceleration PSD is integrated through the dynamics. Monte Carlo uses deterministic truth, so nonzero estimator process noise is a robustness assumption rather than matched stochastic truth.

The EKF requires a sufficiently accurate prior. A preliminary run with 10 km / 1 m/s prior sigmas diverged across the long initial observing gap; do not assume the baseline works for that larger uncertainty. Change the prior explicitly and inspect innovations and Monte Carlo consistency.

## Gateway tracking impact

The main workflow is observer covariance propagation with LSO updates, followed by projection into Gateway optical geometry. `notebooks/06_gateway_tracking_uncertainty.ipynb` reports observer-induced pointing and angular-rate uncertainty using Gateway's normalized state from the earlier project's `scripts/build_target_case_database.m`.

Gateway is treated as known. These are geometrical uncertainty contributions, not a Gateway orbit estimate or a spacecraft camera visibility analysis. Both initial phases are taken at model t=0; no mission epoch is asserted. The prior project's TU is 375695 s, while this study retains its configured 375190.25852 s consistently for both normalized trajectories. No earlier dimensional result is claimed to be reproduced.

The observer state contribution is `H_observer P_observer H_observerᵀ`. It is correlated across time. A downstream Gateway estimator should use a joint/consider formulation or explicit temporal correlations rather than simply adding independent noise at every observation.

## Recommended analysis order

Use notebook 01 for trajectory checks, 02 for LSO measurements, 05 for observer uncertainty over time, and 06 for its effect on Gateway pointing. Notebook 03 inspects noisy EKF state errors. Notebook 04 is optional 100-trial Monte Carlo validation, disabled by default; enable `RUN_MONTE_CARLO` to rerun it. Its completed reference summary remains in `data/reference_results/monte_carlo_summary.json`.

## Interpreting results

Compare angles only with angles plus angular rates using error histories, RSS position/velocity sigma, NIS, and Monte Carlo NEES/coverage. A covariance-only run forecasts local uncertainty without estimating actual errors. Acceleration uncertainty is conditional on the force model. Optional validation uses 100 paired Monte Carlo trials over the baseline one-period arc. Test ensemble convergence, alternative geometry, cadence, noise, and model mismatch before drawing scientific conclusions.

There is no automatic orbit correction, two-body substitution, or inference of a northern trajectory. See `docs/model.md` before replacing the input state.

## Initial validation

The supplied baseline has 389 epochs and 45 above the 10° elevation mask, with about 99.1 km one-period position closure error. All seven scientific/rendering checks pass. All six notebooks executed independently in isolated IPython processes; socket restrictions prevented launching Jupyter kernels in the build environment. Notebook outputs include the generated figures. The normal fresh-kernel execution path is included for local runs and CI.

The completed 100-trial comparison returned final mean NEES ≈97 for angles and ≈101 for angles plus rates, versus a six-state linear-Gaussian reference mean of 6. The EKF covariance is overconfident. The uncertainty forecasts and their pointing projections remain conditional local predictions, not validated physical error bounds.

## Longer uncertainty forecasts and reusable 3D styling

Open `notebooks/05_horizon_uncertainty.ipynb` for position, velocity and derived acceleration uncertainty at 1, 5, 10 and 20 nominal periods, or run:

```powershell
.\.venv\Scripts\python.exe scripts/run_horizon_study.py
```

The forecast continuously propagates one initial covariance and uses all available scheduled observations before each checkpoint. It does not restart the filter at each period or add measurements at report epochs. Results include component standard deviations, RSS standard deviations, elapsed days, observation counts, time since the last measurement, and Moon distance. CSV/JSON and full covariance histories are saved under `results/horizons/`; a compact reference CSV/JSON is versioned in `data/reference_results/`.

**These are nominal local covariance forecasts, not empirical Monte Carlo accuracy.** The supplied rounded state departs the lunar halo region by about ten nominal periods. There is no stationkeeping or periodic tiling, so the long results must not be described as the uncertainty of a maintained halo orbit. Use a full-precision corrected orbit and a defined maintenance strategy before making that claim.

For any new Matplotlib 3D trajectory axes, set the box aspect and call `style_3d_axes(ax)` from `halo_tracking.layout`. It recomputes label placement from the projected grid envelope, rendered tick/text sizes, and a dimensionless fraction of the axes area. Labels stay beyond the grid; tick density follows the projected axis length. Camera rotation, resizing and exports use the same rule. No fixed point/pixel label offsets are required. `geometry()` uses this helper and equal physical scales on all three axes.

For the first-update consistency diagnostic and why P0 should retain its physical interpretation, see `docs/uncertainty_interpretation.md`. The baseline initial uncertainty and 1-arcsecond sensor noise were not changed to fit a NEES statistic.
