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
| `notebooks/04_performance_comparison.ipynb` | Covariance forecast and a small Monte Carlo demonstration |
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

## Gateway scope

Gateway is the mission target of the halo spacecraft, but it contributes no observations to this LSO-to-spacecraft estimation study. Its reference state was not supplied, and the older project loads it from a local MAT database absent from the repository. Consequently Gateway is disabled in the baseline configuration. An optional independently sourced Gateway state can be enabled for geometry plotting; no Gateway trajectory is invented or substituted for the supplied observer state. Onboard Gateway tracking, joint estimation, and observer-error effects on Gateway measurements are subsequent study extensions.

## Interpreting results

Compare angles only with angles plus angular rates using error histories, RSS position/velocity sigma, NIS, and Monte Carlo NEES/coverage. A covariance-only run forecasts local uncertainty without estimating actual errors. Acceleration uncertainty is conditional on the force model. The three-trial notebook demonstration is an execution check, not evidence that an estimator is optimal or statistically calibrated. Increase the trial count and test alternative geometry, cadence, noise, and model mismatch before drawing scientific conclusions.

There is no automatic orbit correction, two-body substitution, or inference of a northern trajectory. See `docs/model.md` before replacing the input state.

## Initial validation

The supplied baseline has 389 epochs and 45 above the 10° elevation mask, with about 99.1 km one-period position closure error. All four scientific test groups pass. All four notebooks executed independently in isolated IPython processes; socket restrictions prevented launching Jupyter kernels in the build environment. Notebook outputs include the generated figures. The normal fresh-kernel execution path is included for local runs and CI.

The three-trial demonstration returned final mean NEES of approximately 157 (angles) and 166 (angles plus rates), versus the six-state linear-Gaussian reference mean of 6. This is a warning of covariance overconfidence/nonlinearity and a very small ensemble, not a validated uncertainty result. Preserve these diagnostics when expanding the study. The initial EKF is a baseline to assess, not a recommended final estimator.
