# Halo observer and Lunar Gateway tracking

Open [the main notebook](notebooks/00_gateway_tracking_comparison.ipynb). It compares Gateway tracking over **one observer period (about 8.06 days)** using exact observer knowledge versus an observer orbit estimated from a lunar surface observer (LSO).

Both cases use the same truth trajectories, Gateway prior and noisy Gateway measurements. The notebook includes 3D geometry, position/velocity/acceleration histories for both spacecraft, estimation errors, predicted uncertainty, and a results table with **position and velocity RMSE**.

## Run

From the repository root in Windows PowerShell (activation is unnecessary):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name halo-tracking --display-name "Python (halo tracking)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Select **Python (halo tracking)** and run the notebook from top to bottom. After pulling updates, restart the kernel. Edit `configs/baseline.json` to change the inputs. Figures and numerical results are saved under `results/gateway_comparison/`.

## Study assumptions

| Parameter | Baseline |
|---|---|
| Dynamics | Earth–Moon CR3BP, no maneuvers |
| Observer initial uncertainty, per axis | 1 km position; 0.01 m/s velocity, 1σ |
| Gateway initial uncertainty, per axis | 1 km position; 0.01 m/s velocity, 1σ |
| Both optical links | Angles only; 1 arcsecond noise per angle |
| Measurement cadence | 30 minutes |
| LSOs | 30° N / 270° E; 30° N / 90° E; 85° S / 90° E; 10° mask |
| Gateway visibility | Earth/Moon occultation |

The imperfect case retains observer–Gateway cross-covariance. LSO measurements update observer knowledge; Gateway measurements update Gateway without refining the observer. Initial covariances are diagonal with zero cross-covariance. Physical plots use Moon-centered inertial axes; acceleration is derived from the state and dynamics.

The EKF comparison illustrates ideal versus measurement-based orbit knowledge. Actual errors and RMSE are from one paired realization; local covariance is not a validated physical error bound. Updated numerical results are generated in the notebook.

The supplied `SNRHL2` catalog observer state is rounded to four decimals, with about 99 km closure error before correction. The notebook uses a numerically differentially corrected periodic reference and checks its one-period closure. Gateway's initial state comes from [the previous cislunar project](https://github.com/scottblender/Gradient-Free-Optimization-for-Cislunar-SDA-2/blob/main/scripts/build_target_case_database.m). Both phases start at model time zero. This study uses LU = 384400 km and TU = 375190.25852 s for both spacecraft; the Gateway source used TU = 375695 s. Sun exclusion, brightness and attitude errors are omitted.

## Supporting code

- `src/halo_tracking/`: dynamics, measurements, estimation and plotting.
- `configs/baseline.json`: initial conditions and study settings.
- `tests/`: numerical model, estimator coupling and figure-layout checks.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_notebooks.py
.\.venv\Scripts\python.exe scripts/run_gateway_comparison.py
```

The last command reproduces results without Jupyter. For environments that block kernel sockets, use `scripts/check_notebooks.py --in-process`.

## System identification data

The final notebook section generates 20 training, 5 validation and 5 test trajectories under `results/system_identification/`. Each NPZ contains uniform times, true rotating states, inertial position/velocity/acceleration, per-site ideal/noisy angles and visibility. `metadata.json` records units, configuration and split definitions.

Initial-condition spreads are 1 km and 0.01 m/s per axis, independently sampled for each one-period arc. The first learning task is next-state prediction from simulated truth states. These are simulation labels, not position measurements obtained from LSOs. Keep whole trajectories within their assigned split; fit scaling on training data only. No learned model, thrust response or stationkeeping controller is included yet.

## Corrected departure and continuous Gateway tracking

The notebook differentially corrects the four-decimal southern NRHO catalog
state to a periodic Earth–Moon CR3BP reference. The new reference must close
to better than 1 m after one period. The corrected period is approximately
1.85564 TU (the rounded catalog supplied 1.8572 TU).

The uncontrolled study propagates the corrected state for 20 periods, alongside
eight deterministic 1-km and 0.01-m/s per-axis initial-condition perturbations.
The **shared-scale 3D trajectory grid** is exported to
`results/departure/observer_period_grid.png`, and the departure plot includes
the ensemble median and 10th–90th percentile envelope. This is reference
departure, not an operational station-keeping threshold.

The Gateway study now performs **one continuous truth and EKF simulation**
across all requested observer periods. It uses one seeded perturbed observer
trajectory, carries the observer and Gateway estimates and covariances across
period boundaries, and reports per-period estimation RMSE and observation
availability in `results/departure/gateway_period_metrics.csv`. The Gateway
truth follows its own repeating CR3BP nominal orbit, without station-keeping.
No filter is reinitialized between periods.

An illustrative **250,000-km line-of-sight range limit** applies independently
to lunar-surface-to-observer and observer-to-Gateway links. These configurable
hard limits in `configs/baseline.json` are study sensitivity assumptions,
**not validated detection, brightness, Sun-exclusion, or instrument models**.
The angular noise is still modeled as constant within the range limit.
Metrics from intervals without observations are prediction results, not proof
of continued tracking capability.

The notebook **reuses** `gateway_period_metrics.csv` when the accompanying
summary matches the configuration, period count, and continuous-study version;
otherwise it regenerates both files. Older independent-window studies are
invalidated automatically. Because this new method answers a different
question, the new RMSE table must not be compared directly to the former
independently reset results.

Run the scientific checks and notebook from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_notebooks.py
```
