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

The supplied `SNRHL2` observer state is a southern NRHO initial condition rounded to four decimals, with about 99 km closure error after one nominal period. Gateway's initial state comes from [the previous cislunar project](https://github.com/scottblender/Gradient-Free-Optimization-for-Cislunar-SDA-2/blob/main/scripts/build_target_case_database.m). Both phases start at model time zero. This study uses LU = 384400 km and TU = 375190.25852 s for both spacecraft; the Gateway source used TU = 375695 s. Sun exclusion, brightness and attitude errors are omitted.

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

## Uncontrolled orbit departure

The notebook compares 20 periods of uncontrolled observer propagation against the repeated one-period CR3BP reference. It exports position/velocity departure, per-period RMS plots, and a **single shared-scale 3D grid** containing all 20 observer periods to `results/departure/`. The rounded observer reference has about 99 km closure mismatch; departure includes reference resets and phase drift. This diagnostic does not identify chaos or select a stationkeeping cutoff.

Gateway follows a single CR3BP reference orbit repeated at its own period. The notebook also compares Gateway position/velocity RMSE, predicted uncertainty, and observation availability across observer periods. Each comparison starts at an observer-period boundary, lasts one complete observer period, and uses the same initial Gateway uncertainty; the tracking windows are independent and consecutive. The results are saved to `results/departure/gateway_period_metrics.csv`; the notebook reuses this CSV when its summary matches the configuration and period count. The shared-scale trajectory grid is `results/departure/observer_period_grid.png`. Gateway station-keeping is not modeled. Positive position-RMSE change indicates worse tracking than period 1, but RMSE from a single noise realization is not a statistical performance guarantee.
