# Halo observer and Lunar Gateway tracking

Open [the main notebook](notebooks/00_gateway_tracking_comparison.ipynb). It compares Gateway tracking over **one observer period (8.06 days)** using exact observer knowledge versus an observer orbit estimated from a lunar surface observer (LSO).

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
| LSO | 30° N, 270° E; 10° elevation mask |
| Gateway visibility | Earth/Moon occultation |

The imperfect case retains observer–Gateway cross-covariance. LSO measurements update observer knowledge; Gateway measurements update Gateway without refining the observer. Initial covariances are diagonal with zero cross-covariance. Physical plots use Moon-centered inertial axes; acceleration is derived from the state and dynamics.

The baseline predicts final Gateway position RSS σ of **33.1 m** with perfect observer knowledge and **513.2 m** with LSO-estimated knowledge. Actual errors and RMSE are from one paired noise realization. Local covariance can be overconfident and is not a validated physical error bound.

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
