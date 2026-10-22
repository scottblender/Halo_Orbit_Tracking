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
| Gateway visibility | Sun 20° / Earth 10° keep-outs, Earth/Moon occultation and illustrative distance limit |

The imperfect case retains observer–Gateway cross-covariance. LSO measurements update observer knowledge; Gateway measurements update Gateway without refining the observer. Initial covariances are diagonal with zero cross-covariance. Physical plots use Moon-centered inertial axes; acceleration is derived from the state and dynamics.

The EKF comparison illustrates ideal versus measurement-based orbit knowledge. Actual errors and RMSE are from one paired realization; local covariance is not a validated physical error bound. Updated numerical results are generated in the notebook.

The supplied `SNRHL2` observer catalog state is rounded to four decimals and has approximately 99 km of one-period closure error. It is intentionally left uncorrected for the uncontrolled-departure study. Gateway's initial state comes from [the previous cislunar project](https://github.com/scottblender/Gradient-Free-Optimization-for-Cislunar-SDA-2/blob/main/scripts/build_target_case_database.m). Both phases start at model time zero. This study uses LU = 384400 km and TU = 375190.25852 s for both spacecraft; the Gateway source used TU = 375695 s. Sun and Earth exclusion cones are included using an approximate solar phase; brightness and attitude errors remain unmodeled.

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

## Uncontrolled observer departure and Gateway tracking

The notebook uses the **original rounded SNRHL2 catalog state** (four decimal
places; approximately 99 km closure error) to reproduce the *uncontrolled
departure* study. That state is NOT made periodic before the 20-period
simulation. The one-period reference deliberately retains the rounded-input
mismatch; jumps in equal-phase departure include this closure error. The
original continuous propagation, not reference resets, produces the
increasing orbital departure seen in the 20-period grid.

The trajectory grids show **cumulative histories**: panel 1 contains the
continuous interval 0→T, panel 2 contains 0→2T, and panel 20 contains 0→20T.
There are no new integrations starting in each panel, and the initial
condition is the same. `results/departure/observer_period_grid.png` uses
per-panel scaling for legibility; `observer_period_grid_fixed_scale.png`
holds the nominal reference axes fixed, with escaped arcs clipped.

Gateway tracking likewise uses **one continuous EKF run** without resetting
the estimator or its covariance. Each row of
`results/departure/gateway_period_metrics.csv` summarizes errors and
observations cumulatively from **0→kT**, alongside the number of new accepted
observations during the most recent period. Low cumulative RMSE when new
observations are zero is not evidence of ongoing tracking capability.

The table distinguishes:

- **Endpoint observer departure (km):** observer position minus a
  phase-matched, repeated one-period halo reference at t=kT. Since the
  rounded catalog reference has ~99 km closure error, this is a nominal
  reference-departure diagnostic, not physical observer-to-Gateway distance.
- **Maximum observer departure (km):** largest deviation from that reference
  anywhere between 0 and kT.
- **Gateway range (km):** actual distance between observer and Gateway,
  which is what the instrument range-limit screen uses.

No per-period percentage RMSE improvement column is shown because cumulative
average error and endpoint uncertainty convey distinct phenomena. The
reference-departure metric requires a corrected periodic orbit for a
station-keeping cutoff analysis; the rounded reference is retained here to
visualize the observed departure without hiding it.

## Optical visibility assumptions

Both sensor links are screened for geometric slant range, physical Earth/Sun
occultation, and **Earth 10° / Sun 20° center-referenced keep-out angles**.
Observer-to-Gateway measurements also exclude lunar occultation; lunar-surface
measurements additionally require the lunar elevation mask. Body-center
separation uses the unified threshold
`max(physical_occultation, configured_keepout)` as in the user's cislunar
sensor-network and space-based observer projects.

The Sun position is approximated by the planar bicircular phase model, not
SPICE, and the specified initial solar phase is a modeling assumption.
Both links use a configurable **250,000-km hard range cutoff** by default,
which is illustrative: an instrument-based detection bound should be
calculated from the limiting magnitude, target size/albedo and solar phase.
These are not hardware-qualified range limits. References for the general
optical-detection modeling:
- [Cislunar Key Region Surveillance Optimization](https://link.springer.com/article/10.1007/s40295-025-00522-6)
  (body exclusion zones and magnitude limits)
- [An Observability-Driven Architecture for Orbit and Telescope Selection](https://amostech.space/year/2026/an-observability-driven-architecture-for-orbit-and-telescope-selection-in-cislunar-space-based-surveillance/)
  (range, phase, eclipse and sensor constraints)

The notebook loads `gateway_period_metrics.csv` only if the accompanying
summary has a matching configuration, period count and
`continuous_cumulative_horizon_keepout_v3` algorithm version. Older
corrected-reference or independent-window results are automatically rejected.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_notebooks.py
```
