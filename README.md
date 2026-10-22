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

The physical observer's truth trajectory begins at the original four-decimal
SNRHL2 catalog state, which does not close after its supplied period (roughly
99 km one-period position mismatch). It is propagated **continuously** without
maneuvers or artificial resets for the entire 20-period analysis. To measure
departure from an actual periodic nominal orbit, **only the reference** is
differentially corrected to a nearby closed CR3BP solution. The corrected
nominal period is used consistently for the study horizons.

The first-period RMS departure is therefore expected to be nonzero: the
rounded physical truth starts away from the corrected periodic reference.
The diagnostics separately report (1) corrected reference closure, (2) rounded
catalog closure and (3) initial truth-to-nominal position offset. Never
interpret the very small corrected-reference closure as a lack of departure
of the uncontrolled rounded observer.

The 20 trajectory grid panels display **cumulative histories**, from 0→T,
0→2T, through 0→20T, all from the same initial state and one continuous
propagation. Dashed curves show the corrected periodic nominal reference,
repeated at matching phase. The exports
`results/departure/observer_period_grid.png` and
`observer_period_grid_fixed_scale.png` respectively use auto-scaling and
fixed nominal-reference axis limits.

Gateway tracking also uses one continuous EKF without periodic resets. Each
CSV row summarizes cumulative tracking RMSE and accepted observations over
0→kT; new observations in the most recent period are reported separately.
The physical observer truth begins at the rounded seed, while the corrected
nominal period is used for the horizon boundaries. A small cumulative Gateway
RMSE with zero new measurements is **not** evidence of ongoing optical tracking.

The table distinguishes three separate measures:

- **Endpoint observer departure:** position difference from the corrected,
  equal-phase *nominal orbit*; not range to Gateway or total path length.
- **Maximum observer departure:** largest nominal-reference difference over
  the entire 0→kT horizon.
- **Endpoint Gateway range:** actual observer–Gateway separation relevant to
  optical slant-range screening.

Cumulative-metric caches use method identifier
`continuous_nominal_reference_cumulative_v4`; stale v3 and earlier CSVs
are rejected automatically.

## Gateway prediction endurance with imperfect dynamics

The final tracking section of the notebook now includes a reproducible
**truth-model mismatch** experiment. Gateway's simulated truth is integrated
continuously under CR3BP plus a constant-direction inertial acceleration of
`2e-9 m/s²` by default, while the filter **only knows CR3BP**. This is an
illustrative acceleration, not a measured or mission-calibrated Gateway force
model. Its vector and amplitude can be changed; zero acceleration retains
the original exactly repeated Gateway truth trajectory.

To study *how long Gateway can be predicted without new measurements*, the
experiment imposes a controlled observer-to-Gateway optical blackout after
**8 corrected nominal observer periods** by default. Earlier measurements
must still pass range, occultation, Sun 20° and Earth 10° exclusion masks.
Lunar surface measurements remain available whenever physical masks allow.
Gateway truth and both EKFs are propagated **continuously** across the cutoff;
nothing is reinitialized or recentered on truth at a period boundary.

The experiment records and plots:
- Instantaneous physical Gateway position error for exact-observer and
  LSO-estimated-observer knowledge.
- Rolling **24-hour** RMS position error, avoiding dilution of recent
  prediction errors by earlier accurate periods.
- Elapsed time since the **last actually accepted Gateway measurement**.
- Conditional RSS position sigma from each actual EKF covariance. Because
  the estimator does not model the applied acceleration, its covariance
  is **not a calibrated coverage or nonlinear error bound**.
- First time the position error exceeds configurable **1, 10 and 100 km**
  thresholds after the blackout; thresholds not crossed before the study ends
  are *right-censored*, not assumed safe forever.

Results are saved separately from the baseline matched-model metrics under
`results/gateway_prediction/`: `gateway_prediction_histories.npz`,
`gateway_prediction_horizons.csv`, `gateway_prediction_thresholds.csv`,
`gateway_prediction_summary.json`, and `blind_prediction_endurance.png`.
Saved runs are reused only when configuration and all disturbance, threshold,
and cutoff settings match. These outputs answer a controlled prediction
endurance question; the chosen outage starts at 8T by **experimental design**
and is not claimed to be the spacecraft's actual loss-of-visibility epoch.

Set `HALO_GATEWAY_TRUTH_ACCELERATION_M_S2` and
`HALO_GATEWAY_BLACKOUT_PERIODS` to explore other disturbances and outage
starts. `HALO_GATEWAY_PERIODS` sets the entire analysis horizon
(default 20; CI uses 2 and automatically places the cutoff at 1T).

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
