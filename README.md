# Halo observer and Lunar Gateway tracking

Open [the main notebook](notebooks/00_gateway_tracking_comparison.ipynb). It compares Gateway tracking over **one observer period (about 8.06 days)** using exact observer knowledge versus an observer orbit estimated from a lunar surface observer (LSO).

Both cases use the same truth trajectories, Gateway prior and noisy Gateway measurements. The notebook includes 3D geometry, position/velocity/acceleration histories for both spacecraft, estimation errors, predicted uncertainty, and a prediction-endurance table with **instantaneous position error and rolling 24-hour position RMSE**.

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
| Observer initial uncertainty, per axis | 1 km position, 0.01 m/s velocity, 1σ |
| Gateway initial uncertainty, per axis | 1 km position, 0.01 m/s velocity, 1σ |
| Both optical links | Angles only, 1 arcsecond noise per angle |
| Measurement cadence | 30 minutes |
| LSOs | 30° N / 270° E, 30° N / 90° E, 85° S / 90° E, 10° mask |
| Gateway visibility | Sun 20° / Earth 10° keep-outs, Earth/Moon occultation and illustrative distance limit |

The imperfect case retains observer–Gateway cross-covariance. LSO measurements update observer knowledge. Gateway measurements update Gateway without refining the observer. Initial covariances are diagonal with zero cross-covariance. Physical plots use Moon-centered inertial axes. Acceleration is derived from the state and dynamics.

The EKF comparison illustrates ideal versus measurement-based orbit knowledge. Actual errors and RMSE are from one paired realization. Local covariance is not a validated physical error bound. Updated numerical results are generated in the notebook.

The supplied `SNRHL2` observer catalog state is rounded to four decimals and has approximately 99 km of one-period closure error. It is intentionally left uncorrected for the uncontrolled-departure study. Gateway's initial state comes from [the previous cislunar project](https://github.com/scottblender/Gradient-Free-Optimization-for-Cislunar-SDA-2/blob/main/scripts/build_target_case_database.m). Both phases start at model time zero. This study uses LU = 384400 km and TU = 375190.25852 s for both spacecraft. The Gateway source used TU = 375695 s. Sun and Earth exclusion cones are included using an approximate solar phase. Brightness and attitude errors remain unmodeled.

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

## Station-keeping system identification dataset — first step

The learning task is now **estimate an impulsive corrective Δv** that keeps
the uncontrolled observer near a **differentially corrected periodic halo
reference**, rather than predict its next unforced CR3BP state. Generate data
by running the notebook's final section or importing
`halo_tracking.identification.generate_dataset`. The generator saves 20
training, 5 validation, and 5 independent held-out test trajectories by
default, each spanning **two corrected nominal periods**. The observation
grid is sampled at 6-hour intervals and guidance decisions at **four epochs
per period**. Every independent arc is propagated continuously. Maneuvers
are candidate labels, not applied to those trajectories.

### Explicit preliminary guidance thresholds

| Measure | Maneuver trigger (OR condition) | Departure corridor |
|---|---:|---:|
| Reference-relative position norm | **50 km** | **100 km** |
| Reference-relative velocity norm | **0.5 m/s** | **1.0 m/s** |

These are **exploratory sensitivity-study parameters, not mission-certified
navigation or station-keeping requirements**. A rounded catalog initial state
is about **23 km and 0.13 m/s** from the corrected nominal state. The
uncontrolled comparison reaches approximately **110 km and 0.9 m/s by
0.5T**, several hundred km by 1.5T, then diverges much further. Thus the
triggers activate before the one-period-scale departure becomes large.
Position and velocity deviations are evaluated in physical Moon-centered
inertial units, including the rotating-frame velocity cross term.

Candidate impulsive burns are three-axis **m/s velocity changes** generated
from a CR3BP state-transition matrix. A weighted regularized least-squares
correction targets the *future nominal state* after **0.25T**. The
**10 m/s** single-impulse cap is another configurable study assumption.
Each candidate is nonlinear-propagated for validation. If a burn exceeds the
cap or fails to reduce the weighted terminal error, its label is **NaN and
`maneuver_label_valid=False`**, not an arbitrary clipped solution. Under
threshold, the "do nothing" Δv label is exactly zero. These labels are a
**single-burn local guidance surrogate**, not globally optimal multi-maneuver
station-keeping trajectories.

### Saved arrays and leakage prevention

`results/system_identification/{training,validation,test}.npz` each contain:

- **Deployment-relevant inputs:** `decision_navigation_states_normalized`,
  `decision_nominal_states_normalized`,
  `decision_navigation_deviation_mci_km_m_s`, phase/time, EKF covariance
  diagonal, per-site noisy angular observations, availability masks and
  elapsed time since the last LSO observation. Navigation estimates use
  all configured lunar surface optical sensors.
- **Supervised labels:** `target_delta_v_rotating_m_s` and
  `target_delta_v_mci_m_s`, `maneuver_required`,
  `maneuver_label_valid`, and nonlinear propagated before/after weighted
  costs.
- **Evaluation only:** `truth_rotating_normalized` and
  `decision_truth_states_normalized`. **Never use these as deployable
  training features**, because truth would not be available onboard.

The `metadata.json` records split provenance, corrected reference,
guidance assumptions, units, validity rules and limitations. Train scaler
statistics must come from the training split only. No trained model or
closed-loop controller is included yet.

### Link to nonlinear ARX

The target is a three-component corrective Δv relative to the corrected,
linearly unstable periodic halo reference. The labels are calculated by a
regularized STM-based correction and checked with nonlinear propagation.

Start by predicting these labels from current navigation inputs. Test whether
navigation history improves that baseline. A nonlinear ARX (NARX) model would
also use past Δv outputs, not just navigation history. Its benefit has not
been demonstrated here.

Candidate burns are not applied to the current arcs. Testing a NARX controller
requires additional trajectories with applied burns and the resulting
navigation states. Evaluate departure, maneuver feasibility, and total Δv.
Keep complete trajectories in separate data splits and exclude invalid labels.
No learned model or closed-loop controller is trained in this step.

## Uncontrolled observer departure and Gateway tracking

Station-keeping is deliberately omitted to expose nominal-orbit departure,
changes in tracking performance, and the corrective Δv requirements and
maneuver feasibility under the assumed burn cap. This is a stress-test
baseline. Candidate burn labels do not yet demonstrate closed-loop recovery.

The physical observer's truth trajectory begins at the original four-decimal
SNRHL2 catalog state, which does not close after its supplied period (roughly
99 km one-period position mismatch). It is propagated **continuously** without
maneuvers or artificial resets for the entire 20-period analysis. To measure
departure from an actual periodic nominal orbit, **only the reference** is
differentially corrected to a nearby closed CR3BP solution. The corrected
nominal period is used consistently for the study horizons.

The first-period RMS departure is therefore expected to be nonzero: the
uncontrolled observer starts away from the corrected periodic reference.
The diagnostics separately report (1) corrected reference closure, (2) rounded
catalog closure and (3) initial truth-to-nominal position offset. Never
interpret the very small corrected-reference closure as a lack of departure
of the uncontrolled NRHO observer.

The 20 trajectory grid panels display **cumulative histories**, from 0→T,
0→2T, through 0→20T, all from the same initial state and one continuous
propagation. Dashed curves show the corrected periodic nominal reference,
repeated at matching phase. The exports
`results/departure/observer_period_grid.png` and
`observer_period_grid_fixed_scale.png` respectively use auto-scaling and
fixed nominal-reference axis limits.

Gateway tracking still uses a continuous EKF with no periodic resets. The
notebook removes the redundant cumulative baseline-Gateway metrics table and
focuses instead on the observation-blackout prediction-endurance study,
which reports instantaneous and rolling errors against a perturbed truth.
The uncontrolled NRHO observer is initialized from the catalog state. A small cumulative
RMSE with zero new measurements would not establish ongoing optical tracking.

The table distinguishes three separate measures:

- **Endpoint observer departure:** position difference from the corrected,
  equal-phase *nominal orbit*. Not range to Gateway or total path length.
- **Maximum observer departure:** largest nominal-reference difference over
  the entire 0→kT horizon.
- **Endpoint Gateway range:** actual observer–Gateway separation relevant to
  optical slant-range screening.

The optional baseline Gateway cumulative-metrics API remains available for
backward compatibility, but the notebook no longer generates or displays its
redundant table. It instead uses the newer prediction-endurance results.

## Gateway prediction endurance with imperfect dynamics

The final tracking section of the notebook now includes a reproducible
**truth-model mismatch** experiment. Gateway's simulated truth is integrated
continuously under CR3BP plus a constant-direction inertial acceleration of
`2e-9 m/s²` by default, while the filter **only knows CR3BP**. This is an
illustrative acceleration, not a measured or mission-calibrated Gateway force
model. Its vector and amplitude can be changed. Zero acceleration retains
the original exactly repeated Gateway truth trajectory.

To study *how long Gateway can be predicted without new measurements*, the
experiment imposes a controlled observer-to-Gateway optical blackout after
**8 corrected nominal observer periods** by default. Earlier measurements
must still pass range, occultation, Sun 20° and Earth 10° exclusion masks.
Lunar surface measurements remain available whenever physical masks allow.
Gateway truth and both EKFs are propagated **continuously** across the cutoff.
Nothing is reinitialized or recentered on truth at a period boundary.

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
  thresholds after the blackout. Thresholds not crossed before the study ends
  are *right-censored*, not assumed safe forever.

Results are saved separately from the baseline matched-model metrics under
`results/gateway_prediction/`: `gateway_prediction_histories.npz`,
`gateway_prediction_horizons.csv`, `gateway_prediction_thresholds.csv`,
`gateway_prediction_summary.json`, and `blind_prediction_endurance.png`.
Saved runs are reused only when configuration and all disturbance, threshold,
and cutoff settings match. These outputs answer a controlled prediction
endurance question. The chosen outage starts at 8T by **experimental design**
and is not claimed to be the spacecraft's actual loss-of-visibility epoch.

Set `HALO_GATEWAY_TRUTH_ACCELERATION_M_S2` and
`HALO_GATEWAY_BLACKOUT_PERIODS` to explore other disturbances and outage
starts. `HALO_GATEWAY_PERIODS` sets the entire analysis horizon
(default 20. CI uses 2 and automatically places the cutoff at 1T).

## Optical visibility assumptions

Both sensor links are screened for geometric slant range, physical Earth/Sun
occultation, and **Earth 10° / Sun 20° center-referenced keep-out angles**.
Observer-to-Gateway measurements also exclude lunar occultation. Lunar-surface
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

## Nominal-orbit stability diagnostics

The notebook calls `halo_tracking.stability.orbit_stability` to integrate the
corrected reference and its STM over one period, report Floquet multipliers,
and check spectral-radius sensitivity to integration tolerances. A robust
multiplier magnitude above one identifies local linear instability. Near-unit
results remain unresolved. The reported e-folding time is a local modal
growth timescale, not a maneuver deadline.

A separate seeded ensemble starts around the corrected nominal state and
reports sampled departure-corridor exits over five periods. It does not
replace the catalog-seed departure study or prove nonlinear stability.
`save_stability` writes `stability_summary.json` and
`stability_histories.npz` under `results/stability/`. The notebook also exports
`nominal_perturbation_departure.png`. No burns are applied in these diagnostics.
