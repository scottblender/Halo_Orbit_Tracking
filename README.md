# Halo observer and Lunar Gateway tracking

## Current research experiment: 365-day observer-only NARX station-keeping

**Use [notebooks/01_narx_results.ipynb](notebooks/01_narx_results.ipynb) and
[scripts/run_one_year_study.py](scripts/run_one_year_study.py) for the current
observer station-keeping experiment. Older descriptions further below document
historical prototypes; those should not be mistaken for the active architecture.**

The six model assumptions and acceptance checks are:

1. **Observer reference:** numerically differentially corrected periodic Earth–Moon CR3BP halo, with closure checked by `correct_halo_reference`.
2. **Gateway truth:** supplied nominal 9:2 NRHO state validated for one-period closure (position and velocity), evaluated as a prescribed, repeated periodic CR3BP ephemeris. It has **zero commanded maneuvers**. This verifies closure of the supplied seed within the model, *not* flight ephemeris fidelity or independent 9:2-family classification.
3. **Measurements:** three lunar surface optical sites, plus observer-to-Gateway angles where available, drive a coupled EKF. The controller uses estimated observer deviations, previous **executed** burns, orbital phase, visibility and estimated covariance, never truth.
4. **STM policy:** at each candidate decision, an STM-based least-squares correction is checked by nonlinear propagation and scheduled/applied to the observer.
5. **Learned policy:** a supervised *NARX-inspired* (lagged-feature gradient-boosting) classifier, three-component Δv regressor and discrete burn-delay predictor are fitted **only** on short training trajectories. During the 365-day rollout, their actual predictions are executed with **no STM fallback**.
6. **Paired comparison:** uncontrolled, STM and NARX policies receive identical observer initial perturbations, EKF initial-estimation errors, and per-epoch pre-sampled LSO/camera noise. Geometry and accepted observations can differ as policies diverge.

The annual run reports per-trajectory/full-sample corridor survival, incomplete
runs (counted as failures), observer departure and total executed Δv,
Gateway tracking RMSE and joint 100-m/0.01-m/s compliance. Results are **not**
claimed until the 365-day generator has actually completed.

Run from a checkout with development dependencies installed:

```bash
python -m pip install -e ".[dev]"
python scripts/run_one_year_study.py --days 365 --arcs 5
```

Output: `results/one_year/summary.json`, `arc_metrics.csv`, three
`{uncontrolled,stm,narx}_arc_XXX.npz` files per realization, and the frozen
`observer_narx.joblib` bundle. Notebook 01 loads these artifacts and produces
comparative trajectory, Δv, Gateway estimation, and survival summaries.

For a quick implementation smoke test only:

```bash
python scripts/run_one_year_study.py --days 0.5 --arcs 1 --training-arcs 5 --output results/one_year_smoke
```

The full-year results are sensitive to the short-arc teacher-training population
and the idealized CR3BP / optical sensor model. They must not be interpreted
as high-fidelity operational mission performance. Increase trajectory diversity
and held-out arc count before quantitative reliability claims.

---


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
| Observer initial uncertainty, per axis | 1 km position, 0.1 m/s velocity, 1σ |
| Gateway initial uncertainty, per axis | 1 km position, 0.1 m/s velocity, 1σ |
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

## Phase-aware correction labels and Gateway tracking accuracy

The primary objective is **estimating Lunar Gateway position within 100 m and velocity within a proposed 0.01 m/s**. These are estimation-error targets, measured against the simulated true Gateway state. They are not a claim of achieved accuracy.

The observer's uncontrolled halo departure is a separate problem. The observer's **physical reference-departure** trigger remains **50 km or 0.5 m/s**, with a **100 km / 1 m/s** nominal corridor and **10 m/s** candidate-impulse cap. The stricter Gateway **physical trajectory-correction** labels use **50 m or 0.005 m/s** triggers, **100 m / 0.01 m/s** tolerances, and **0.5 m/s** cap. These separate Gateway trajectory-correction labels are not themselves a tracking-error controller. The 0.01 m/s goal is a provisional design metric that requires feasibility testing.

Run `python scripts/run_narx_dataset.py` for independent training, validation and test arcs in `results/narx_phase_aware/`. The generator considers several possible burn delays and evaluates each candidate at a common terminal epoch using linearized guidance and nonlinear CR3BP propagation. It records Δv, selected delay and orbital phase, validity masks and weighted terminal errors for both spacecraft.

**What this dataset establishes:** examples of estimated-state-based corrective Δv and burn timing for two spacecraft relative to their respective prescribed trajectories. The observer reference is corrected and periodic. The Gateway reference is its initial-state CR3BP propagation, not a proven periodic orbit.

**What it does not establish:** achieved 100 m Gateway *tracking* accuracy, guaranteed terminal tolerance satisfaction, fuel-optimality, or closed-loop control. The phase-aware generator currently uses noisy simulated state proxies, not sensor-derived two-spacecraft navigation. Calculated burns are not executed, and histories do not include previous applied controls. Consequently it supports initial phase-aware supervised regression, not a complete applied-control NARX experiment.

The notebook now distinguishes those concepts and shows label coverage from both the original observer-only generator and the newer phase-aware generator. A future controlled experiment must apply burns, propagate their effects into observation geometry and both estimators, and measure Gateway tracking error over multiple noisy runs.

## Two-stage NARX guidance baseline

Run `python scripts/run_two_stage_narx.py` to regenerate the covariance-sampled Monte Carlo episodes and train an initial two-stage **supervised imitation** model. The first stage classifies whether a burn is requested. For feasible requested burns, the second stage predicts the three Δv components (m/s) and which of four available burn delays the STM guidance selected.

Inputs are the most recent estimated observer or Gateway reference deviations, **previous executed** burns, orbital phase, optical availability and EKF position-covariance trace. No simulated truth is supplied to the predictor. Training uses only complete training episodes. Validation and held-out test episodes are independent.

The output `results/narx_two_stage/metrics.json` reports classification precision, recall, F1 and confusion matrix, Δv RMSE, burn-delay class accuracy, and number of feasible labels. Δv and timing scores are **teacher-forced** on decisions with feasible reference labels. These metrics measure how well the learned model imitates the STM policy. They do not establish whether a learned controller maintains Gateway tracking within 100 m and 0.01 m/s. That requires replacing the guidance policy with model predictions during a separate controlled Monte Carlo evaluation.

## Monte Carlo initial conditions for the NARX dataset

Each closed-loop NARX episode starts near the **corrected periodic halo reference**, rather than alternating between that reference and the rounded catalog state. Initial physical observer and Gateway states are drawn from Gaussian distributions with the same **1 km position and 0.1 m/s velocity (1σ per axis)** uncertainties used by the navigation filters. Separate independent draws represent the initial navigation estimation error. Thus the true orbit offset from the reference is not confused with the estimator error.

In normalized coordinates the initialization is
`x_true(0) = x_ref(0) + δx_physical` and
`x_est(0) = x_true(0) + δx_navigation`, with independent
`δx_physical, δx_navigation ~ N(0, P0)`.
The exported training files contain both offsets, initial truth/estimate and the block-diagonal covariance so that each Monte Carlo realization can be reproduced and evaluated. Training, validation and test splits use separate random streams. The rounded catalog trajectory remains in the **uncontrolled baseline study**, not in the Monte Carlo training population.

## Executed-burn NARX training data

The **uncontrolled halo observer tracks Lunar Gateway** with optical measurements. As the observer departs from the corrected periodic halo, its estimated position and the available viewing geometry affect Gateway tracking. The study compares exact observer knowledge against the observer estimated using three lunar surface observatories. Three is the network used in this experiment; determining whether fewer stations suffice requires a separate comparison.

The corrected halo is a nearly periodic CR3BP reference. The original rounded catalog state is instead propagated continuously as an uncontrolled physical trajectory. These are **different states**. Instability describes the local behavior around the periodic reference, not whether its numerical closure is accurate.

The new generator completes an **applied-control simulation loop**:

1. Propagate the observer and Gateway true states under CR3BP, including prior executed burns.
2. Generate angle-only LSO and observer-to-Gateway measurements when the optical geometry permits.
3. Propagate and update a coupled twelve-state EKF using those measurements.
4. Compare candidate burn phases using the EKF states, propagate each candidate nonlinearly, and schedule accepted impulses.
5. Apply each scheduled Δv to the simulated truth and EKF prediction, repeat observation and navigation updates, and save the complete navigation and executed-control histories.

Run `python scripts/run_closed_loop_narx.py` to generate independent training, validation and test files under `results/narx_closed_loop/`. Each NPZ file contains estimated states, reference errors, selected Δv/phase labels, actual executed Δv, causal lag histories, visibility, covariance summaries and **Gateway tracking error relative to truth**. The next burn is never included in its own lagged-control input. The notebook produces a smaller illustration of this closed-loop dataset.

The observer's maneuver trigger remains **50 km or 0.5 m/s**, with a **100 km / 1 m/s** reference-departure corridor. The separate Gateway physical-correction trigger is **50 m or 0.005 m/s**, with a **100 m / 0.01 m/s** reference-departure corridor. The engineering goal of **Gateway *tracking* error below 100 m and 0.01 m/s** is assessed from the estimated-minus-true Gateway state, not assumed from those maneuver triggers.

The controller data record executed commands, but no trained NARX model is included. The guidance policy is an STM-based candidate selector, not a learned controller, and it does not directly optimize Gateway estimation uncertainty. There are no actuator execution errors or unmodeled forces in this closed-loop generator. Meeting the Gateway tracking requirement remains to be demonstrated over independently tested trajectories.

The notebook also includes the governing equations for the CR3BP dynamics, optical measurement model, tracking errors, impulsive burns, STM-weighted guidance and lagged NARX inputs, along with descriptions of what the actual plots establish.

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
