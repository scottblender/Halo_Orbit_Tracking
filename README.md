# Halo-orbit station-keeping and Lunar Gateway tracking

This repository uses two **complementary research notebooks**. Notebook 00 preserves the original tracking, observability, geometry, and unstable-orbit analyses that motivate why a halo observer needs station-keeping. Notebook 01 is the separate **one-year NARX station-keeping experiment** and contains annual data generation, controller comparisons, and results.

## Read the notebooks in order

**[00 — Gateway tracking comparison and problem motivation](notebooks/00_gateway_tracking_comparison.ipynb)**

The original 26-cell notebook is restored **with all existing explanations, figures, plots, executed outputs, and diagnostic visualizations**. It establishes the measurement model and three-lunar-surface-observatory (LSO) network; compares perfect versus LSO-estimated observer knowledge for Gateway tracking; shows observer/Gateway trajectories and estimation errors; examines reference-orbit differential correction and stability; visualizes uncontrolled observer departure over successive periods; and studies tracking during observer-to-Gateway optical blackouts. Its uncontrolled observer is deliberately a **motivational baseline**, not the NARX controller.

**[01 — One-year NARX station-keeping results](notebooks/01_narx_results.ipynb)**

All **365-day simulation generation, training, STM-teacher / learned-policy comparison, cumulative observer Δv, corridor compliance, and Gateway tracking results** belong here. The three observer alternatives are uncontrolled, STM-guided, and NARX-guided. The last case executes model-predicted impulses rather than borrowing STM corrections.

The newer nominal one-year periodic-reference/LSO visibility illustration is also **preserved**, as an optional [supplement](docs/one_year_nominal_geometry_supplement.ipynb) rather than being substituted for Notebook 00.

## One-year model assumptions

- **Halo observer:** corrected periodic Earth–Moon CR3BP halo reference, observer-only executed station-keeping burns.
- **Gateway:** prescribed nominal 9:2 NRHO with strict CR3BP periodic-closure check; Gateway is tracked but never commanded to station-keep. Periodic closure is not a high-fidelity operational ephemeris validation.
- **Measurements and EKF:** simulated noisy LSO optical angles establish the halo observer's navigation estimate, with observer-to-Gateway optical angles tracking Gateway when visibility permits. Guidance is based on **estimated** states, not simulated truth.
- **Control baseline:** STM corrections and nonlinear feasibility checks on the estimated observer state.
- **Learned alternative:** gradient-boosted NARX-inspired maneuver classifier, three-component Δv regression, and burn-delay classifier, trained on separate STM-teacher episodes.
- **Fair evaluation:** identical initial observer physical errors, initial EKF errors, and pre-sampled per-epoch noise across paired controller alternatives. Actual visibility can differ as their trajectories diverge.

## Regenerate the annual experiment and the plots

Install the development dependencies and run the annual generator:

```bash
python -m pip install -e ".[dev]"
python scripts/run_one_year_study.py --days 365 --arcs 5
```

Then execute **only Notebook 01**, reusing that dataset. Example for bash:

```bash
HALO_YEAR_REUSE=1 HALO_YEAR_ARCS=5 HALO_YEAR_CADENCE_S=21600 HALO_YEAR_DECISIONS=8 python scripts/check_notebooks.py --one notebooks/01_narx_results.ipynb
```

On Windows PowerShell, set the corresponding variables with `$env:HALO_YEAR_REUSE='1'`, etc., before running the Python command. The annual workflow's single-pair pilot instead uses 24-hour optical cadence and 4 decision intervals per orbit.

**Notebook 00 is intentionally not re-executed by the annual workflow.** This prevents its original explanatory figures from being replaced by shortened CI-run outputs. The [year-long workflow](https://github.com/scottblender/Halo_Orbit_Tracking/actions/workflows/one-year-pilot.yml) executes and embeds plots **only in Notebook 01**; all raw annual NPZ/CSV/JSON outputs remain available as workflow artifacts.

## Interpretation

Reaching day 365 does **not** necessarily mean meeting the observer's departure corridor continuously. The first one-pair pilot demonstrated inadequate learned NARX station-keeping; the STM case also had corridor violations. Treat the first run as a baseline, not a successful controller or a statistically reliable mission success rate. Gateway RMSE on an early-terminated trajectory covers only simulated epochs.
