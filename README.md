# One-year halo observer NARX station-keeping and Lunar Gateway tracking

This repository studies 365-day station-keeping for an Earth–Moon CR3BP halo observer
using LSO optical measurements and a coupled navigation EKF, while estimating
Lunar Gateway on a prescribed nominal 9:2 NRHO.

## Notebooks — start here

- [00 — One-year orbital references and LSO measurements](notebooks/00_one_year_problem_setup.ipynb):
  periodicity checks for the corrected observer halo and Gateway, plus one-year
  nominal optical visibility.
- [01 — One-year STM vs NARX results](notebooks/01_narx_results.ipynb):
  short-arc teacher training, held-out offline NARX scores, actual 365-day
  uncontrolled/STM/NARX rollouts, observer departure, cumulative Δv,
  and Gateway tracking plots.

The obsolete short-duration cumulative tracking notebook is removed from the
current checkout and remains recoverable in Git history.

## Model and comparison

1. **Observer reference:** differential-corrected periodic CR3BP halo.
2. **Gateway target:** a supplied nominal 9:2 NRHO state is numerically checked
   for periodic closure within the same CR3BP, then repeated as a prescribed
   truth trajectory. Gateway receives **zero station-keeping commands**.
   This does not validate its operational ephemeris or independently prove
   the 9:2 family designation.
3. **Measurements:** three LSOs supply simulated noisy optical angles to the
   joint EKF. Observer-to-Gateway angles are accepted when visible.
4. **STM control:** computes and executes an observer-only nonlinear-checked
   state-transition-matrix maneuver from the estimated observer state.
5. **NARX control:** a supervised, gradient-boosted *NARX-inspired* predictor
   executes predicted observer burn decisions, Δv vectors and delays, with no
   STM fallback at deployment. It is not a recurrent NARX neural network.
6. **Fair paired trials:** uncontrolled/STM/NARX alternatives use identical
   initial physical/navigation errors and per-epoch simulated measurement-noise
   draws; geometry-dependent measurement acceptance can differ by controller.

Early numerical termination or excessive departure is counted as unsuccessful
for all remaining epochs. RMSE for a failed arc uses only simulated epochs and
must not be described as full-year tracking accuracy.

## Plots on GitHub

Static GitHub notebook previews show plots **only when their outputs are saved**.
The [one-year pilot workflow](https://github.com/scottblender/Halo_Orbit_Tracking/actions/workflows/one-year-pilot.yml)
generates real 365-day trajectories, executes both new notebooks, and saves
embedded figures in its artifact. For in-repository PR branches, the workflow
also attempts to commit these executed notebook outputs to the PR branch.
The Scientific Checks workflow saves executed notebook outputs on main.

To reproduce the one-pair 365-day pilot from the checkout:

```bash
python -m pip install -e ".[dev]"
python scripts/run_one_year_study.py --days 365 --arcs 1 --measurement-cadence-s 86400 --decisions-per-period 4
python scripts/check_notebooks.py
```

For five paired one-year test arcs and plots that use the same data:

```bash
python scripts/run_one_year_study.py --days 365 --arcs 5
HALO_YEAR_REUSE=1 HALO_YEAR_ARCS=5 HALO_YEAR_CADENCE_S=21600 HALO_YEAR_DECISIONS=8 python scripts/check_notebooks.py
```

On Windows PowerShell, set the displayed environment variables with
`$env:VARIABLE='value'` before calling the notebook checker.

Machine-readable annual results are saved in `results/one_year/` as
`summary.json`, `arc_metrics.csv`, and per-controller NPZ histories, with
the observer learned-policy bundle and PNG plots.

## Scientific interpretation

A policy **completing 365 days** may still violate its allowable
position/velocity departure corridor during the year. Report full-year
completion separately from corridor compliance. Compare paired annual Δv,
maneuver counts, first corridor exit, and Gateway tracking/visibility.
A one-pair pilot demonstrates numerical behavior, not a reliable population
success probability; larger independent training and test ensembles and
higher-fidelity dynamics/measurements are future validation priorities.
