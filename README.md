# LSO-informed NRHO station keeping

**Question:** Can angular data from three lunar surface observatories (LSOs)
support identification and impulse magnitude/phase planning that keeps an
NRHO spacecraft within **10 km** in position and **0.1 m/s** in velocity of its
nominal orbit?

| Notebook | Purpose |
| --- | --- |
| [01 — Data and identification](notebooks/01_stationkeeping_identification.ipynb) | LSO-derived directional input/output histories, raw angles and visibility, feedback block diagram, Welch/FRF, stationarity, phase dependence, nonlinearity, stability and up to 24 orbits of uncontrolled propagation |
| [02 — STM grey box](notebooks/02_stm_greybox_stationkeeping.ipynb) | Raw angular-innovation identification, held-out validation, gap-aware residual/stationarity tests, and LSO-guided station keeping, summarized in three figures |

Each figure includes one purpose sentence and one result sentence computed from
the current run. Orbit projections replace the overlapping 3-D axes. Conclusions
distinguish the nonlinear autonomous CR3BP dynamics from their phase-dependent
local STM and time-dependent observation geometry.

The active workflow uses `lso_stationkeeping.py`, `lso_diagnostics.py` and
`lso_report_plots.py`. It reuses this repository's LSO geometry, angular
measurements, elevation/range/keep-out masks, dynamics and differential correction.
The grey-box estimator fits **raw visible angular innovations**, not filtered
states treated as independent measurements. Guidance receives an LSO-derived
estimate. Truth is used only for sensor simulation and evaluation. Missing angles
remain missing, and estimates/covariances coast through visibility gaps.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_stationkeeping.py tests/test_lso_stationkeeping.py
python scripts/check_notebooks.py --in-process
```

Run Notebook 01 then Notebook 02, or run either independently from the repository
root. Notebook 02 rebuilds missing/stale LSO data. The notebook builder is
`scripts/build_stationkeeping_notebooks.py`. Configuration is in
`configs/stationkeeping.json`. Noise, sites and optical gates come from
`configs/baseline.json`. Uniform angular sampling is declared in the experiment
metadata (one epoch per decision, approximately 1.5 hours), rather than silently
inheriting the older tracking demonstration's cadence.

Saved results under `artifacts/stationkeeping/` include raw angular streams,
visibility, navigation covariance, independent split seeds/checksums, fitted
parameters, predictions, residual tests and true corridor histories. Full-orbit
Welch/FRF curves include the navigation filter and are exploratory. They are not
claimed as the physical plant's global LTI response. Missing observations are
never interpolated for raw-innovation tests. NARX remains deferred.

Historical full-state helpers in `stationkeeping.py` support isolated numerical
tests only. The active notebooks use the LSO workflow. Older Gateway/NARX modules
remain historical. Sampled controller performance is reported honestly, including
failures and propagation cutoffs, separately from successful code execution.
