# NRHO station keeping: system identification and STM grey box

**Question:** Can a phase-dependent state-space model identify the response to
small impulses and use it to choose impulse magnitude and orbital phase while
keeping an NRHO observer within **10 km** in position and **0.1 m/s** in velocity
of its nominal orbit?

| Notebook | Purpose |
| --- | --- |
| [01 — Identification](notebooks/01_stationkeeping_identification.ipynb) | Directional input/output plots, Welch PSDs, MIMO FRF and coherence, stationarity, excitation/coupling/phase diagnostics, linearization range, stability and uncontrolled degradation |
| [02 — STM grey box](notebooks/02_stm_greybox_stationkeeping.ipynb) | Periodic state-space model, actuator-gain PEM, held-out prediction/free simulation, innovation whiteness/input correlation/stationarity, autonomous impulse magnitude/phase planning and nonlinear corridor checks |

Run Notebook 01 then Notebook 02. Both use reusable functions in
`src/halo_tracking/stationkeeping*.py`; explanations stay short and figures use
large, bold type. Notebook 02 can independently rebuild missing identification
data. The [course-method map](docs/course_alignment.md) documents adaptations
from the supplied Studios 3–13, frame/impulse conventions, diagnostic assumptions
and interpretation limits. Class notebooks are not redistributed.

## Reproduce

```sh
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_stationkeeping.py
python scripts/check_notebooks.py --in-process
```

Run from the repository root. Alternatively open both notebooks in JupyterLab
and Run All. Their builder is `scripts/build_stationkeeping_notebooks.py`.
Configuration lives in `configs/stationkeeping.json`; `baseline.json` supplies
only the Earth–Moon dynamics/orbit seed. The active experiment does not inherit
the old teacher tolerances or tracking controller settings.

The nonlinear simulator generates independent open-loop commanded impulses and
synthetic full-state navigation measurements. Training/validation/test splits
use distinct trajectory seeds. The grey box identifies three unknown actuator
gains; it never fits against saved truth or reference burn labels. The nonlinear
closed loop chooses its own candidate burn times and compares true 3-D departure
norms against the required corridor, including pre/post impulse samples.

Outputs under `artifacts/stationkeeping/` include checksummed data, configuration,
seeds, figures, gain estimates, held-out predictions, stationarity reports,
residual tests, selected burn phases and nonlinear corridor histories. The CI
artifact contains both executed notebooks and these results, even if a numerical
check fails. The notebook prints an observed corridor failure explicitly; a
single sampled simulation is not a continuous-time or Monte Carlo guarantee.

## Scope

The nonlinear dynamics are autonomous; their local departure model is linear
time periodic. Whole-orbit Welch curves are exploratory finite-record summaries,
not an assumed global LTI transfer function. Whiteness assesses the noise model;
response agreement and held-out errors assess dynamics. Phase and perturbation
diagnostics inform the next model discussion. **NARX is deferred.**

This workflow focuses on the observer's own station keeping. Prior Gateway,
optical tracking and NARX modules/artifacts remain historical research code;
their notebooks and active NARX workflow are retired. Historical results remain
available in Git history. No Gateway or tracking-performance claim is made by
the two active notebooks.
