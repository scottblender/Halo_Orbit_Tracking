# Halo observer station-keeping: STM, gradient boosting and neural NARX

[Notebook 00](notebooks/00_gateway_tracking_comparison.ipynb) retains its original
motivation, tracking analyses, embedded figures and outputs byte for byte.
[Notebook 01](notebooks/01_narx_results.ipynb) is the complete executable neural
NARX experiment, with saved results and figures.

Notebook 01 regenerates independent STM training/validation/test trajectories
(20/5/5 arcs), trains the original gradient-boosted baseline and a genuine
`narxpy==0.1.0` network, reports held-out teacher-forced and recursive metrics,
and executes five paired 365-day nonlinear CR3BP trials for four controllers:
uncontrolled, STM, gradient-boosted, and neural NARX. No learned controller uses
an STM fallback. The original baseline implementation remains available in
`two_stage_narx.py`; the historical `narx` trajectory filename denotes that baseline.

The neural architecture has four input and four output feedback delays,
11 exogenous variables, 64 tanh hidden units and eight linear command outputs.
Training uses past teacher commands; deployment feeds back past neural predictions.
Inputs use estimated MCI departure in km/m/s and navigation context; burn outputs
use rotating-frame m/s and one of four nonnegative delays. See Notebook 01 for
the equations, indexing convention, warmup, scaling, decoding and limitations.
`narxpy` was installed and used directly; no compatibility fallback was needed.

## Reproduce

Use Python 3.10 or later with wheels supported by PyTorch. The verified local
runtime and package versions are saved with the experiment.

```sh
python -m pip install -e ".[dev,neural-narx]"
python -m pytest -q
python scripts/check_notebooks.py --one notebooks/01_narx_results.ipynb
```

The notebook runner uses an isolated IPython process and embeds PNG outputs.
Its default selection is Notebook 01 only. It never rewrites Notebook 00 unless
that file is explicitly selected. On Windows, use a short checkout path if
package installation encounters the Windows path-length limit. Set temporary,
Matplotlib and IPython directories to writable paths if required by the host.

For a command-line study without notebook figures:

```sh
python scripts/run_one_year_study.py --days 365 --arcs 5 --output results/one_year
```

`--measurement-cadence-s` and `--decisions-per-period` also configure newly
generated teacher data. Reused datasets must match the requested configuration;
the audit rejects mismatches. Annual decisions use exactly the teacher spacing,
with an evaluation-only endpoint at day 365.

## Verified run

The saved run trained the neural network for 600 epochs (validation selected
epoch 110) and executed all 20 controller episodes across five paired annual
trials. STM completed 365 days in 5/5 trials; uncontrolled, gradient-boosted and
neural NARX completed 0/5, all stopping at the departure cutoff. No controller
maintained the full-year corridor. Recursive held-out neural feasible-burn RMSE
was 0.304 m/s, with 11.1% feasible-command recall. These results demonstrate
execution, not successful learned station-keeping. The test suite passed all
44 tests; saved checkpoint replay reproduced 213 recorded online neural outputs.

## Saved evidence

[artifacts/neural_narx](artifacts/neural_narx/) contains the independently
inspectable training splits, training-loss history, selected neural weights,
baseline model, held-out metrics, consistency audit, annual trajectory NPZ files,
paired initial conditions and complete noise arrays, per-arc CSV metrics,
summary JSON, configuration, runtime versions, and PNG plots. Notebook outputs
also embed the figures for GitHub previews. The audit independently replays every
feasible STM target through nonlinear dynamics and verifies causal histories,
unit conversions, delay membership, and agreement with executed teacher burns.

Training and deployment use the same corrected periodic observer reference,
closure-validated prescribed Gateway reference, six-hour optical cadence plus
decision epochs, and eight decisions per observer period. Controller alternatives
share initial physical/navigation errors and epoch-indexed optical noise;
geometry-dependent measurement acceptance can differ. Gateway never receives burns.

Completing 365 days does not imply compliance with the 100 km/1 m/s corridor.
All unevaluated epochs after termination count as unsuccessful. Failed-arc RMSE
and delta-v cover only simulated time and must not be called full-year accuracy
or fuel savings. See `arc_metrics.csv` for termination reasons and valid counts.
The experiment is an idealized CR3BP/navigation study with a small ensemble,
not an operational ephemeris or reliability certification. See the final
Notebook 01 discussion for measured differences between STM and learned NARX.
